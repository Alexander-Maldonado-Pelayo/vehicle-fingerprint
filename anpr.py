"""
All-in-one Flock-style vehicle pipeline: detect + track + fingerprint + speed + log.
Works on an image or a video.  Needs: ultralytics torch torchvision opencv-python numpy
  python anpr.py photo.jpg          # image -> output.jpg
  python anpr.py clip.mp4           # video -> output.mp4 + vehicles.csv
  python anpr.py 0                  # webcam
Requires a trained fingerprint.pth (train separately with fingerprint_model.py).
"""
import sys, csv, os
from collections import defaultdict, deque
import numpy as np, cv2, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ---- config ------------------------------------------------------------
ATTRS = ["make", "model", "color", "type"]
VEHICLE_IDS = {2, 3, 5, 7}          # car, motorcycle, bus, truck (COCO)
EVERY_N, MIN_CONF, WINDOW, MAX_MISSING = 5, 0.35, 10, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- calibration: lane-based, one bird's-eye homography per carriageway ----
LANE_WIDTH_M = 3.75                  # metres per lane
NUM_LANES_LEFT, NUM_LANES_RIGHT = 5, 3
BEV_SCALE = 18                       # px per metre in the bird's-eye view
VISIBLE_LENGTH_M = 70                # length of road the road-quads span
# 4 road-plane points per carriageway: far-left, far-right, near-right, near-left
SRC_ROAD_L = np.float32([[155, 415], [600, 395], [845, 1079], [0, 1079]])
SRC_ROAD_R = np.float32([[845, 395], [1050, 415], [1080, 1079], [845, 1079]])

def make_H(src, n_lanes):
    w = n_lanes * LANE_WIDTH_M * BEV_SCALE       # BEV width  in px
    h = VISIBLE_LENGTH_M * BEV_SCALE             # BEV length in px
    return cv2.getPerspectiveTransform(src, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))

REGIONS = [(SRC_ROAD_L, make_H(SRC_ROAD_L, NUM_LANES_LEFT)),
           (SRC_ROAD_R, make_H(SRC_ROAD_R, NUM_LANES_RIGHT))]

_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
COLORS = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 200), (0, 128, 255)]


# ---- multi-task fingerprint model --------------------------------------
class VehicleFingerprint(nn.Module):
    def __init__(self, head_sizes):
        super().__init__()
        bb = models.resnet50(); self.dim = bb.fc.in_features; bb.fc = nn.Identity()
        self.backbone = bb
        self.heads = nn.ModuleDict({a: nn.Linear(self.dim, n) for a, n in head_sizes.items()})
    def forward(self, x):
        f = self.backbone(x); return {a: h(f) for a, h in self.heads.items()}


class Fingerprinter:
    def __init__(self, weights="fingerprint.pth"):
        ck = torch.load(weights, map_location=DEVICE)
        self.vocabs = ck["vocabs"]
        self.model = VehicleFingerprint(ck["head_sizes"]).to(DEVICE)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
    @torch.no_grad()
    def __call__(self, bgr):
        x = _tf(Image.fromarray(bgr[:, :, ::-1])).unsqueeze(0).to(DEVICE)
        out = self.model(x); res = {}
        for a in ATTRS:
            p = out[a].softmax(1)[0]; c, i = p.max(0)
            res[a] = (self.vocabs[a][i], c.item())
        return res


def to_world(px, py):
    # find which carriageway the ground point sits in, map with its homography.
    # BEV result is in px -> divide by BEV_SCALE to get metres. None = off-road.
    for src, Hm in REGIONS:
        if cv2.pointPolygonTest(src, (float(px), float(py)), False) >= 0:
            wx, wy = cv2.perspectiveTransform(np.array([[[px, py]]], np.float32), Hm)[0][0]
            return wx / BEV_SCALE, wy / BEV_SCALE
    return None


# ---- per-vehicle accumulator -------------------------------------------
class Track:
    def __init__(self, tid, f): self.id = tid; self.first = self.last = f
    votes = None
    def init(self, f):
        self.votes = {a: defaultdict(float) for a in ATTRS}
        self.hist = deque(maxlen=WINDOW); self.speeds = []
    def vote(self, fp):
        for a in ATTRS:
            lbl, c = fp[a]
            if c >= MIN_CONF: self.votes[a][lbl] += c
    def best(self):
        return {a: (max(v, key=v.get) if v else "?") for a, v in self.votes.items()}
    def add_pos(self, f, w, fps):
        self.last = f
        if w is None: return                 # vehicle outside the calibrated road
        self.hist.append((f, *w))
        if len(self.hist) >= 2:
            (f0, x0, y0), (f1, x1, y1) = self.hist[0], self.hist[-1]
            dt = (f1 - f0) / fps
            if dt > 0: self.speeds.append(np.hypot(x1 - x0, y1 - y0) / dt * 3.6)
    def speed(self): return self.speeds[-1] if self.speeds else None
    def row(self, fps):
        b = self.best()
        return {"id": self.id, "first_s": round(self.first / fps, 2),
            "last_s": round(self.last / fps, 2), **b,
            "avg_kmh": round(float(np.mean(self.speeds)), 1) if self.speeds else 0,
            "max_kmh": round(float(np.max(self.speeds)), 1) if self.speeds else 0}


def draw(img, box, tid, attrs, spd):
    x1, y1, x2, y2 = box; col = COLORS[tid % len(COLORS)]
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    s = f"({spd:.0f} km/h)" if spd is not None else ""
    if attrs:                        # trained fingerprint available -> full panel
        lines = [f"ID {tid} {s}".strip()] + [f"{a.capitalize()}: {attrs[a]}" for a in ATTRS]
    else:                            # no fingerprint.pth -> just Car + speed
        lines = [f"Car {s}".strip()]
    lh = 18; bh = lh * len(lines) + 6
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0] for t in lines) + 6
    top = y1 - bh if y1 - bh > 0 else y2
    cv2.rectangle(img, (x1, top), (x1 + tw, top + bh), col, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 3, top + lh * (i + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)


# ---- main --------------------------------------------------------------
def main(source):
    det = YOLO("yolov8n.pt")
    # fingerprint is OPTIONAL: without a trained model, boxes are labeled "Car"
    if os.path.exists("fingerprint.pth"):
        fp = Fingerprinter()
    else:
        fp = None
        print("[note] fingerprint.pth not found -> running detection + speed only "
              "(labels: 'Car'). Train it with fingerprint_model.py for make/model.")
    is_img = str(source).lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_img:                                    # ---- single image ----
        img = cv2.imread(source)
        for b in det(img, conf=.35, verbose=False)[0].boxes:
            if int(b.cls[0]) not in VEHICLE_IDS: continue
            x1, y1, x2, y2 = map(int, b.xyxy[0]); crop = img[y1:y2, x1:x2]
            if not crop.size: continue
            attrs = None
            if fp is not None:
                f = fp(crop)
                attrs = {a: (l if c >= MIN_CONF else "?") for a, (l, c) in f.items()}
            draw(img, (x1, y1, x2, y2), 0, attrs, None)
        cv2.imwrite("output.jpg", img); print("saved -> output.jpg"); return

    src = int(source) if str(source).isdigit() else source   # ---- video ----
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    live = isinstance(src, int)
    # write real H.264 via imageio-ffmpeg (cv2's mp4v/H264 encoders are broken on
    # this build -> unplayable files). imageio wants RGB frames.
    import imageio
    out = None if live else imageio.get_writer("output.mp4", fps=fps, codec="libx264",
        macro_block_size=None, pixelformat="yuv420p", output_params=["-movflags", "+faststart"])
    log = open("vehicles.csv", "w", newline="")
    cw = csv.DictWriter(log, fieldnames=["id", "first_s", "last_s", *ATTRS, "avg_kmh", "max_kmh"])
    cw.writeheader()
    tracks, seen, fi = {}, {}, 0

    while True:
        ok, fr = cap.read()
        if not ok: break
        fi += 1
        r = det.track(fr, persist=True, conf=.35, classes=list(VEHICLE_IDS), verbose=False)[0]
        if r.boxes.id is not None:
            for box, tid in zip(r.boxes.xyxy, r.boxes.id):
                x1, y1, x2, y2 = map(int, box); tid = int(tid)
                if tid not in tracks:
                    tracks[tid] = Track(tid, fi); tracks[tid].init(fi)
                t = tracks[tid]; seen[tid] = fi
                t.add_pos(fi, to_world((x1 + x2) // 2, y2), fps)
                if fp is not None and (not t.speeds or fi % EVERY_N == 0):
                    crop = fr[y1:y2, x1:x2]
                    if crop.size: t.vote(fp(crop))
                attrs = t.best() if fp is not None else None
                draw(fr, (x1, y1, x2, y2), tid, attrs, t.speed())
        for tid in [i for i, f in seen.items() if fi - f > MAX_MISSING]:
            cw.writerow(tracks[tid].row(fps)); del tracks[tid], seen[tid]
        if live:
            cv2.imshow("anpr", fr)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        elif out is not None: out.append_data(fr[:, :, ::-1])   # BGR -> RGB
    for tid in list(tracks): cw.writerow(tracks[tid].row(fps))
    cap.release(); log.close()
    if out is not None: out.close()
    cv2.destroyAllWindows(); print("saved -> output.mp4 + vehicles.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "input.mp4")
