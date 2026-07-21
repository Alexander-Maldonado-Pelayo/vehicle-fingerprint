"""
Flock-style video pipeline + SPEED estimation via perspective homography.

  YOLO tracker  -> stable track ids
  Homography    -> map road pixels to real-world meters (bird's-eye)
  Speed         -> meters travelled / elapsed time, smoothed per track
  Fingerprinter -> make/model/color/type (temporal voting, every N frames)

Calibrate first with calibrate.py, paste SRC_POINTS / REAL_W / REAL_H below.

Run: python fingerprint_video_speed.py input.mp4      # -> output.mp4
"""
import sys
from collections import defaultdict, deque
import numpy as np
import cv2
from ultralytics import YOLO
from fingerprint_model import Fingerprinter, ATTRS

# ---- calibration (from calibrate.py) -----------------------------------
SRC_POINTS = [(580, 220), (760, 220), (960, 500), (300, 500)]  # TL,TR,BR,BL px
REAL_W = 3.5        # real width  of that rectangle, meters (e.g. one lane)
REAL_H = 24.0       # real length of that rectangle, meters (e.g. 2 dash cycles)

# ---- params ------------------------------------------------------------
VEHICLE_IDS = {2, 3, 5, 7}
EVERY_N = 5
MIN_CONF = 0.35
SPEED_WINDOW = 10        # frames of history used to compute speed
KMH = 3.6               # m/s -> km/h

detector = YOLO("yolov8n.pt")
fingerprint = Fingerprinter("fingerprint.pth")

# Build homography: image pixels -> top-down meters
_dst = np.float32([[0, 0], [REAL_W, 0], [REAL_W, REAL_H], [0, REAL_H]])
H = cv2.getPerspectiveTransform(np.float32(SRC_POINTS), _dst)


def to_world(px, py):
    """Map an image point (ground contact) to real-world meters."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    wx, wy = cv2.perspectiveTransform(pt, H)[0][0]
    return float(wx), float(wy)


class TrackMemory:
    def __init__(self):
        self.votes = defaultdict(lambda: {a: defaultdict(float) for a in ATTRS})
        # per-track history of (frame_index, world_x, world_y)
        self.hist = defaultdict(lambda: deque(maxlen=SPEED_WINDOW))

    def vote(self, tid, fp):
        for a in ATTRS:
            label, conf = fp[a]
            self.votes[tid][a][label] += conf

    def best(self, tid):
        out = {}
        for a in ATTRS:
            tally = self.votes[tid][a]
            out[a] = max(tally, key=tally.get) if tally else "?"
        return out

    def seen(self, tid):
        return tid in self.votes

    def add_pos(self, tid, frame_i, world):
        self.hist[tid].append((frame_i, world[0], world[1]))

    def speed_kmh(self, tid, fps):
        """Distance between oldest & newest world positions / elapsed time."""
        h = self.hist[tid]
        if len(h) < 2:
            return None
        (f0, x0, y0), (f1, x1, y1) = h[0], h[-1]
        dt = (f1 - f0) / fps
        if dt <= 0:
            return None
        dist = np.hypot(x1 - x0, y1 - y0)       # meters, on the ground plane
        return dist / dt * KMH


COLORS = [(0, 255, 0), (255, 128, 0), (0, 200, 255),
          (255, 0, 200), (0, 128, 255), (200, 255, 0)]


def draw(img, box, tid, attrs, speed):
    x1, y1, x2, y2 = box
    color = COLORS[tid % len(COLORS)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    spd = f"{speed:.0f} km/h" if speed is not None else "-- km/h"
    lines = [f"ID {tid}   {spd}"] + [f"{a.capitalize()}: {attrs[a]}" for a in ATTRS]
    font, scale, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 18
    bh = lh * len(lines) + 6
    tw = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines) + 6
    top = y1 - bh if y1 - bh > 0 else y2
    cv2.rectangle(img, (x1, top), (x1 + tw, top + bh), color, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 3, top + lh * (i + 1) - 4),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def run(source, out="output.mp4"):
    cap_src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    live = isinstance(cap_src, int)
    writer = None if live else cv2.VideoWriter(
        out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    mem = TrackMemory()
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        results = detector.track(frame, persist=True, conf=0.35,
                                 classes=list(VEHICLE_IDS), verbose=False)[0]

        if results.boxes.id is not None:
            for box, tid in zip(results.boxes.xyxy, results.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)

                # ground-contact point = bottom-center of the box
                gx, gy = (x1 + x2) // 2, y2
                mem.add_pos(tid, frame_i, to_world(gx, gy))

                if not mem.seen(tid) or frame_i % EVERY_N == 0:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        fp = fingerprint(crop)
                        fp = {a: (lbl if c >= MIN_CONF else "?", c)
                              for a, (lbl, c) in fp.items()}
                        mem.vote(tid, fp)

                speed = mem.speed_kmh(tid, fps)
                draw(frame, (x1, y1, x2, y2), tid, mem.best(tid), speed)

        if live:
            cv2.imshow("fingerprint+speed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            writer.write(frame)
            if frame_i % 30 == 0:
                print(f"frame {frame_i}")

    cap.release()
    if writer:
        writer.release()
        print(f"saved -> {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "input.mp4")
