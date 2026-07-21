"""
Flock-style video pipeline: detect + track + fingerprint + speed + CSV LOG.

Produces:
  output.mp4   -- annotated video
  vehicles.csv -- one row per unique vehicle:
      id, first_seen_s, last_seen_s, duration_s, frames,
      make, model, color, type, avg_speed_kmh, max_speed_kmh

A track is "closed" and written when it hasn't been seen for MAX_MISSING
frames (it left the scene); any still-open tracks are flushed at the end.

Calibrate first with calibrate.py; paste SRC_POINTS / REAL_W / REAL_H below.
Run: python fingerprint_video_log.py input.mp4
"""
import sys
import csv
from collections import defaultdict, deque
import numpy as np
import cv2
from ultralytics import YOLO
from fingerprint_model import Fingerprinter, ATTRS

# ---- calibration (from calibrate.py) -----------------------------------
SRC_POINTS = [(580, 220), (760, 220), (960, 500), (300, 500)]
REAL_W = 3.5
REAL_H = 24.0

# ---- params ------------------------------------------------------------
VEHICLE_IDS = {2, 3, 5, 7}
EVERY_N = 5
MIN_CONF = 0.35
SPEED_WINDOW = 10
MAX_MISSING = 30        # frames a track can vanish before we close & log it
KMH = 3.6

detector = YOLO("yolov8n.pt")
fingerprint = Fingerprinter("fingerprint.pth")

_dst = np.float32([[0, 0], [REAL_W, 0], [REAL_W, REAL_H], [0, REAL_H]])
H = cv2.getPerspectiveTransform(np.float32(SRC_POINTS), _dst)


def to_world(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    wx, wy = cv2.perspectiveTransform(pt, H)[0][0]
    return float(wx), float(wy)


class Track:
    """All accumulated state for one vehicle."""
    def __init__(self, tid, frame_i):
        self.id = tid
        self.first = self.last = frame_i
        self.votes = {a: defaultdict(float) for a in ATTRS}
        self.hist = deque(maxlen=SPEED_WINDOW)   # (frame_i, wx, wy)
        self.speeds = []                         # sampled km/h readings

    def vote(self, fp):
        for a in ATTRS:
            label, conf = fp[a]
            self.votes[a][label] += conf

    def best(self):
        out = {}
        for a in ATTRS:
            t = self.votes[a]
            out[a] = max(t, key=t.get) if t else "?"
        return out

    def add_pos(self, frame_i, world, fps):
        self.last = frame_i
        self.hist.append((frame_i, world[0], world[1]))
        if len(self.hist) >= 2:
            (f0, x0, y0), (f1, x1, y1) = self.hist[0], self.hist[-1]
            dt = (f1 - f0) / fps
            if dt > 0:
                self.speeds.append(np.hypot(x1 - x0, y1 - y0) / dt * KMH)

    def current_speed(self):
        return self.speeds[-1] if self.speeds else None

    def row(self, fps):
        b = self.best()
        avg = float(np.mean(self.speeds)) if self.speeds else 0.0
        mx = float(np.max(self.speeds)) if self.speeds else 0.0
        return {
            "id": self.id,
            "first_seen_s": round(self.first / fps, 2),
            "last_seen_s": round(self.last / fps, 2),
            "duration_s": round((self.last - self.first) / fps, 2),
            "frames": self.last - self.first + 1,
            "make": b["make"], "model": b["model"],
            "color": b["color"], "type": b["type"],
            "avg_speed_kmh": round(avg, 1),
            "max_speed_kmh": round(mx, 1),
        }


CSV_FIELDS = ["id", "first_seen_s", "last_seen_s", "duration_s", "frames",
              "make", "model", "color", "type", "avg_speed_kmh", "max_speed_kmh"]

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


def run(source, out="output.mp4", log="vehicles.csv"):
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

    log_file = open(log, "w", newline="")
    csv_w = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
    csv_w.writeheader()

    tracks = {}             # tid -> Track (currently active)
    last_seen = {}          # tid -> last frame index seen
    frame_i = 0
    logged = 0

    def close_track(tid):
        nonlocal logged
        csv_w.writerow(tracks[tid].row(fps))
        logged += 1
        del tracks[tid]
        del last_seen[tid]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        results = detector.track(frame, persist=True, conf=0.35,
                                 classes=list(VEHICLE_IDS), verbose=False)[0]

        present = set()
        if results.boxes.id is not None:
            for box, tid in zip(results.boxes.xyxy, results.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)
                present.add(tid)

                if tid not in tracks:
                    tracks[tid] = Track(tid, frame_i)
                t = tracks[tid]
                last_seen[tid] = frame_i

                gx, gy = (x1 + x2) // 2, y2
                t.add_pos(frame_i, to_world(gx, gy), fps)

                if len(t.speeds) == 0 or frame_i % EVERY_N == 0:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        fp = fingerprint(crop)
                        fp = {a: (lbl if c >= MIN_CONF else "?", c)
                              for a, (lbl, c) in fp.items()}
                        t.vote(fp)

                draw(frame, (x1, y1, x2, y2), tid, t.best(), t.current_speed())

        # close tracks that have been missing too long
        for tid in [i for i, f in last_seen.items()
                    if frame_i - f > MAX_MISSING]:
            close_track(tid)

        if live:
            cv2.imshow("fingerprint+speed+log", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            writer.write(frame)
            if frame_i % 30 == 0:
                print(f"frame {frame_i}  active={len(tracks)}  logged={logged}")

    # flush every track still open at end of video
    for tid in list(tracks):
        close_track(tid)

    cap.release()
    if writer:
        writer.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"logged {logged} vehicles -> {log}")
    if not live:
        print(f"annotated video -> {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "input.mp4")
