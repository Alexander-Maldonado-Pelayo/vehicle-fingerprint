"""
Flock-style vehicle fingerprinting on VIDEO with per-vehicle tracking.

  YOLO tracker  -> stable track IDs across frames
  Fingerprinter -> make/model/color/type per crop (every N frames)
  Temporal vote -> confidence-weighted accumulation so labels stay stable
  cv2.VideoWriter -> annotated output video

Run:
  python fingerprint_video.py input.mp4                 # -> output.mp4
  python fingerprint_video.py 0                         # webcam, live window
"""
import sys
from collections import defaultdict
import cv2
from ultralytics import YOLO
from fingerprint_model import Fingerprinter, ATTRS

VEHICLE_IDS = {2, 3, 5, 7}       # car, motorcycle, bus, truck (COCO)
EVERY_N = 5                       # re-fingerprint a track every N frames
MIN_CONF = 0.35

detector = YOLO("yolov8n.pt")
fingerprint = Fingerprinter("fingerprint.pth")


class TrackMemory:
    """Accumulate confidence-weighted votes per track id, per attribute."""
    def __init__(self):
        # votes[track_id][attr][label] = summed confidence
        self.votes = defaultdict(lambda: {a: defaultdict(float) for a in ATTRS})

    def update(self, tid, fp):
        for a in ATTRS:
            label, conf = fp[a]
            self.votes[tid][a][label] += conf

    def best(self, tid):
        """Return {attr: label} using the highest-voted label so far."""
        out = {}
        for a in ATTRS:
            tally = self.votes[tid][a]
            out[a] = max(tally, key=tally.get) if tally else "?"
        return out

    def seen(self, tid):
        return tid in self.votes


COLORS = [(0, 255, 0), (255, 128, 0), (0, 200, 255),
          (255, 0, 200), (0, 128, 255), (200, 255, 0)]


def draw(img, box, tid, attrs):
    x1, y1, x2, y2 = box
    color = COLORS[tid % len(COLORS)]        # consistent color per track
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    lines = [f"ID {tid}"] + [f"{a.capitalize()}: {attrs[a]}" for a in ATTRS]
    font, scale, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 18
    bh = lh * len(lines) + 6
    tw = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines) + 6
    top = y1 - bh if y1 - bh > 0 else y2
    cv2.rectangle(img, (x1, top), (x1 + tw, top + bh), color, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 3, top + lh * (i + 1) - 4),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def run(source, out="output.mp4"):
    # source: path string, or "0" for webcam
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

    memory = TrackMemory()
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        # persist=True keeps track ids stable across calls (ByteTrack)
        results = detector.track(frame, persist=True, conf=0.35,
                                 classes=list(VEHICLE_IDS), verbose=False)[0]

        if results.boxes.id is not None:
            for box, tid in zip(results.boxes.xyxy, results.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)

                # fingerprint on first sighting, then every N frames
                if not memory.seen(tid) or frame_i % EVERY_N == 0:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        fp = fingerprint(crop)
                        # only count confident votes toward the tally
                        fp = {a: (lbl if c >= MIN_CONF else "?", c)
                              for a, (lbl, c) in fp.items()}
                        memory.update(tid, fp)

                draw(frame, (x1, y1, x2, y2), tid, memory.best(tid))

        if live:
            cv2.imshow("fingerprint", frame)
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
