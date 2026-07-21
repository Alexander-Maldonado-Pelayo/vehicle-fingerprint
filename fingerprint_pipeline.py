"""
Flock-style vehicle fingerprint pipeline.
  detect vehicles (YOLO) -> fingerprint each (make/model/color/type) -> draw panel

Run: python fingerprint_pipeline.py image.jpg
Needs: yolov8n.pt (auto-downloads) and a trained fingerprint.pth.
"""
import sys
import cv2
from ultralytics import YOLO
from fingerprint_model import Fingerprinter, ATTRS

VEHICLE_IDS = {2, 3, 5, 7}
MIN_CONF = 0.35          # below this, show attribute as '?'

detector = YOLO("yolov8n.pt")
fingerprint = Fingerprinter("fingerprint.pth")


def draw_panel(img, box, fp):
    """fp = {'make': (label, conf), ...}. Draw a green box + attribute list."""
    x1, y1, x2, y2 = box
    green = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)

    lines = []
    for a in ATTRS:
        label, conf = fp[a]
        shown = label if conf >= MIN_CONF else "?"
        lines.append(f"{a.capitalize()}: {shown}")

    font, scale, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 18
    bh = lh * len(lines) + 6
    tw = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines) + 6
    # keep panel on-screen if the box is near the top
    top = y1 - bh if y1 - bh > 0 else y2
    cv2.rectangle(img, (x1, top), (x1 + tw, top + bh), green, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 3, top + lh * (i + 1) - 4),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def run(image_path, out="output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    for b in detector(img, conf=0.35)[0].boxes:
        if int(b.cls[0]) not in VEHICLE_IDS:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        fp = fingerprint(crop)                 # the "fingerprint": all attributes
        draw_panel(img, (x1, y1, x2, y2), fp)

    cv2.imwrite(out, img)
    print(f"saved -> {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "cars.jpg")
