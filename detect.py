"""
Stage 1: Vehicle detection with YOLOv8 (pretrained on COCO).
Draws green boxes + labels like the reference image.
Run: python detect.py path/to/image.jpg
"""
import sys
import cv2
from ultralytics import YOLO

# COCO class ids we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# 'n' = nano (fast), swap for yolov8m.pt / yolov8x.pt for accuracy
model = YOLO("yolov8n.pt")


def draw_label(img, box, text):
    x1, y1, x2, y2 = map(int, box)
    green = (0, 255, 0)
    # bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)
    # label background + text (above the box)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), green, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def detect(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    results = model(img, conf=0.35)[0]

    crops = []
    for b in results.boxes:
        cls = int(b.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crops.append(((x1, y1, x2, y2), img[y1:y2, x1:x2].copy()))

        # For now the label is just the COCO class.
        # In the full pipeline you'd replace this with the make/model prediction.
        label = VEHICLE_CLASSES[cls]
        draw_label(img, (x1, y1, x2, y2), label)

    out = "output.jpg"
    cv2.imwrite(out, img)
    print(f"Detected {len(crops)} vehicles -> {out}")
    return crops


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "cars.jpg"
    detect(path)
