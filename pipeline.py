"""
Full pipeline: detect vehicles (YOLO) -> classify make/model -> draw labels.
Run: python pipeline.py path/to/image.jpg
Requires a trained car_model.pth (see classifier.py).
"""
import sys
import cv2
from ultralytics import YOLO
from classifier import CarClassifier

VEHICLE_CLASSES = {2, 3, 5, 7}          # car, motorcycle, bus, truck (COCO)
CONF_THRESH = 0.35                       # min classifier confidence to trust a label

detector = YOLO("yolov8n.pt")
classifier = CarClassifier("car_model.pth")


def draw_label(img, box, make, model):
    x1, y1, x2, y2 = map(int, box)
    green = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)

    lines = [f"Make: {make}", f"Model: {model}"]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    lh = 18
    box_h = lh * len(lines) + 6
    w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines) + 6
    cv2.rectangle(img, (x1, y1 - box_h), (x1 + w, y1), green, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 3, y1 - box_h + lh * (i + 1) - 4),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def run(image_path, out="output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    for b in detector(img, conf=0.35)[0].boxes:
        if int(b.cls[0]) not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label, conf = classifier.predict(crop)          # e.g. "Volvo_V50"
        if conf < CONF_THRESH:
            make, model = "Unknown", "?"
        else:
            # class folders named Make_Model  ->  split on first underscore
            make, _, model = label.partition("_")
            model = model.replace("_", " ") or "?"

        draw_label(img, (x1, y1, x2, y2), make, model)

    cv2.imwrite(out, img)
    print(f"saved -> {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "cars.jpg")
