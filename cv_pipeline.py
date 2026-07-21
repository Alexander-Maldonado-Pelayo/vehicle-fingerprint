"""
Full pipeline using ONLY OpenCV (cv2.dnn) for inference. No torch / ultralytics.

Stage 1: YOLO object detection via cv2.dnn        -> vehicle boxes
Stage 2: ONNX make/model classifier via cv2.dnn   -> "Make: X  Model: Y"

You need two model files:
  1. yolov8n.onnx   -- export once with: `yolo export model=yolov8n.pt format=onnx`
                       (or download any YOLO .onnx). COCO 80-class model.
  2. car_model.onnx -- your make/model classifier exported to ONNX, plus a
                       classes.txt (one 'Make_Model' name per line, in order).

Run: python cv_pipeline.py image.jpg
"""
import sys
import cv2
import numpy as np

# ---- config ------------------------------------------------------------
YOLO_ONNX   = "yolov8n.onnx"
CLS_ONNX    = "car_model.onnx"
CLASSES_TXT = "classes.txt"
YOLO_SIZE   = 640            # YOLOv8 default input
CLS_SIZE    = 224            # classifier input
CONF, IOU   = 0.35, 0.45
CLS_CONF    = 0.35           # min classifier confidence to trust a label
VEHICLE_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck (COCO)

# ---- load networks -----------------------------------------------------
det_net = cv2.dnn.readNetFromONNX(YOLO_ONNX)
cls_net = cv2.dnn.readNetFromONNX(CLS_ONNX)
with open(CLASSES_TXT) as f:
    CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]

# Use CPU (default). For CUDA build of OpenCV, uncomment:
# det_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# det_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def detect_vehicles(img):
    """Run YOLOv8 through cv2.dnn, return list of (x1,y1,x2,y2)."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (YOLO_SIZE, YOLO_SIZE),
                                 swapRB=True, crop=False)
    det_net.setInput(blob)
    out = det_net.forward()                 # (1, 84, 8400) for YOLOv8
    out = out[0].T                          # -> (8400, 84): [cx,cy,w,h, 80 scores]

    boxes, scores, ids = [], [], []
    x_scale, y_scale = w / YOLO_SIZE, h / YOLO_SIZE
    for row in out:
        class_scores = row[4:]
        cid = int(np.argmax(class_scores))
        conf = float(class_scores[cid])
        if conf < CONF or cid not in VEHICLE_IDS:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = int((cx - bw / 2) * x_scale)
        y1 = int((cy - bh / 2) * y_scale)
        boxes.append([x1, y1, int(bw * x_scale), int(bh * y_scale)])
        scores.append(conf)
        ids.append(cid)

    # Non-max suppression (also pure OpenCV)
    keep = cv2.dnn.NMSBoxes(boxes, scores, CONF, IOU)
    result = []
    for i in np.array(keep).flatten():
        x, y, bw, bh = boxes[i]
        result.append((max(0, x), max(0, y), x + bw, y + bh))
    return result


def classify(crop):
    """Run the make/model classifier through cv2.dnn on a car crop."""
    blob = cv2.dnn.blobFromImage(crop, 1/255.0, (CLS_SIZE, CLS_SIZE),
                                 swapRB=True, crop=False)
    # ImageNet normalization (match how you trained the classifier)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    blob = (blob - mean) / std
    cls_net.setInput(blob.astype(np.float32))
    logits = cls_net.forward()[0]
    probs = np.exp(logits - logits.max()); probs /= probs.sum()   # softmax
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])


def draw(img, box, make, model):
    x1, y1, x2, y2 = box
    green = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)
    lines = [f"Make: {make}", f"Model: {model}"]
    lh = 18
    bh = lh * len(lines) + 6
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
             for t in lines) + 6
    cv2.rectangle(img, (x1, y1 - bh), (x1 + tw, y1), green, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 3, y1 - bh + lh * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def run(image_path, out="output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    for box in detect_vehicles(img):
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        label, conf = classify(crop)
        if conf < CLS_CONF:
            make, model = "Unknown", "?"
        else:
            make, _, model = label.partition("_")
            model = model.replace("_", " ") or "?"
        draw(img, box, make, model)

    cv2.imwrite(out, img)
    print(f"saved -> {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "cars.jpg")
