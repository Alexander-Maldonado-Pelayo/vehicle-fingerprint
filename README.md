# 🚗 Vehicle Fingerprint — Traffic Analytics Pipeline

A computer-vision pipeline that **detects**, **tracks**, and **analyzes** vehicles in
images and video. Inspired by commercial systems like Flock Safety, it goes beyond
plain object detection to build a *fingerprint* of each vehicle — make, model, color,
and body type — while tracking it across frames, estimating its speed, and logging
every vehicle to CSV.

```
Video ─▶ Detect (YOLOv8) ─▶ Track (ByteTrack) ─▶ Fingerprint (ResNet-50, multi-task)
                                   │                        make / model / color / type
                                   ▼
                          Speed (homography) ─▶ CSV log (per vehicle)
```

> **Status:** detection · tracking · speed · logging all working end-to-end on real
> footage. Make/model recognition is optional and requires training a model (see below).

---

## ✨ Features

- **Vehicle detection** — YOLOv8 (COCO) finds cars, trucks, buses, motorcycles.
- **Multi-object tracking** — persistent IDs across frames via ByteTrack, so each
  vehicle is followed, not just re-detected.
- **Make / model / color / type fingerprinting** — a single multi-task ResNet-50 with
  one head per attribute (optional; needs a trained model).
- **Temporal voting** — attributes are accumulated per track with confidence weighting,
  so labels stabilize instead of flickering frame to frame.
- **Speed estimation** — perspective homography maps the road to a bird's-eye view and
  measures real-world distance over time, with a separate calibration per carriageway.
- **CSV logging** — one row per unique vehicle: ID, timings, attributes, avg/max speed.
- **Single-file pipeline** — the whole runtime lives in [`anpr.py`](anpr.py) (~180 lines).

---

## 🧠 How it works

The design is a classic **two-stage detector + classifier**, extended with tracking,
temporal fusion, and geometry:

| Stage | Component | Why |
|-------|-----------|-----|
| **Detect** | YOLOv8n (pretrained, COCO) | Fast, accurate vehicle boxes with zero training. |
| **Track** | ByteTrack (`persist=True`) | Stable IDs → attributes/speed belong to *one* vehicle. |
| **Fingerprint** | ResNet-50, 4 classification heads | Shared features, multi-task — more accurate & efficient than 4 separate models. |
| **Fuse** | Confidence-weighted voting per track | A car's make doesn't change; aggregate over frames for a stable label. |
| **Speed** | `cv2.getPerspectiveTransform` per lane group | Corrects perspective so far and near vehicles are measured on the same scale. |

Key engineering decisions:

- **Bottom-center of each box** is used as the ground-contact point for speed — it sits
  on the road plane the homography describes.
- **Per-carriageway homographies** — opposing traffic streams have different geometry;
  one transform would smear the speeds.
- **Masked multi-task loss** — training data rarely labels all four attributes, so each
  image only trains the heads it has labels for (`ignore_index` on the rest).

---

## 🚀 Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. run (YOLO weights auto-download on first use)
python anpr.py path/to/photo.jpg      # image  -> output.jpg
python anpr.py path/to/clip.mp4       # video  -> output.mp4 + vehicles.csv
python anpr.py 0                      # webcam (press q to quit)
```

Without a trained fingerprint model, boxes are labeled `Car (<speed> km/h)`.
Add a trained `fingerprint.pth` (below) and they upgrade to full make/model panels.

### Try it on sample footage

```bash
# free, no-login sample traffic clips
curl -L https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4 -o car-detection.mp4
python anpr.py car-detection.mp4
```

---

## 📏 Speed calibration

Speed accuracy depends on a one-time calibration per camera. In [`anpr.py`](anpr.py):

- `SRC_ROAD_L` / `SRC_ROAD_R` — four road-plane corner pixels per carriageway,
  ordered `far-left, far-right, near-right, near-left`.
- `LANE_WIDTH_M`, `NUM_LANES_LEFT/RIGHT`, `VISIBLE_LENGTH_M` — physical road description.

Grab the pixel corners interactively:

```bash
python calibrate.py clip.mp4      # click 4 points, copy the printed values
```

Detection, tracking, and logging work **without** calibration — only the km/h values
depend on it.

---

## 🏷️ Training the make/model fingerprint (optional)

The fingerprint model is not included (weights aren't committed). See
**[TRAINING.md](TRAINING.md)** for a step-by-step Google Colab guide (free GPU, ~30 min)
using [`train_colab.py`](train_colab.py). In short:

```bash
# 1. download a car dataset (e.g. Stanford Cars) and point prepare_dataset.py at it
python prepare_dataset.py     # -> manifest.csv  (path,make,model,color,type)

# 2. train (GPU strongly recommended — see notes)
python fingerprint_model.py   # -> fingerprint.pth

# 3. drop fingerprint.pth next to anpr.py; labels upgrade automatically
```

**Datasets:** [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
(make/model/type), VeRi-776 (color/type), CompCars (make/model at scale).

**Notes & honest expectations:**
- Training on CPU is impractical (~30 h); use a GPU (a free Colab T4 does it in ~30–45 min).
- Color/type reach ~90%+, make ~85%, **model is the hard head** (~80% on Stanford Cars).
- Stanford Cars is **US-market** — European models won't be recognized without adding
  CompCars or your own labeled images.

---

## 🗂️ Project structure

```
anpr.py                     ⭐ complete runtime pipeline (run this)
fingerprint_model.py        multi-task make/model/color/type model + training
prepare_dataset.py          builds manifest.csv from public datasets
calibrate.py                click-to-calibrate tool for speed

# step-by-step build-up (educational)
detect.py                   stage 1 only: YOLO boxes
classifier.py               single-label make/model classifier
pipeline.py                 detect + classify on an image
fingerprint_pipeline.py     detect + multi-task fingerprint on an image
fingerprint_video.py        + tracking with temporal voting
fingerprint_video_speed.py  + speed via homography
fingerprint_video_log.py    + CSV log (multi-file version of anpr.py)
cv_haar.py / cv_pipeline.py  OpenCV-only variants (cv2.dnn, no PyTorch at runtime)

requirements.txt · LICENSE · .gitignore
```

---

## 📤 Outputs

- `output.jpg` / `output.mp4` — annotated media (H.264, plays everywhere).
- `vehicles.csv` — one row per unique vehicle:
  `id, first_s, last_s, make, model, color, type, avg_kmh, max_kmh`.

---

## 🛠️ Tech stack

**Python · YOLOv8 (Ultralytics) · PyTorch · OpenCV · ByteTrack · ResNet-50 · imageio-ffmpeg**

---

## 🐛 Troubleshooting

- **`ModuleNotFoundError` when running** — you're likely using a different Python than
  the one where packages were installed. Confirm with
  `python -c "import torch, cv2, ultralytics"`; in VS Code, set the interpreter via
  *Python: Select Interpreter*.
- **Video won't play / "file corrupt"** — some OpenCV builds ship broken video encoders.
  This project writes video through `imageio-ffmpeg` (real H.264) specifically to avoid
  that; make sure it's installed (`pip install imageio imageio-ffmpeg`).
- **`yolov8n.pt` fails to load** — an interrupted download left a corrupt file; delete it
  and re-run to re-fetch.

---

## ⚖️ Responsible use

This is an educational computer-vision project. Systems that track and log vehicles
(especially with license-plate recognition) carry privacy and legal obligations that
vary by jurisdiction. The detection/classification is standard CV; the "track vehicles
over time" layer is where real-world deployment constraints apply. Use accordingly.

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.
