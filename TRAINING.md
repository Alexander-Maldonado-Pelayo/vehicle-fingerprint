# Training the make/model fingerprint

The pipeline runs without a trained model (boxes say `Car`). To unlock
**make / model / type** labels you train `fingerprint.pth` once and drop it next to
`anpr.py`. This guide uses **Google Colab's free GPU** — training is ~30 min there
vs ~30 hours on a laptop CPU.

The script that does the work is [`train_colab.py`](train_colab.py).

---

## Steps

### 1. Open a Colab notebook with a GPU
- Go to https://colab.research.google.com → **New notebook**
- **Runtime → Change runtime type → Hardware accelerator: GPU (T4)** → Save

### 2. Get the training script into Colab
Easiest — clone this repo in a cell:

```python
!git clone https://github.com/Alexander-Maldonado-Pelayo/vehicle-fingerprint.git
%cd vehicle-fingerprint
```

### 3. Install the one extra dependency
```python
!pip install -q datasets
```
(Colab already has torch, torchvision, and PIL.)

### 4. Train
```python
!python train_colab.py
```

What happens:
- downloads Stanford Cars from HuggingFace (~2 GB, no login)
- parses class names (`"Volvo V50 Wagon 2007"` → make `Volvo`, model `V50 Wagon`, type `wagon`)
- fine-tunes ResNet-50 with make/model/type heads for 15 epochs
- prints per-epoch loss, saves `fingerprint.pth`, and auto-downloads it to your computer

### 5. Use it locally
Put the downloaded `fingerprint.pth` in your `car_detect` folder next to `anpr.py`:

```bash
python anpr.py car-detection.mp4
```

Labels upgrade from `Car (72 km/h)` to `Make: … / Model: … / Type: …` automatically —
`anpr.py` detects the file and adapts to whatever attributes the model was trained on.

---

## Notes & honest expectations

- **GPU required in practice.** On CPU the script still runs but takes ~30 h; the
  script warns you if no GPU is found.
- **Accuracy:** type is easiest (~90%+), make good (~85%), **model is the hard head**
  (~80% on Stanford Cars' own test set).
- **US-market only.** Stanford Cars does not contain many European models (e.g. Opel
  Zafira, Peugeot 508). To recognize those, add the **CompCars** dataset or your own
  labeled images to the training set.
- **No color head.** Stanford Cars has no color labels, so this model predicts
  make/model/type only. To add color, train on VeRi-776 as well (see
  `prepare_dataset.py`, which merges datasets with a masked loss).

---

## If the dataset id fails to load

The script uses the HuggingFace mirror `tanganke/stanford_cars`. If that ever 404s,
edit the `load_dataset(...)` line in `train_colab.py` to another mirror, e.g.:

```python
ds = load_dataset("Multimodal-Fatima/StanfordCars_train", split="train")
```

The class-name parsing works the same as long as the split exposes an integer
`label` column with a `.names` list of the 196 class strings.
