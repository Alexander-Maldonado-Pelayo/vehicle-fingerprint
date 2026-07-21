"""
Train the vehicle make/model/type fingerprint on Google Colab (free GPU).

WHY COLAB: training ResNet-50 on ~8k car images is ~30 min on a Colab T4 GPU
vs ~30 hours on a laptop CPU. Runtime -> Change runtime type -> GPU (T4).

WHAT IT DOES:
  1. loads Stanford Cars from HuggingFace (no login, ~2 GB)
  2. parses each class name "Volvo V50 Wagon 2007" -> make / model / type
  3. fine-tunes ResNet-50 with one classification head per attribute
  4. saves fingerprint.pth in EXACTLY the format anpr.py expects
  5. (in Colab) downloads fingerprint.pth to your machine

HOW TO RUN (in a Colab notebook cell):
  !pip install -q datasets
  # then paste this file's contents into a cell and run, OR:
  !python train_colab.py

Then drop the downloaded fingerprint.pth next to anpr.py. Labels upgrade
automatically from "Car" to "Make / Model / Type".

Output checkpoint format (must match anpr.py's Fingerprinter):
  {"state_dict": ..., "vocabs": {attr: [labels]}, "head_sizes": {attr: n}}
"""
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 224
EPOCHS = 15
BATCH = 32
LR = 1e-4
OUT = "fingerprint.pth"

# Stanford Cars has make/model/type but NOT color, so we train these three heads.
ATTRS = ["make", "model", "type"]
TYPE_KEYWORDS = {
    "sedan": "sedan", "coupe": "coupe", "convertible": "convertible",
    "hatchback": "hatchback", "suv": "suv", "wagon": "wagon",
    "minivan": "minivan", "van": "van", "cab": "pickup", "crew": "pickup",
}

_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), _norm])


def parse_class_name(name):
    """'Volvo V50 Wagon 2007' -> ('Volvo', 'V50 Wagon', 'wagon')."""
    name = re.sub(r"\s+\d{4}$", "", name).strip()      # drop trailing year
    make, _, model = name.partition(" ")
    t = ""
    low = name.lower()
    for kw, val in TYPE_KEYWORDS.items():
        if kw in low:
            t = val
            break
    return make, model.strip(), t


# ----------------------------------------------------------------------
class CarsDataset(Dataset):
    """Wraps a HuggingFace Stanford Cars split; yields (image_tensor, {attr: idx})."""
    IGNORE = -1

    def __init__(self, hf_split, class_names, vocabs=None, idx=None):
        self.ds = hf_split
        self.class_names = class_names
        # parse every class once into (make, model, type)
        self.parsed = [parse_class_name(n) for n in class_names]
        if vocabs is None:
            cols = {a: set() for a in ATTRS}
            for make, model, t in self.parsed:
                vals = {"make": make, "model": model, "type": t}
                for a in ATTRS:
                    if vals[a]:
                        cols[a].add(vals[a])
            vocabs = {a: sorted(cols[a]) for a in ATTRS}
        self.vocabs = vocabs
        self.idx = idx or {a: {n: i for i, n in enumerate(vocabs[a])} for a in ATTRS}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        img = train_tf(row["image"].convert("RGB"))
        make, model, t = self.parsed[row["label"]]
        vals = {"make": make, "model": model, "type": t}
        labels = {a: self.idx[a].get(vals[a], self.IGNORE) for a in ATTRS}
        return img, labels


class VehicleFingerprint(nn.Module):
    """Identical architecture to anpr.py so the checkpoint loads there."""
    def __init__(self, head_sizes):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.dim = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.heads = nn.ModuleDict(
            {a: nn.Linear(self.dim, n) for a, n in head_sizes.items()})

    def forward(self, x):
        f = self.backbone(x)
        return {a: h(f) for a, h in self.heads.items()}


def main():
    from datasets import load_dataset
    print(f"device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: no GPU detected. In Colab: Runtime -> Change runtime type -> GPU.")

    # HuggingFace mirror of Stanford Cars. If this id ever 404s, swap for another
    # mirror, e.g. "Multimodal-Fatima/StanfordCars_train".
    print("loading Stanford Cars from HuggingFace...")
    ds = load_dataset("tanganke/stanford_cars", split="train")
    class_names = ds.features["label"].names
    print(f"{len(ds)} images, {len(class_names)} classes")

    data = CarsDataset(ds, class_names)
    head_sizes = {a: len(data.vocabs[a]) for a in ATTRS}
    print("head sizes:", head_sizes)

    dl = DataLoader(data, BATCH, shuffle=True, num_workers=2, pin_memory=True)
    model = VehicleFingerprint(head_sizes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=CarsDataset.IGNORE)

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for imgs, labels in dl:
            imgs = imgs.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss = 0.0
            for a in ATTRS:
                y = labels[a].to(DEVICE)
                if (y != CarsDataset.IGNORE).any():
                    loss = loss + loss_fn(logits[a], y)
            loss.backward()
            opt.step()
            running += float(loss)
        print(f"epoch {epoch+1}/{EPOCHS}  loss={running/len(dl):.3f}")

    torch.save({"state_dict": model.state_dict(),
                "vocabs": data.vocabs,
                "head_sizes": head_sizes}, OUT)
    print(f"saved -> {OUT}")

    # In Colab, download it to your machine:
    try:
        from google.colab import files
        files.download(OUT)
    except Exception:
        print("Not in Colab — copy fingerprint.pth next to anpr.py manually.")


if __name__ == "__main__":
    main()
