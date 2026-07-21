"""
Flock-style "Vehicle Fingerprint": a multi-task deep-learning model.

One shared CNN backbone, several classification heads that each predict one
attribute (make, model, color, body type). Trained jointly with a summed loss.
This mirrors how commercial vehicle-recognition systems describe a vehicle by
a bundle of attributes instead of a single label.

Dataset: an ImageFolder-style set won't do (we need MULTIPLE labels per image),
so we use a CSV manifest:

    manifest.csv
    ------------
    path,make,model,color,type
    imgs/0001.jpg,Volvo,V50,silver,wagon
    imgs/0002.jpg,Opel,Zafira,gray,minivan
    ...
"""
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 224

# The attributes our fingerprint predicts. Add/remove heads freely.
ATTRS = ["make", "model", "color", "type"]

_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), _norm])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.ToTensor(), _norm])


# ----------------------------------------------------------------------
class FingerprintDataset(Dataset):
    def __init__(self, manifest, vocabs=None, tf=train_tf):
        self.rows = list(csv.DictReader(open(manifest)))
        self.tf = tf
        # build a label vocabulary per attribute (or reuse one passed in).
        # "" means the attribute is unknown for that row -> excluded from the
        # vocab and mapped to IGNORE (-1) so it is masked out of the loss.
        if vocabs is None:
            vocabs = {a: sorted({r[a] for r in self.rows if r[a]}) for a in ATTRS}
        self.vocabs = vocabs
        self.idx = {a: {name: i for i, name in enumerate(vocabs[a])} for a in ATTRS}

    IGNORE = -1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.tf(Image.open(r["path"]).convert("RGB"))
        # unknown attribute ("") -> IGNORE, so this row won't train that head
        labels = {a: self.idx[a].get(r[a], self.IGNORE) for a in ATTRS}
        return img, labels


# ----------------------------------------------------------------------
class VehicleFingerprint(nn.Module):
    """Shared backbone + one linear head per attribute."""
    def __init__(self, head_sizes):          # head_sizes: {"make": 40, "model": 196, ...}
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feat_dim = bb.fc.in_features
        bb.fc = nn.Identity()                # backbone now outputs a feature vector
        self.backbone = bb
        self.heads = nn.ModuleDict(
            {a: nn.Linear(self.feat_dim, n) for a, n in head_sizes.items()})

    def forward(self, x):
        f = self.backbone(x)                 # shared features
        return {a: head(f) for a, head in self.heads.items()}   # per-attribute logits


# ----------------------------------------------------------------------
def train(manifest="manifest.csv", epochs=20, batch=32, lr=1e-4,
          out="fingerprint.pth"):
    ds = FingerprintDataset(manifest, tf=train_tf)
    dl = DataLoader(ds, batch, shuffle=True, num_workers=4)

    head_sizes = {a: len(ds.vocabs[a]) for a in ATTRS}
    model = VehicleFingerprint(head_sizes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # ignore_index=-1 makes masked (unknown) labels contribute zero loss/grad
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1,
                                  ignore_index=FingerprintDataset.IGNORE)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels in dl:
            imgs = imgs.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            # multi-task loss = sum of each head's loss.
            # skip a head entirely if EVERY label in the batch is masked
            # (CrossEntropyLoss returns nan when all targets are ignored).
            loss = 0.0
            for a in ATTRS:
                y = labels[a].to(DEVICE)
                if (y != FingerprintDataset.IGNORE).any():
                    loss = loss + loss_fn(logits[a], y)
            loss.backward()
            opt.step()
            running += float(loss)
        print(f"epoch {epoch+1}/{epochs}  loss={running/len(dl):.3f}")

    torch.save({"state_dict": model.state_dict(),
                "vocabs": ds.vocabs,
                "head_sizes": head_sizes}, out)
    print(f"saved -> {out}")


# ----------------------------------------------------------------------
class Fingerprinter:
    """Load trained weights; return an attribute dict for a cropped car (cv2 BGR)."""
    def __init__(self, weights="fingerprint.pth"):
        ckpt = torch.load(weights, map_location=DEVICE)
        self.vocabs = ckpt["vocabs"]
        self.model = VehicleFingerprint(ckpt["head_sizes"]).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, bgr_crop):
        rgb = bgr_crop[:, :, ::-1]
        x = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
        logits = self.model(x)
        result = {}
        for a in ATTRS:
            probs = logits[a].softmax(1)[0]
            conf, idx = probs.max(0)
            result[a] = (self.vocabs[a][idx], conf.item())   # (label, confidence)
        return result


if __name__ == "__main__":
    train()
