"""
Stage 2: Make/Model classifier (fine-tuned ResNet on a car dataset).

Two parts:
  - train(): fine-tune on an ImageFolder-style dataset (one folder per make/model)
  - CarClassifier: load trained weights and predict on a cropped car image

Dataset layout expected by train():
    dataset/
      train/
        Volvo_V50/     img1.jpg img2.jpg ...
        Opel_Zafira/   ...
      val/
        Volvo_V50/ ...
        ...

Stanford Cars (196 classes) is the classic choice:
  https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# ImageNet normalization (ResNet was pretrained on it)
_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), _norm,
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), _norm,
])


def build_model(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)   # replace head
    return m.to(DEVICE)


def train(data_dir="dataset", epochs=15, batch=32, lr=1e-4, out="car_model.pth"):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", train_tf)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", eval_tf)
    classes = train_ds.classes

    train_dl = DataLoader(train_ds, batch, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch, shuffle=False, num_workers=4)

    model = build_model(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

        # quick val accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"epoch {epoch+1}/{epochs}  val_acc={correct/total:.3f}")

    torch.save({"state_dict": model.state_dict(), "classes": classes}, out)
    print(f"saved -> {out}")


class CarClassifier:
    """Wrap trained weights for inference on cropped car images (numpy BGR from cv2)."""
    def __init__(self, weights="car_model.pth"):
        ckpt = torch.load(weights, map_location=DEVICE)
        self.classes = ckpt["classes"]
        self.model = build_model(len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, bgr_crop):
        # cv2 gives BGR numpy; convert to RGB PIL
        rgb = bgr_crop[:, :, ::-1]
        x = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
        probs = self.model(x).softmax(1)[0]
        conf, idx = probs.max(0)
        return self.classes[idx], conf.item()


if __name__ == "__main__":
    train()   # python classifier.py  -> trains on ./dataset
