"""
Build a unified manifest.csv from public car datasets for the fingerprint model.

Columns: path,make,model,color,type   (empty string = attribute unknown)

Supported sources (download separately, point the paths below at them):

  Stanford Cars  -> make, model, type      (no color)
    https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
    needs: cars_train/  + devkit/cars_train_annos.mat + devkit/cars_meta.mat

  VeRi-776       -> color, type            (no make/model)
    (research download) needs: image_train/ + train_label.xml

Missing attributes are written as "" and get MASKED during training
(see the updated fingerprint_model.py), so a row trains only the heads it has.

Run: python prepare_dataset.py
"""
import os
import csv
import re
import xml.etree.ElementTree as ET

# ---- edit these to your local download paths ---------------------------
STANFORD_DIR = "stanford_cars"     # contains cars_train/ and devkit/
VERI_DIR     = "veri"              # contains image_train/ and train_label.xml
OUT = "manifest.csv"

# body-type keywords that appear inside Stanford Cars class names
TYPE_KEYWORDS = {
    "sedan": "sedan", "coupe": "coupe", "convertible": "convertible",
    "hatchback": "hatchback", "suv": "suv", "wagon": "wagon",
    "minivan": "minivan", "van": "van", "cab": "pickup", "crew": "pickup",
}

# VeRi numeric id -> label maps (from the VeRi-776 spec)
VERI_COLOR = {1: "yellow", 2: "orange", 3: "green", 4: "gray", 5: "red",
              6: "blue", 7: "white", 8: "golden", 9: "brown", 10: "black"}
VERI_TYPE = {1: "sedan", 2: "suv", 3: "van", 4: "hatchback", 5: "mpv",
             6: "pickup", 7: "bus", 8: "truck", 9: "wagon"}


def parse_type(class_name):
    low = class_name.lower()
    for kw, t in TYPE_KEYWORDS.items():
        if kw in low:
            return t
    return ""


def parse_stanford(rows):
    """Append Stanford Cars rows: make, model, type (color left blank)."""
    try:
        from scipy.io import loadmat
    except ImportError:
        print("  [stanford] scipy not installed -> skipping. pip install scipy")
        return
    annos = os.path.join(STANFORD_DIR, "devkit", "cars_train_annos.mat")
    meta = os.path.join(STANFORD_DIR, "devkit", "cars_meta.mat")
    imgdir = os.path.join(STANFORD_DIR, "cars_train")
    if not (os.path.exists(annos) and os.path.exists(meta)):
        print("  [stanford] devkit .mat files not found -> skipping")
        return

    class_names = [str(c[0]) for c in loadmat(meta)["class_names"][0]]
    n = 0
    for a in loadmat(annos)["annotations"][0]:
        cls = int(a["class"][0][0]) - 1           # 1-indexed in the file
        fname = str(a["fname"][0])
        name = class_names[cls]                    # e.g. "Volvo V50 Wagon 2007"
        # strip trailing 4-digit year, split make (first token) from model
        name = re.sub(r"\s+\d{4}$", "", name).strip()
        make, _, model = name.partition(" ")
        rows.append({
            "path": os.path.join(imgdir, fname),
            "make": make, "model": model.strip(),
            "color": "", "type": parse_type(name),
        })
        n += 1
    print(f"  [stanford] added {n} rows")


def parse_veri(rows):
    """Append VeRi rows: color, type (make/model left blank)."""
    label_xml = os.path.join(VERI_DIR, "train_label.xml")
    imgdir = os.path.join(VERI_DIR, "image_train")
    if not os.path.exists(label_xml):
        print("  [veri] train_label.xml not found -> skipping")
        return
    # VeRi xml is latin-1 and sometimes not perfectly formed; parse leniently
    tree = ET.parse(label_xml)
    n = 0
    for item in tree.iter("Item"):
        fname = item.get("imageName")
        cid = int(item.get("colorID", 0))
        tid = int(item.get("typeID", 0))
        rows.append({
            "path": os.path.join(imgdir, fname),
            "make": "", "model": "",
            "color": VERI_COLOR.get(cid, ""), "type": VERI_TYPE.get(tid, ""),
        })
        n += 1
    print(f"  [veri] added {n} rows")


def main():
    rows = []
    print("building manifest...")
    parse_stanford(rows)
    parse_veri(rows)

    # keep only rows whose image actually exists on disk
    rows = [r for r in rows if os.path.exists(r["path"])]
    if not rows:
        print("no rows! check STANFORD_DIR / VERI_DIR paths and downloads.")
        return

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "make", "model", "color", "type"])
        w.writeheader()
        w.writerows(rows)

    # quick coverage report
    for a in ["make", "model", "color", "type"]:
        have = sum(1 for r in rows if r[a])
        print(f"  {a:6s}: {have}/{len(rows)} rows labeled")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
