
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

import cv2

SOURCE = Path(r"C:\Users\selda\Downloads\PKLot\PKLot\PKLot")
OUTPUT = Path("dataset/yolo")

SAMPLE_SIZE = 200
random.seed(42)

# Collect image/XML pairs
pairs = []

for image_path in SOURCE.rglob("*.jpg"):
    xml_path = image_path.with_suffix(".xml")

    if xml_path.exists():
        pairs.append((image_path, xml_path))

print(f"Found {len(pairs)} image/XML pairs.")

if not pairs:
    raise RuntimeError(
        "No image/XML pairs found. Check the SOURCE path."
    )

# Use images from different parking lots and weather conditions
groups = {}

for image_path, xml_path in pairs:
    relative = image_path.relative_to(SOURCE)
    group = relative.parts[:2]
    groups.setdefault(group, []).append((image_path, xml_path))

selected = []
group_keys = sorted(groups)

while len(selected) < SAMPLE_SIZE:
    added = False

    for key in group_keys:
        if groups[key] and len(selected) < SAMPLE_SIZE:
            index = random.randrange(len(groups[key]))
            selected.append(groups[key].pop(index))
            added = True

    if not added:
        break

# Keep entire capture dates together to reduce data leakage
date_groups = {}

for image_path, xml_path in selected:
    relative = image_path.relative_to(SOURCE)
    date_key = str(relative.parent)
    date_groups.setdefault(date_key, []).append((image_path, xml_path))

dates = list(date_groups)
random.shuffle(dates)

train_dates = set(dates[:max(1, int(len(dates) * 0.8))])

for split in ("train", "val"):
    (OUTPUT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

converted = 0

for image_path, xml_path in selected:
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        continue

    labels = []

    for space in root.findall(".//space"):
        occupied = space.get("occupied")

        if occupied not in ("0", "1"):
            continue

        points = []

        for point in space.findall("./contour/point"):
            x = float(point.get("x"))
            y = float(point.get("y"))
            points.append((x, y))

        if len(points) < 4:
            continue

        x_min = max(0, min(p[0] for p in points))
        y_min = max(0, min(p[1] for p in points))
        x_max = min(width, max(p[0] for p in points))
        y_max = min(height, max(p[1] for p in points))

        if x_max <= x_min or y_max <= y_min:
            continue

        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        # 0 = free, 1 = occupied
        class_id = int(occupied)

        labels.append(
            f"{class_id} {x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    relative = image_path.relative_to(SOURCE)
    split = "train" if str(relative.parent) in train_dates else "val"

    # Unique filename across parking lots and dates
    filename = "_".join(relative.with_suffix("").parts)

    destination = OUTPUT / "images" / split / f"{filename}.jpg"
    label_path = OUTPUT / "labels" / split / f"{filename}.txt"

    shutil.copy2(image_path, destination)
    label_path.write_text("\n".join(labels), encoding="utf-8")

    converted += 1

yaml_content = f"""path: {OUTPUT.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: free
  1: occupied
"""

(OUTPUT / "data.yaml").write_text(
    yaml_content, encoding="utf-8"
)

print(f"Successfully prepared {converted} images.")
print(f"Dataset saved to: {OUTPUT.resolve()}")
