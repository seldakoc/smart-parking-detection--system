from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load trained model
model = YOLO(str(MODEL_PATH))

# Ask user for an image
image_path = input("Enter image path: ").strip().strip('"')

if not Path(image_path).is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Run detection on CPU
result = model.predict(
    source=image_path,
    conf=0.25,
    device="cpu",
    verbose=False
)[0]

# Count detected parking spaces
counts = Counter()

for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = result.names[class_id]
    counts[class_name] += 1

free_spaces = counts["free"]
occupied_spaces = counts["occupied"]
total_spaces = free_spaces + occupied_spaces

# Draw detection boxes without labels
annotated = result.plot(labels=False, conf=False)

# Create parking status panel
height, width = annotated.shape[:2]

panel_width = 250
panel_height = 125

x1 = max(0, width - panel_width - 15)
y1 = 15

cv2.rectangle(
    annotated,
    (x1, y1),
    (width - 15, y1 + panel_height),
    (30, 30, 30),
    -1
)

cv2.putText(
    annotated, "PARKING STATUS",
    (x1 + 15, y1 + 25),
    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
    (255, 255, 255), 2
)

cv2.putText(
    annotated, f"FREE: {free_spaces}",
    (x1 + 15, y1 + 55),
    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
    (255, 100, 0), 2
)

cv2.putText(
    annotated, f"OCCUPIED: {occupied_spaces}",
    (x1 + 15, y1 + 85),
    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
    (255, 255, 0), 2
)

cv2.putText(
    annotated, f"TOTAL: {total_spaces}",
    (x1 + 15, y1 + 115),
    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
    (255, 255, 255), 2
)

# Save result
output_path = OUTPUT_DIR / "parking_result.jpg"
cv2.imwrite(str(output_path), annotated)

print(f"Free spaces: {free_spaces}")
print(f"Occupied spaces: {occupied_spaces}")
print(f"Total spaces: {total_spaces}")
print(f"Result saved to: {output_path}")

# Display result
cv2.imshow("Smart Parking Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()