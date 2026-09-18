import cv2
import os

INPUT = "data/reference.png"
OUTPUT_DIR = "data/task_b_crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = cv2.imread(INPUT)

if image is None:
    raise FileNotFoundError("Could not load reference image.")

h, w = image.shape[:2]

print(f"Reference size: {w} x {h}")

# Detected reference marker
circle_x = 227
circle_y = 187

# Crop size
crop_w = 250
crop_h = 200

# Test crops deliberately placed around the marker
crops = {
    # Circle should be visible
    "crop_marker": (120, 100),

    # Circle should be outside, but nearby
    "crop_right": (300, 100),
    "crop_left": (0, 100),
    "crop_below": (120, 250),
    "crop_above": (120, 0),
}

for name, (x, y) in crops.items():

    x2 = min(x + crop_w, w)
    y2 = min(y + crop_h, h)

    crop = image[y:y2, x:x2]

    output = os.path.join(
        OUTPUT_DIR,
        f"{name}.png"
    )

    cv2.imwrite(output, crop)

    print(
        f"{name}: "
        f"x={x}, y={y}, "
        f"size={crop.shape[1]}x{crop.shape[0]} "
        f"-> {output}"
    )
