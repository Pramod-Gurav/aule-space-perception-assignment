import cv2
import numpy as np

IMAGE_PATH = "data/reference.png"
OUTPUT_PATH = "outputs/marker_detection.png"


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}"
    )

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

H, W = gray.shape

print(f"Image size: {W} x {H}")


# --------------------------------------------------
# 1. Detect the middle rectangle
# --------------------------------------------------

edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

candidates = []

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 50000:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    ratio = w / float(h)

    if 1.0 < ratio < 1.5:
        candidates.append(
            (area, x, y, w, h)
        )


if not candidates:
    raise RuntimeError(
        "Middle rectangle not detected."
    )

# The smaller large rectangle is the middle port boundary
candidates.sort(key=lambda item: item[0])

_, x, y, w, h = candidates[0]

print(
    f"Middle rectangle: "
    f"x={x}, y={y}, "
    f"w={w}, h={h}"
)


# --------------------------------------------------
# 2. Focus on top-left region
# --------------------------------------------------

# The reference marker is located near the
# top-left corner of the middle rectangle.

roi_x1 = x
roi_y1 = y

roi_x2 = x + int(w * 0.45)
roi_y2 = y + int(h * 0.45)

roi = gray[
    roi_y1:roi_y2,
    roi_x1:roi_x2
]


# --------------------------------------------------
# 3. Create dark-pixel mask
# --------------------------------------------------

mask = np.zeros_like(roi, dtype=np.uint8)

mask[roi < 100] = 255


# --------------------------------------------------
# 4. Distance transform
# --------------------------------------------------

# Distance transform tells us how far every
# foreground pixel is from the nearest background.
#
# Thin rectangle lines -> small distance
# Filled circle -> large distance

distance = cv2.distanceTransform(
    mask,
    cv2.DIST_L2,
    5
)


# --------------------------------------------------
# 5. Find maximum-distance point
# --------------------------------------------------

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
    distance
)

local_x, local_y = max_loc

marker_x = roi_x1 + local_x
marker_y = roi_y1 + local_y


print(
    f"Maximum distance: "
    f"{max_val:.2f} pixels"
)

print(
    f"Marker center: "
    f"({marker_x:.1f}, {marker_y:.1f})"
)


# Distance transform maximum is approximately
# the radius of the filled circular region.
estimated_radius = max_val

print(
    f"Estimated marker radius: "
    f"{estimated_radius:.1f} pixels"
)


# --------------------------------------------------
# 6. Validate that detected point is reasonable
# --------------------------------------------------

# A circle should be significantly thicker than
# the rectangular border.

if max_val < 15:

    print(
        "WARNING: Marker confidence is low."
    )


# --------------------------------------------------
# 7. Draw result
# --------------------------------------------------

output = image.copy()

# Middle rectangle
cv2.rectangle(
    output,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

# Marker
cv2.circle(
    output,
    (
        int(round(marker_x)),
        int(round(marker_y))
    ),
    max(3, int(round(estimated_radius))),
    (0, 255, 0),
    2
)

cv2.circle(
    output,
    (
        int(round(marker_x)),
        int(round(marker_y))
    ),
    4,
    (0, 0, 255),
    -1
)

cv2.imwrite(
    OUTPUT_PATH,
    output
)

print(
    f"Saved: {OUTPUT_PATH}"
)
