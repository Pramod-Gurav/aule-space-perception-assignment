import cv2
import numpy as np

IMAGE_PATH = "data/reference.png"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError("Reference image not found.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

candidates = []

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 150000:
        continue

    perimeter = cv2.arcLength(
        contour,
        True
    )

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approx) == 4:

        points = approx.reshape(4, 2)

        candidates.append(
            (area, points)
        )


if not candidates:
    raise RuntimeError(
        "Could not detect port corners."
    )


# Largest quadrilateral = outer port
candidates.sort(
    key=lambda x: x[0],
    reverse=True
)

area, corners = candidates[0]


# --------------------------------------------------
# Order corners
# TL, TR, BR, BL
# --------------------------------------------------

def order_points(points):

    points = np.float32(points)

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    s = points.sum(axis=1)

    ordered[0] = points[np.argmin(s)]  # TL
    ordered[2] = points[np.argmax(s)]  # BR

    diff = np.diff(
        points,
        axis=1
    ).flatten()

    ordered[1] = points[np.argmin(diff)]  # TR
    ordered[3] = points[np.argmax(diff)]  # BL

    return ordered


corners = order_points(corners)

print(
    f"Port contour area: {area:.1f}"
)

print("\nOrdered port corners:")

labels = [
    "Top-left",
    "Top-right",
    "Bottom-right",
    "Bottom-left"
]

for label, point in zip(labels, corners):

    print(
        f"  {label}: "
        f"({point[0]:.2f}, {point[1]:.2f})"
    )


# Draw corners
output = image.copy()

for i, point in enumerate(corners):

    x, y = point.astype(int)

    cv2.circle(
        output,
        (x, y),
        6,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        output,
        labels[i],
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

cv2.imwrite(
    "outputs/task_c_corners.png",
    output
)

print(
    "\nSaved: outputs/task_c_corners.png"
)
