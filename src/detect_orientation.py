import cv2
import numpy as np
import math


def detect_orientation(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 10000:
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

        if len(approx) != 4:
            continue

        candidates.append(approx)

    if not candidates:
        print("No quadrilateral found.")
        return

    # Largest quadrilateral
    contour = max(
        candidates,
        key=cv2.contourArea
    )

    points = contour.reshape(4, 2)

    print("Detected corners:")

    for i, (x, y) in enumerate(points):
        print(
            f"  Corner {i + 1}: "
            f"({x}, {y})"
        )

    # Find the top-most edge.
    # For each pair of corners, calculate edge angle.

    print("\nEdge orientations:")

    for i in range(4):

        p1 = points[i]
        p2 = points[(i + 1) % 4]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        print(
            f"  Edge {i + 1}: "
            f"{angle:.2f} degrees"
        )


if __name__ == "__main__":

    print("\nREFERENCE IMAGE")
    detect_orientation(
        "data/reference.png"
    )

    print("\nROTATED IMAGE")
    detect_orientation(
        "data/rotated_30deg.png"
    )
