import cv2
import numpy as np
import math
from pathlib import Path


REFERENCE_PATH = "data/reference.png"
TEST_DIR = Path("data/rotation_tests")


def detect_outer_quadrilateral(image):
    """
    Detect the largest quadrilateral representing
    the outer port boundary.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

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

        if len(approx) == 4:
            candidates.append(
                (area, approx)
            )

    if not candidates:
        return None

    # Largest quadrilateral
    _, quadrilateral = max(
        candidates,
        key=lambda item: item[0]
    )

    return quadrilateral.reshape(4, 2)


def calculate_edge_angles(points):
    """
    Calculate the orientation of all four edges.
    """

    angles = []

    for i in range(4):

        p1 = points[i]
        p2 = points[(i + 1) % 4]

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        angles.append(angle)

    return angles


def normalize_line_angle(angle):
    """
    A square has lines rather than arrows.

    Therefore:
        0° and 180° represent the same line.

    Normalize the angle to [-90°, 90°).
    """

    while angle >= 90:
        angle -= 180

    while angle < -90:
        angle += 180

    return angle


def estimate_orientation(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    points = detect_outer_quadrilateral(
        image
    )

    if points is None:
        return None

    raw_angles = calculate_edge_angles(
        points
    )

    normalized_angles = [
        normalize_line_angle(angle)
        for angle in raw_angles
    ]

    # Because opposite edges are parallel,
    # they represent the same orientation.
    orientation = normalized_angles[0]

    return orientation, raw_angles


def main():

    print("REFERENCE")

    reference_result = estimate_orientation(
        REFERENCE_PATH
    )

    if reference_result is None:
        raise RuntimeError(
            "Could not detect reference quadrilateral."
        )

    reference_orientation = (
        reference_result[0]
    )

    print(
        f"Reference orientation: "
        f"{reference_orientation:.2f}°"
    )

    print("\nTESTS")

    for path in sorted(
        TEST_DIR.glob("*.png")
    ):

        result = estimate_orientation(
            path
        )

        if result is None:
            print(
                f"{path.name}: "
                "DETECTION FAILED"
            )
            continue

        orientation, raw_angles = result

        # Extract ground truth from filename
        ground_truth = int(
            path.stem.split("_")[1]
        )

        estimated_rotation = (
            orientation -
            reference_orientation
        )

        estimated_rotation = (
            normalize_line_angle(
                estimated_rotation
            )
        )

        error = abs(
            estimated_rotation -
            ground_truth
        )

        print(
            f"{path.name}: "
            f"GT={ground_truth:>3}°, "
            f"estimated={estimated_rotation:>7.2f}°, "
            f"error={error:>6.2f}°"
        )


if __name__ == "__main__":
    main()
