import cv2
import numpy as np
import math
from pathlib import Path


REFERENCE_PATH = "data/reference.png"
TEST_DIR = Path("data/rotation_tests")


def detect_outer_square(image):
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

    _, square = max(
        candidates,
        key=lambda x: x[0]
    )

    return square.reshape(4, 2)


def detect_circle(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=15,
        maxRadius=80
    )

    if circles is None:
        return None

    circles = np.round(
        circles[0]
    ).astype(int)

    return tuple(
        max(circles, key=lambda c: c[2])
    )


def calculate_circle_direction(
    square,
    circle
):
    # Centre of the detected outer square
    square_center = np.mean(
        square,
        axis=0
    )

    circle_center = np.array(
        circle[:2],
        dtype=float
    )

    dx = (
        circle_center[0]
        - square_center[0]
    )

    dy = (
        circle_center[1]
        - square_center[1]
    )

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    return angle, square_center


def analyze(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    square = detect_outer_square(
        image
    )

    circle = detect_circle(
        image
    )

    if square is None:
        return None

    if circle is None:
        return None

    angle, center = calculate_circle_direction(
        square,
        circle
    )

    return angle, center, circle


def main():

    reference = analyze(
        REFERENCE_PATH
    )

    if reference is None:
        raise RuntimeError(
            "Reference detection failed."
        )

    reference_angle = reference[0]

    print(
        f"Reference circle direction: "
        f"{reference_angle:.2f}°"
    )

    print("\nTESTS")

    for path in sorted(
        TEST_DIR.glob("*.png")
    ):

        result = analyze(path)

        if result is None:
            print(
                f"{path.name}: "
                "DETECTION FAILED"
            )
            continue

        angle = result[0]

        ground_truth = int(
            path.stem.split("_")[1]
        )

        estimated = (
            angle - reference_angle
        )

        # Normalize to [-180, 180]
        while estimated > 180:
            estimated -= 360

        while estimated < -180:
            estimated += 360

        print(
            f"{path.name}: "
            f"GT={ground_truth:>3}°, "
            f"estimated={estimated:>7.2f}°"
        )


if __name__ == "__main__":
    main()
