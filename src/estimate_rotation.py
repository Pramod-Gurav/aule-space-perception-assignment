import cv2
import numpy as np
import math


REFERENCE_PATH = "data/reference.png"
INPUT_PATH = "data/rotated_30deg.png"


def detect_reference_circle(image):
    """Detect the circular reference marker."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        maxRadius=80,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Select the largest detected circle.
    circle = max(
        circles,
        key=lambda c: c[2]
    )

    return tuple(circle)


def calculate_angle(center, point):
    """Calculate angle of a point relative to an image center."""

    dx = point[0] - center[0]
    dy = point[1] - center[1]

    return math.degrees(
        math.atan2(dy, dx)
    )


def normalize_angle(angle):
    """Normalize angle to [-180, 180]."""

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def main():

    reference = cv2.imread(REFERENCE_PATH)
    input_image = cv2.imread(INPUT_PATH)

    if reference is None:
        raise FileNotFoundError(
            f"Could not read {REFERENCE_PATH}"
        )

    if input_image is None:
        raise FileNotFoundError(
            f"Could not read {INPUT_PATH}"
        )

    # Detect circles automatically
    reference_circle = detect_reference_circle(
        reference
    )

    input_circle = detect_reference_circle(
        input_image
    )

    if reference_circle is None:
        raise RuntimeError(
            "Reference circle could not be detected."
        )

    if input_circle is None:
        raise RuntimeError(
            "Input circle could not be detected."
        )

    print(
        "Reference circle:"
        f" center=({reference_circle[0]}, "
        f"{reference_circle[1]}), "
        f"radius={reference_circle[2]}"
    )

    print(
        "Input circle:"
        f" center=({input_circle[0]}, "
        f"{input_circle[1]}), "
        f"radius={input_circle[2]}"
    )

    # Image centers
    ref_h, ref_w = reference.shape[:2]
    input_h, input_w = input_image.shape[:2]

    ref_center = (
        ref_w / 2,
        ref_h / 2
    )

    input_center = (
        input_w / 2,
        input_h / 2
    )

    # Calculate directions
    reference_angle = calculate_angle(
        ref_center,
        reference_circle
    )

    input_angle = calculate_angle(
        input_center,
        input_circle
    )

    estimated_rotation = normalize_angle(
        input_angle - reference_angle
    )

    print(
        f"Reference direction: "
        f"{reference_angle:.2f} degrees"
    )

    print(
        f"Input direction: "
        f"{input_angle:.2f} degrees"
    )

    print(
        f"Estimated rotation: "
        f"{estimated_rotation:.2f} degrees"
    )


if __name__ == "__main__":
    main()
