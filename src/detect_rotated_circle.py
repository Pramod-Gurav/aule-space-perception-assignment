import cv2
import numpy as np

IMAGE_PATH = "data/rotated_30deg.png"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

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
        print("No circle detected.")
        return

    circles = np.round(circles[0]).astype(int)

    print(f"Circles detected: {len(circles)}")

    for i, (x, y, radius) in enumerate(circles):
        print(
            f"Circle {i + 1}: "
            f"center=({x}, {y}), "
            f"radius={radius}"
        )


if __name__ == "__main__":
    main()
