import cv2
import numpy as np

IMAGE_PATH = "data/reference.png"
OUTPUT_PATH = "outputs/circle_detection.png"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce small image noise
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=15,
        maxRadius=80,
    )

    output = image.copy()

    if circles is None:
        print("No circles detected.")
    else:
        circles = np.round(circles[0]).astype(int)

        print(f"Circles detected: {len(circles)}")

        for i, (x, y, radius) in enumerate(circles):
            print(
                f"Circle {i + 1}: "
                f"center=({x}, {y}), radius={radius}"
            )

            cv2.circle(
                output,
                (x, y),
                radius,
                (0, 255, 0),
                2
            )

            cv2.circle(
                output,
                (x, y),
                3,
                (0, 0, 255),
                -1
            )

    cv2.imwrite(OUTPUT_PATH, output)

    print(f"Saved result to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
