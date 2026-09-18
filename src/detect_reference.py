import cv2
import numpy as np

IMAGE_PATH = "data/reference.png"
OUTPUT_PATH = "outputs/reference_detection.png"


def main():
    # Load image
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect strong boundaries
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")
    print(f"Contours detected: {len(contours)}")

    # Draw sufficiently large contours
    detected = image.copy()

    contour_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Approximate contour with fewer points
        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # Four corners -> likely a square/rectangle
        if len(approximation) == 4:
            x, y, w, h = cv2.boundingRect(approximation)

            cv2.rectangle(
                detected,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            contour_count += 1

            print(
                f"Rectangle candidate: "
                f"x={x}, y={y}, w={w}, h={h}, area={area:.1f}"
            )

    print(f"Rectangle candidates: {contour_count}")

    # Save result
    cv2.imwrite(OUTPUT_PATH, detected)

    print(f"Saved result to: {OUTPUT_PATH}")

    # Display
    cv2.imshow("Reference Detection", detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
