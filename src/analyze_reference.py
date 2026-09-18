import cv2
import numpy as np

IMAGE_PATH = "data/reference.png"
OUTPUT_PATH = "outputs/reference_analysis.png"


def detect_rectangles(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 5000:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        # Avoid storing duplicate rectangles.
        duplicate = False

        for old_x, old_y, old_w, old_h in rectangles:
            if (
                abs(x - old_x) < 10
                and abs(y - old_y) < 10
                and abs(w - old_w) < 10
                and abs(h - old_h) < 10
            ):
                duplicate = True
                break

        if not duplicate:
            rectangles.append((x, y, w, h))

    # Sort from largest to smallest
    rectangles.sort(
        key=lambda r: r[2] * r[3],
        reverse=True
    )

    return rectangles


def detect_circle(gray):
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

    # For this reference image, choose the first detected circle.
    return tuple(circles[0])


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    rectangles = detect_rectangles(edges)
    circle = detect_circle(gray)

    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print("\nDetected rectangles:")

    for i, (x, y, w, h) in enumerate(rectangles):
        print(
            f"  Rectangle {i + 1}: "
            f"x={x}, y={y}, "
            f"w={w}, h={h}, "
            f"area={w * h}"
        )

    print("\nDetected circle:")

    if circle is None:
        print("  No circle detected.")
    else:
        x, y, radius = circle

        print(
            f"  center=({x}, {y}), "
            f"radius={radius}"
        )

    # Draw detections
    output = image.copy()

    for i, (x, y, w, h) in enumerate(rectangles):
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"R{i + 1}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    if circle is not None:
        x, y, radius = circle

        cv2.circle(
            output,
            (x, y),
            radius,
            (0, 0, 255),
            2
        )

        cv2.circle(
            output,
            (x, y),
            4,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            output,
            "REFERENCE CIRCLE",
            (x + radius + 5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imwrite(
        OUTPUT_PATH,
        output
    )

    print(
        f"\nSaved result to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
