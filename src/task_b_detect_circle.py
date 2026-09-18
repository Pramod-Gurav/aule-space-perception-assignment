import cv2
import glob
import os

INPUT_DIR = "data/task_b_crops"
OUTPUT_DIR = "outputs/task_b"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))

for path in files:

    image = cv2.imread(path)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try several parameter combinations
    detected = None

    for param2 in [30, 25, 22, 20, 18]:

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
            param2=param2,
            minRadius=20,
            maxRadius=80
        )

        if circles is not None:

            # Choose the first detected circle
            x, y, r = circles[0][0]

            detected = (
                int(round(x)),
                int(round(y)),
                int(round(r))
            )

            break

    name = os.path.basename(path)

    if detected is None:

        print(f"{name}: Circle NOT visible")

        continue

    x, y, r = detected

    print(
        f"{name}: "
        f"Circle visible at "
        f"x={x}, y={y}, radius={r}"
    )

    # Draw detection
    output = image.copy()

    cv2.circle(
        output,
        (x, y),
        r,
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

    output_path = os.path.join(
        OUTPUT_DIR,
        name
    )

    cv2.imwrite(output_path, output)
