import cv2
import glob
import os

INPUT_DIR = "data/task_b_crops"

# Crop dimensions
CROP_W = 250
CROP_H = 200

# If the circle is within this distance from
# the image center, consider it visible/centered.
DEAD_ZONE = 25


def detect_circle(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    # Try multiple thresholds
    for param2 in [30, 25, 22, 20, 18]:

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=param2,
            minRadius=20,
            maxRadius=80
        )

        if circles is not None:

            x, y, r = circles[0][0]

            return (
                float(x),
                float(y),
                float(r)
            )

    return None


def get_movement(circle):

    if circle is None:
        return "SEARCH"

    x, y, _ = circle

    center_x = CROP_W / 2
    center_y = CROP_H / 2

    dx = x - center_x
    dy = y - center_y

    # Marker is sufficiently centered
    if abs(dx) <= DEAD_ZONE and abs(dy) <= DEAD_ZONE:
        return "STOP"

    movements = []

    if dx > DEAD_ZONE:
        movements.append("RIGHT")

    elif dx < -DEAD_ZONE:
        movements.append("LEFT")

    if dy > DEAD_ZONE:
        movements.append("DOWN")

    elif dy < -DEAD_ZONE:
        movements.append("UP")

    return " + ".join(movements)


files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.png")
    )
)

for path in files:

    image = cv2.imread(path)

    if image is None:
        continue

    name = os.path.basename(path)

    circle = detect_circle(image)

    print(f"\n{name}")

    if circle is None:

        print("  Circle: NOT VISIBLE")
        print("  Movement: SEARCH")

        continue

    x, y, r = circle

    print(
        f"  Circle center: "
        f"({x:.1f}, {y:.1f})"
    )

    print(
        f"  Crop center: "
        f"({CROP_W / 2:.1f}, {CROP_H / 2:.1f})"
    )

    print(
        f"  Offset: "
        f"dx={x - CROP_W / 2:.1f}, "
        f"dy={y - CROP_H / 2:.1f}"
    )

    movement = get_movement(circle)

    print(
        f"  Camera movement: "
        f"{movement}"
    )
