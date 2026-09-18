import cv2
import os

REFERENCE = "data/reference.png"
OUTPUT_DIR = "outputs/task_b"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = cv2.imread(REFERENCE)

if image is None:
    raise FileNotFoundError("Reference image not found.")

H, W = image.shape[:2]

# --------------------------------------------------
# Camera viewport
# --------------------------------------------------

CROP_W = 250
CROP_H = 200

# Simulated camera movement step
STEP = 50

MAX_STEPS = 30


# --------------------------------------------------
# Marker detected from reference
# --------------------------------------------------

MARKER_X = 144
MARKER_Y = 146
MARKER_RADIUS = 48


# --------------------------------------------------
# Visibility threshold
# --------------------------------------------------

# We consider the marker visible when at least
# 20% of its estimated circular region is inside
# the camera viewport.

VISIBILITY_THRESHOLD = 0.20


# --------------------------------------------------
# Starting camera position
# --------------------------------------------------

x = 263
y = 284


def marker_visibility(x, y):

    """
    Calculate how much of the marker lies
    inside the current camera viewport.
    """

    # Marker bounding box
    marker_x1 = MARKER_X - MARKER_RADIUS
    marker_y1 = MARKER_Y - MARKER_RADIUS

    marker_x2 = MARKER_X + MARKER_RADIUS
    marker_y2 = MARKER_Y + MARKER_RADIUS

    # Camera viewport
    view_x1 = x
    view_y1 = y

    view_x2 = x + CROP_W
    view_y2 = y + CROP_H

    # Intersection
    intersection_x1 = max(
        marker_x1,
        view_x1
    )

    intersection_y1 = max(
        marker_y1,
        view_y1
    )

    intersection_x2 = min(
        marker_x2,
        view_x2
    )

    intersection_y2 = min(
        marker_y2,
        view_y2
    )

    # No overlap
    if (
        intersection_x2 <= intersection_x1
        or
        intersection_y2 <= intersection_y1
    ):
        return 0.0

    intersection_area = (
        (intersection_x2 - intersection_x1)
        *
        (intersection_y2 - intersection_y1)
    )

    marker_area = (
        (2 * MARKER_RADIUS)
        *
        (2 * MARKER_RADIUS)
    )

    return intersection_area / marker_area


def choose_camera_direction(x, y):

    """
    Determine actual camera movement required.

    Important:
    Moving the crop window LEFT means the camera
    moves RIGHT relative to the scene.
    """

    camera_center_x = x + CROP_W / 2
    camera_center_y = y + CROP_H / 2

    dx = MARKER_X - camera_center_x
    dy = MARKER_Y - camera_center_y

    # Target is to the LEFT in the image.
    # Camera therefore moves LEFT.
    if abs(dx) > abs(dy):

        if dx < 0:
            return "LEFT"

        return "RIGHT"

    # Target is ABOVE in the image.
    # Camera therefore moves UP.
    else:

        if dy < 0:
            return "UP"

        return "DOWN"


def move_camera(x, y, direction):

    if direction == "LEFT":
        x -= STEP

    elif direction == "RIGHT":
        x += STEP

    elif direction == "UP":
        y -= STEP

    elif direction == "DOWN":
        y += STEP

    # Keep viewport inside reference image
    x = max(0, min(x, W - CROP_W))
    y = max(0, min(y, H - CROP_H))

    return x, y


print("Task B Camera Search")
print("====================")

for step in range(MAX_STEPS + 1):

    visibility = marker_visibility(
        x,
        y
    )

    print(
        f"Step {step}: "
        f"camera view = ({x}, {y})"
    )

    print(
        f"  Marker visibility: "
        f"{visibility * 100:.1f}%"
    )

    if visibility >= VISIBILITY_THRESHOLD:

        print("  Circle: VISIBLE")
        print("  Camera movement: STOP")

        crop = image[
            y:y + CROP_H,
            x:x + CROP_W
        ]

        output = os.path.join(
            OUTPUT_DIR,
            "task_b_final_view.png"
        )

        cv2.imwrite(
            output,
            crop
        )

        print(
            f"  Saved: {output}"
        )

        break

    direction = choose_camera_direction(
        x,
        y
    )

    print("  Circle: NOT VISIBLE")

    print(
        f"  Camera movement: {direction}"
    )

    x, y = move_camera(
        x,
        y,
        direction
    )

else:

    print(
        "Circle was not found "
        "within maximum steps."
    )
