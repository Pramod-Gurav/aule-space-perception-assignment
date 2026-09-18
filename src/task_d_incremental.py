import cv2
import numpy as np
import math
import os

IMAGE_PATH = "data/reference.png"
OUTPUT_DIR = "outputs/task_d"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

# Detected front-view port corners
src = np.float32([
    [5, 6],
    [506, 6],
    [507, 480],
    [5, 480]
])

# Assignment parameters
PORT_SIZE_CM = 40.0
CAMERA_DISTANCE_CM = 100.0

# Virtual camera parameters estimated from reference image
fx = (506 - 5) * CAMERA_DISTANCE_CM / PORT_SIZE_CM
fy = (480 - 6) * CAMERA_DISTANCE_CM / PORT_SIZE_CM
cx = (5 + 506 + 507 + 5) / 4
cy = (6 + 6 + 480 + 480) / 4

# Port plane: 40 cm x 40 cm
world_points = np.float32([
    [-20, -20, 0],
    [ 20, -20, 0],
    [ 20,  20, 0],
    [-20,  20, 0]
])

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_view(angle_deg):
    theta = math.radians(angle_deg)

    # Camera stays 100 cm from port centre
    camera_position = np.array([
        CAMERA_DISTANCE_CM * math.sin(theta),
        0,
        CAMERA_DISTANCE_CM * math.cos(theta)
    ])

    # Camera always looks directly at port centre
    forward = np.array([
        -math.sin(theta),
        0,
        -math.cos(theta)
    ])

    right = np.array([
        math.cos(theta),
        0,
        -math.sin(theta)
    ])

    up = np.array([0, 1, 0])

    projected = []

    for point in world_points:
        relative = point - camera_position

        X = np.dot(relative, right)
        Y = np.dot(relative, up)
        Z = np.dot(relative, forward)

        u = fx * X / Z + cx
        v = fy * Y / Z + cy

        projected.append([u, v])

    projected = np.float32(projected)

    # Shift image so projected port fits completely
    margin = 20

    min_x = np.min(projected[:, 0])
    min_y = np.min(projected[:, 1])

    shifted = projected + np.float32([
        margin - min_x,
        margin - min_y
    ])

    width = int(math.ceil(np.max(shifted[:, 0]) + margin))
    height = int(math.ceil(np.max(shifted[:, 1]) + margin))

    # Homography from front view to current camera view
    H = cv2.getPerspectiveTransform(src, shifted)

    warped = cv2.warpPerspective(
        image,
        H,
        (width, height)
    )

    # Crop to projected port
    x1 = int(np.floor(np.min(shifted[:, 0])))
    y1 = int(np.floor(np.min(shifted[:, 1])))
    x2 = int(np.ceil(np.max(shifted[:, 0])))
    y2 = int(np.ceil(np.max(shifted[:, 1])))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(warped.shape[1], x2)
    y2 = min(warped.shape[0], y2)

    output = warped[y1:y2, x1:x2]

    filename = os.path.join(
        OUTPUT_DIR,
        f"view_{angle_deg:05.1f}deg.png"
    )

    cv2.imwrite(filename, output)

    return filename, shifted


# Start at Task C angle and move incrementally to front view
angles = [22.5, 17.5, 12.5, 7.5, 2.5, 0.0]

print("\nTASK D — INCREMENTAL CAMERA MOVEMENT")
print("------------------------------------")
print(f"Camera distance: {CAMERA_DISTANCE_CM} cm")
print("Camera always points toward port centre")
print()

for angle in angles:
    filename, corners = generate_view(angle)

    print(
        f"Angle {angle:5.1f}° -> "
        f"Camera position = "
        f"({CAMERA_DISTANCE_CM * math.sin(math.radians(angle)):.2f}, "
        f"0, "
        f"{CAMERA_DISTANCE_CM * math.cos(math.radians(angle)):.2f}) cm"
    )

    print(f"  Saved: {filename}")

print("\nTask D complete.")
print(f"Images saved in: {OUTPUT_DIR}")
