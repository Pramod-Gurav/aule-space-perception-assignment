import cv2
import numpy as np
import math
import os

IMAGE_PATH = "data/reference.png"
OUTPUT_PATH = "outputs/task_c_22_5deg.png"

# --------------------------------------------------
# 1. Load reference image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

h, w = image.shape[:2]

# Detected port corners
src = np.float32([
    [5, 6],       # top-left
    [506, 6],     # top-right
    [507, 480],   # bottom-right
    [5, 480]      # bottom-left
])

# --------------------------------------------------
# 2. Physical parameters from assignment
# --------------------------------------------------

PORT_SIZE_CM = 40.0
CAMERA_DISTANCE_CM = 100.0
ANGLE_DEG = 22.5

theta = math.radians(ANGLE_DEG)

# --------------------------------------------------
# 3. Estimate virtual camera intrinsics
#
# The real camera intrinsics are not provided.
# We estimate focal length from the known 40 cm port
# and its observed pixel size.
# --------------------------------------------------

port_width_px = (506 - 5)
port_height_px = (480 - 6)

fx = port_width_px * CAMERA_DISTANCE_CM / PORT_SIZE_CM
fy = port_height_px * CAMERA_DISTANCE_CM / PORT_SIZE_CM

cx = (5 + 506 + 507 + 5) / 4
cy = (6 + 6 + 480 + 480) / 4

print("Virtual camera:")
print(f"  fx = {fx:.2f}")
print(f"  fy = {fy:.2f}")
print(f"  cx = {cx:.2f}")
print(f"  cy = {cy:.2f}")

# --------------------------------------------------
# 4. Define port as a 40 cm x 40 cm planar surface
# --------------------------------------------------

world_points = np.float32([
    [-20, -20, 0],   # top-left
    [ 20, -20, 0],   # top-right
    [ 20,  20, 0],   # bottom-right
    [-20,  20, 0]    # bottom-left
])

# --------------------------------------------------
# 5. Place virtual camera 100 cm from centre
#    at 22.5 degrees to the right
# --------------------------------------------------

camera_position = np.array([
    CAMERA_DISTANCE_CM * math.sin(theta),
    0,
    CAMERA_DISTANCE_CM * math.cos(theta)
])

# Camera looks directly at the centre of the port
forward = np.array([
    -math.sin(theta),
    0,
    -math.cos(theta)
])

# Camera horizontal direction
right = np.array([
    math.cos(theta),
    0,
    -math.sin(theta)
])

# Image vertical direction
up = np.array([
    0,
    1,
    0
])

# --------------------------------------------------
# 6. Project the 3D port into the virtual camera
# --------------------------------------------------

dst = []

for point in world_points:

    relative = point - camera_position

    X = np.dot(relative, right)
    Y = np.dot(relative, up)
    Z = np.dot(relative, forward)

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    dst.append([u, v])

dst = np.float32(dst)

print("\nProjected 22.5 degree corners:")

for name, point in zip(
    ["Top-left", "Top-right", "Bottom-right", "Bottom-left"],
    dst
):
    print(f"  {name}: ({point[0]:.2f}, {point[1]:.2f})")

# --------------------------------------------------
# 7. Shift projected image so everything fits
# --------------------------------------------------

min_x = np.min(dst[:, 0])
min_y = np.min(dst[:, 1])

margin = 20

shift_x = margin - min_x
shift_y = margin - min_y

dst_shifted = dst + np.float32([shift_x, shift_y])

max_x = np.max(dst_shifted[:, 0])
max_y = np.max(dst_shifted[:, 1])

output_width = int(math.ceil(max_x + margin))
output_height = int(math.ceil(max_y + margin))

# --------------------------------------------------
# 8. Compute homography
# --------------------------------------------------

H = cv2.getPerspectiveTransform(src, dst_shifted)

# --------------------------------------------------
# 9. Warp image
# --------------------------------------------------

warped = cv2.warpPerspective(
    image,
    H,
    (output_width, output_height)
)

os.makedirs("outputs", exist_ok=True)

# Crop to the projected port
x_min = int(np.floor(np.min(dst_shifted[:, 0])))
y_min = int(np.floor(np.min(dst_shifted[:, 1])))
x_max = int(np.ceil(np.max(dst_shifted[:, 0])))
y_max = int(np.ceil(np.max(dst_shifted[:, 1])))

x_min = max(0, x_min)
y_min = max(0, y_min)
x_max = min(warped.shape[1], x_max)
y_max = min(warped.shape[0], y_max)

cropped = warped[y_min:y_max, x_min:x_max]

cv2.imwrite(OUTPUT_PATH, cropped)

print("\nHomography matrix:")
print(H)

print(f"\nProjected port size: {cropped.shape[1]} x {cropped.shape[0]}")
print(f"Saved: {OUTPUT_PATH}")
