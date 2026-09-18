import cv2
import numpy as np
import math
import glob
import os

REFERENCE = "data/reference.png"

# Load reference
ref = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)

if ref is None:
    raise FileNotFoundError("Reference image not found.")

# ORB
orb = cv2.ORB_create(nfeatures=1500)

kp1, des1 = orb.detectAndCompute(ref, None)

print(f"Reference keypoints: {len(kp1)}")
print()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def estimate_rotation(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    kp2, des2 = orb.detectAndCompute(img, None)

    if des2 is None:
        return None

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    # Keep good matches
    good = [m for m in matches if m.distance < 40]

    if len(good) < 3:
        return None

    src_pts = np.float32([
        kp1[m.queryIdx].pt for m in good
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt for m in good
    ]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if M is None:
        return None

    inliers = int(mask.sum())

    # Extract rotation
    angle = math.degrees(
        math.atan2(M[1, 0], M[0, 0])
    )

    return angle, len(matches), len(good), inliers


files = sorted(
    glob.glob("data/rotation_tests/*.png")
)

print(
    f"{'Image':<25}"
    f"{'Estimated':>12}"
    f"{'Expected':>12}"
    f"{'Error':>12}"
    f"{'Inliers':>12}"
)

print("-" * 73)

for path in files:

    result = estimate_rotation(path)

    if result is None:
        print(f"{os.path.basename(path):<25} FAILED")
        continue

    angle, total, good, inliers = result

    # Extract expected angle from filename
    filename = os.path.basename(path)
    expected = int(
        filename.replace("rotation_", "").replace(".png", "")
    )

    # Our OpenCV image-generation convention produces
    # a negative image-coordinate rotation.
    expected = -expected

    error = abs(angle - expected)

    print(
        f"{filename:<25}"
        f"{angle:>12.2f}"
        f"{expected:>12.2f}"
        f"{error:>12.2f}"
        f"{inliers:>12}"
    )
