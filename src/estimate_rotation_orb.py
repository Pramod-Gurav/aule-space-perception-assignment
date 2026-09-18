import cv2
import numpy as np
import math

REFERENCE = "data/reference.png"
INPUT = "data/rotated_30deg.png"

# Load images
ref = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)
inp = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)

if ref is None or inp is None:
    raise FileNotFoundError("Could not load input images.")

# ORB
orb = cv2.ORB_create(nfeatures=1500)

kp1, des1 = orb.detectAndCompute(ref, None)
kp2, des2 = orb.detectAndCompute(inp, None)

print(f"Reference keypoints: {len(kp1)}")
print(f"Input keypoints: {len(kp2)}")

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)

print(f"Total matches: {len(matches)}")

# Use reasonably good matches
good_matches = [m for m in matches if m.distance < 40]

print(f"Good matches: {len(good_matches)}")

if len(good_matches) < 3:
    raise RuntimeError("Not enough matches for transformation estimation.")

# Extract corresponding points
src_pts = np.float32([
    kp1[m.queryIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

dst_pts = np.float32([
    kp2[m.trainIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

# Estimate affine transformation using RANSAC
M, inlier_mask = cv2.estimateAffinePartial2D(
    src_pts,
    dst_pts,
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0
)

if M is None:
    raise RuntimeError("Could not estimate transformation.")

# Number of RANSAC inliers
inliers = int(inlier_mask.sum())

print(f"RANSAC inliers: {inliers}/{len(good_matches)}")

print("\nAffine transformation:")
print(M)

# Rotation extraction
#
# Affine matrix:
#
# [ a  b  tx ]
# [ c  d  ty ]
#
# For rotation:
#
# angle = atan2(c, a)

a = M[0, 0]
c = M[1, 0]

angle = math.degrees(math.atan2(c, a))

print(f"\nEstimated rotation: {angle:.2f} degrees")
print("Ground truth rotation: +30.00 degrees")
print(f"Absolute error: {abs(angle - 30):.2f} degrees")
