# Aule Space — Computer Vision Challenge

## Overview

This project implements the four tasks from the Aule Space Perception Intern assignment using Python and OpenCV.

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure
aule-perception-assignment/
├── README.md
├── requirements.txt
├── data/
│   └── reference.png
├── src/
├── outputs/
├── results/
├── notebooks/
├── tests/
└── report/
```

## Task A — Rotation Estimation
Objective

Rotate the reference image by an arbitrary angle and estimate the rotation.

Approach
Detect ORB keypoints and descriptors.
Match features using a BF Hamming matcher.
Use RANSAC with estimateAffinePartial2D() to remove incorrect matches.
Extract the rotation angle from the affine transformation matrix.
### Run

```bash
python src/test_rotation_orb.py
```

Synthetic rotation tests were used to validate the method.

Task B — Camera Movement Until Circle Visibility
Objective

Given an arbitrary crop of the port image, determine camera movement steps until the reference circle becomes visible.

Approach
Detect the marker location in the reference image.
Treat the crop as a simulated camera viewport.
Calculate the marker overlap with the viewport.
Generate movement commands until sufficient visibility is reached.

The implementation uses an image-space simulation because camera calibration and physical camera parameters were not provided.

### Run

```bash
python src/create_task_b_crops.py
python src/task_b_search.py
```

Output:

outputs/task_b/
Task C — 22.5° Perspective View
Objective

Generate a view of the port from a slightly right-side camera position, 100 cm from the port centre, at 22.5°.

Approach
Detect the four corners of the port.
Model the port as a 40 cm × 40 cm planar surface.
Place a virtual camera 100 cm from the port centre.
Position the camera at 22.5°.
Project the planar corners into the virtual camera.
Calculate the homography.
Warp the reference image using the homography.
### Run

```bash
python src/task_c_corners.py
python src/task_c_homography.py
```

Output:

outputs/task_c_22_5deg.png
Assumption

The assignment does not provide real camera intrinsic calibration parameters. Therefore, a virtual pinhole-camera model with estimated focal parameters is used for the perspective transformation.

Task D — Incremental Camera Movement
Objective

Move the camera incrementally from the 22.5° view to the front view while keeping the camera 100 cm from the port centre and looking directly at the centre.

Approach

The camera follows an orbital path around the port centre.

The generated viewing angles are:

```text
22.5° → 17.5° → 12.5° → 7.5° → 2.5° → 0°
```

At every step:

Camera distance remains 100 cm.
Camera points toward the port centre.
A homography generates the corresponding view.
### Run

```bash
python src/task_d_incremental.py
```

Output:

outputs/task_d/
Generated Outputs
outputs/
├── marker_detection.png
├── task_b/
├── task_c_22_5deg.png
└── task_d/
Reproducibility

Run the scripts from the project root:

python src/test_rotation_orb.py
python src/create_task_b_crops.py
python src/task_b_search.py
python src/task_c_corners.py
python src/task_c_homography.py
python src/task_d_incremental.py

All processing is implemented using Python and OpenCV.

