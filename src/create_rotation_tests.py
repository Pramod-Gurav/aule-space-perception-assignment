import cv2
from pathlib import Path


IMAGE_PATH = "data/reference.png"
OUTPUT_DIR = Path("data/rotation_tests")

ANGLES = [10, 20, 30, 45, 60, 90]


def rotate_image(image, angle):
    height, width = image.shape[:2]

    center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderValue=(255, 255, 255)
    )

    return rotated


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for angle in ANGLES:

        rotated = rotate_image(
            image,
            angle
        )

        output_path = (
            OUTPUT_DIR /
            f"rotation_{angle:03d}.png"
        )

        cv2.imwrite(
            str(output_path),
            rotated
        )

        print(
            f"Created {output_path} "
            f"(ground truth = {angle}°)"
        )


if __name__ == "__main__":
    main()
