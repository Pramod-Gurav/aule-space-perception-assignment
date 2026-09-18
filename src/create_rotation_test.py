import cv2


IMAGE_PATH = "data/reference.png"
OUTPUT_PATH = "data/rotated_30deg.png"

ANGLE = 30.0


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    height, width = image.shape[:2]

    # Rotate around the image centre.
    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        ANGLE,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderValue=(255, 255, 255)
    )

    cv2.imwrite(
        OUTPUT_PATH,
        rotated
    )

    print(f"Created rotated image: {OUTPUT_PATH}")
    print(f"Ground-truth rotation: {ANGLE} degrees")


if __name__ == "__main__":
    main()
