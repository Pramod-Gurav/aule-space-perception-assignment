import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "data/reference.png"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    print(f"Image shape: {image.shape}")
    print(f"Height: {image.shape[0]}")
    print(f"Width:  {image.shape[1]}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title("Aule Satellite Port - Reference Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
