import cv2


REFERENCE_PATH = "data/reference.png"
INPUT_PATH = "data/rotated_30deg.png"


def main():

    reference = cv2.imread(
        REFERENCE_PATH,
        cv2.IMREAD_GRAYSCALE
    )

    input_image = cv2.imread(
        INPUT_PATH,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise FileNotFoundError(
            REFERENCE_PATH
        )

    if input_image is None:
        raise FileNotFoundError(
            INPUT_PATH
        )

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=1000
    )

    # Detect features and descriptors
    keypoints_ref, descriptors_ref = (
        orb.detectAndCompute(
            reference,
            None
        )
    )

    keypoints_input, descriptors_input = (
        orb.detectAndCompute(
            input_image,
            None
        )
    )

    print(
        f"Reference keypoints: "
        f"{len(keypoints_ref)}"
    )

    print(
        f"Input keypoints: "
        f"{len(keypoints_input)}"
    )

    if descriptors_ref is None:
        print("No descriptors in reference.")
        return

    if descriptors_input is None:
        print("No descriptors in input.")
        return

    # ORB uses binary descriptors,
    # therefore use Hamming distance.
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        descriptors_ref,
        descriptors_input
    )

    matches = sorted(
        matches,
        key=lambda match: match.distance
    )

    print(
        f"Total matches: "
        f"{len(matches)}"
    )

    print("\nBest 10 matches:")

    for match in matches[:10]:

        print(
            f"distance={match.distance:.2f}, "
            f"reference_index={match.queryIdx}, "
            f"input_index={match.trainIdx}"
        )


if __name__ == "__main__":
    main()
