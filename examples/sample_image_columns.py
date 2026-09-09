import numpy as np
import pandas as pd
from PIL import Image

from flysim.vision.columns import (
    add_normalized_coordinates,
    build_right_eye_columns,
)
from flysim.vision.image_input import sample_image_brightness


ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

IMAGE_PATH = "data/input/test.png"


def main():
    annotations = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    columns = build_right_eye_columns(annotations)
    columns = add_normalized_coordinates(columns)

    image = Image.open(IMAGE_PATH).convert("L")

    image = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    brightness = sample_image_brightness(
        image=image,
        columns=columns,
    )

    print("IMAGE -> VISUAL COLUMNS")
    print("-----------------------")
    print("Columns:", len(brightness))
    print("Min:", brightness["brightness"].min())
    print("Mean:", brightness["brightness"].mean())
    print("Max:", brightness["brightness"].max())

    print()
    print(brightness.head(10).to_string(index=False))


if __name__ == "__main__":
    main()