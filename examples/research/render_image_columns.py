import matplotlib.pyplot as plt
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
    image = np.asarray(image, dtype=np.float32) / 255.0

    stimulus = sample_image_brightness(
        image=image,
        columns=columns,
    )

    plot_data = columns.merge(
        stimulus,
        on=["hex1", "hex2"],
    )

    plt.figure(figsize=(9, 9))

    plt.scatter(
        plot_data["hex1"] + 0.5 * (plot_data["hex2"] % 2),
        plot_data["hex2"] * (np.sqrt(3) / 2),
        c=plot_data["brightness"],
        cmap="gray",
        vmin=0,
        vmax=1,
        marker="h",
        s=145,
    )

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    plt.title("Image sampled by 892 right-eye visual columns")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()