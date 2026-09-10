import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from flysim.vision.columns import (
    build_right_eye_columns,
)
from flysim.vision.eye_geometry import (
    build_eye_geometry,
)
from flysim.rendering.eye_render import (
    render_eye,
)
from flysim.vision.image_input import (
    sample_image_brightness_from_geometry,
)


ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

IMAGE_PATH = "data/input/test.png"


def load_image(path: str) -> np.ndarray:
    image = Image.open(
        path
    ).convert("L")

    return (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )


def main() -> None:
    annotations = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    columns = build_right_eye_columns(
        annotations
    )

    geometry = build_eye_geometry(
        hex1=columns["hex1"].to_numpy(),
        hex2=columns["hex2"].to_numpy(),
    )

    image = load_image(
        IMAGE_PATH
    )

    sampled = sample_image_brightness_from_geometry(
        image=image,
        geometry=geometry,
        blur_radius_px=18,
        fov_x_degrees=110.0,
        fov_y_degrees=90.0,
    )

    brightness = sampled[
        "brightness"
    ].to_numpy()

    print("ANGULAR EYE SAMPLING")
    print("--------------------")
    print(
        "Visual columns:",
        len(brightness),
    )
    print(
        "Brightness min:",
        float(brightness.min()),
    )
    print(
        "Brightness mean:",
        float(brightness.mean()),
    )
    print(
        "Brightness max:",
        float(brightness.max()),
    )

    print()
    print("First 10 samples:")

    print(
        sampled[
            [
                "hex1",
                "hex2",
                "image_x",
                "image_y",
                "brightness",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    render_eye(
        geometry=geometry,
        brightness=brightness,
    )

    plt.show()


if __name__ == "__main__":
    main()
