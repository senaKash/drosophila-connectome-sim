import numpy as np
import pandas as pd


def sample_image_brightness(
    image: np.ndarray,
    columns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Берёт яркость изображения в позициях visual columns.

    image:
        2D grayscale array со значениями от 0 до 1.

    columns:
        DataFrame с:
        - hex1, hex2 — координаты visual column
        - xn, yn — нормализованные координаты от 0 до 1

    Возвращает:
        hex1, hex2, brightness
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")

    required_columns = {
        "hex1",
        "hex2",
        "xn",
        "yn",
    }

    if not required_columns.issubset(columns.columns):
        raise ValueError(
            "columns must contain hex1, hex2, xn and yn"
        )

    height, width = image.shape

    # Переводим координаты 0..1
    # в обычные индексы пикселей.
    pixel_x = np.rint(
        columns["xn"].to_numpy()
        * (width - 1)
    ).astype(int)

    pixel_y = np.rint(
        columns["yn"].to_numpy()
        * (height - 1)
    ).astype(int)

    brightness = image[
        pixel_y,
        pixel_x,
    ]

    return pd.DataFrame(
        {
            "hex1": columns["hex1"].to_numpy(),
            "hex2": columns["hex2"].to_numpy(),
            "brightness": brightness,
        }
    )