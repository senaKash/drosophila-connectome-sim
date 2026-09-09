from __future__ import annotations

import numpy as np
import pandas as pd

from flysim.vision.eye_geometry import EyeGeometry


def sample_image_brightness(
    image: np.ndarray,
    columns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Старый плоский sampling по нормализованным координатам.

    Используется в тестах и debug-визуализациях.
    """
    if image.ndim != 2:
        raise ValueError(
            "image must be a 2D grayscale array"
        )

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


def _sample_patch_mean(
    image: np.ndarray,
    x: int,
    y: int,
    radius_px: int,
) -> float:
    """
    Усредняет небольшой участок изображения вокруг точки.

    Это простая модель пространственной интеграции:
    одна фасетка получает не один пиксель,
    а среднюю яркость небольшой области.
    """
    height, width = image.shape

    x0 = max(0, x - radius_px)
    x1 = min(width, x + radius_px + 1)

    y0 = max(0, y - radius_px)
    y1 = min(height, y + radius_px + 1)

    patch = image[
        y0:y1,
        x0:x1,
    ]

    if patch.size == 0:
        return 0.0

    return float(
        patch.mean()
    )


def sample_image_brightness_from_geometry(
    image: np.ndarray,
    geometry: EyeGeometry,
    blur_radius_px: int = 18,
    fov_x_degrees: float = 110.0,
    fov_y_degrees: float = 90.0,
) -> pd.DataFrame:
    """
    Семплирует изображение через направления взгляда фасеток.

    В отличие от плоского sampling, здесь каждая фасетка
    имеет собственный угол обзора.
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "image must be a 2D grayscale array"
        )

    if blur_radius_px < 0:
        raise ValueError(
            "blur_radius_px must be non-negative"
        )

    height, width = image.shape

    # Горизонтальный угол взгляда фасетки.
    yaw = np.arctan2(
        geometry.dx,
        geometry.dz,
    )

    # Вертикальный угол взгляда.
    pitch = np.arctan2(
        geometry.dy,
        np.sqrt(
            geometry.dx**2
            + geometry.dz**2
        ),
    )

    fov_x = np.deg2rad(
        fov_x_degrees
    )

    fov_y = np.deg2rad(
        fov_y_degrees
    )

    # Угол 0 смотрит в центр изображения.
    #
    # -fov/2 -> край изображения
    # +fov/2 -> противоположный край
    image_x_normalized = (
        0.5
        + yaw / fov_x
    )

    image_y_normalized = (
        0.5
        - pitch / fov_y
    )

    image_x_normalized = np.clip(
        image_x_normalized,
        0.0,
        1.0,
    )

    image_y_normalized = np.clip(
        image_y_normalized,
        0.0,
        1.0,
    )

    image_x = np.rint(
        image_x_normalized
        * (width - 1)
    ).astype(int)

    image_y = np.rint(
        image_y_normalized
        * (height - 1)
    ).astype(int)

    brightness = np.empty(
        len(image_x),
        dtype=np.float32,
    )

    for i in range(len(image_x)):
        brightness[i] = _sample_patch_mean(
            image=image,
            x=int(image_x[i]),
            y=int(image_y[i]),
            radius_px=blur_radius_px,
        )

    return pd.DataFrame(
        {
            "hex1": geometry.hex1,
            "hex2": geometry.hex2,
            "image_x": image_x,
            "image_y": image_y,
            "brightness": brightness,
        }
    )