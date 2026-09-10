from pathlib import Path

import imageio.v3 as iio
import numpy as np


def load_video_frames(
    path: str,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """
    Загружает видео как список grayscale-кадров.

    Каждый кадр:
        numpy array
        значения от 0.0 до 1.0
    """
    video_path = Path(path)

    if not video_path.exists():
        raise FileNotFoundError(path)

    frames = []

    for frame in iio.imiter(video_path):
        # RGB -> grayscale
        gray = (
            0.299 * frame[..., 0]
            + 0.587 * frame[..., 1]
            + 0.114 * frame[..., 2]
        )

        gray = gray.astype(np.float32) / 255.0

        frames.append(gray)

        if (
            max_frames is not None
            and len(frames) >= max_frames
        ):
            break

    return frames