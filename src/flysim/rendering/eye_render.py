import matplotlib.pyplot as plt
import numpy as np

from flysim.vision.eye_geometry import EyeGeometry


def render_eye(
    geometry: EyeGeometry,
    brightness: np.ndarray,
):
    """
    Рисует brightness на изогнутой поверхности глаза.

    Даже при brightness=0 фасетка остаётся немного видимой,
    чтобы читалась форма всего compound eye.
    """
    brightness = np.asarray(
        brightness,
        dtype=float,
    )

    if len(brightness) != len(geometry.x3d):
        raise ValueError(
            "brightness count must match eye geometry"
        )

    # Базовая видимость глаза:
    # 0 -> тёмно-серый
    # 1 -> белый
    facet_intensity = 0.12 + 0.88 * brightness

    fig = plt.figure(
        figsize=(8, 8),
        facecolor="black",
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
        facecolor="black",
    )

    ax.scatter(
        geometry.x3d,
        geometry.y3d,
        geometry.z3d,
        c=facet_intensity,
        cmap="gray",
        vmin=0,
        vmax=1,
        marker="h",
        s=95,
        edgecolors=(0.18, 0.18, 0.18, 1.0),
        linewidths=0.45,
    )

    # Более читаемый угол, чтобы чувствовалась выпуклость
    ax.view_init(
        elev=12,
        azim=-62,
    )

    ax.set_box_aspect(
        (
            1.25,
            1.0,
            0.55,
        )
    )

    ax.set_axis_off()

    plt.tight_layout(
        pad=0,
    )

    return fig, ax