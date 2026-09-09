from __future__ import annotations

import numpy as np

from flysim.vision.eye_geometry import EyeGeometry


def reconstruct_fly_view(
    geometry: EyeGeometry,
    brightness: np.ndarray,
    width: int = 640,
    height: int = 480,
    eye_aspect: float = 1.25,
) -> np.ndarray:
    """
    Human-readable развёртка поля зрения compound eye.

    3D-направления visual columns проецируются на плоскость
    азимутальной проекцией.

    Центр искажается слабо, края — сильнее.
    Каждая область получает яркость ближайшей visual column.
    """
    brightness = np.asarray(
        brightness,
        dtype=np.float32,
    )

    if len(brightness) != len(geometry.dx):
        raise ValueError(
            "brightness length must match eye geometry"
        )

    # Направления всех visual columns.
    directions = np.column_stack(
        (
            geometry.dx,
            geometry.dy,
            geometry.dz,
        )
    )

    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    # Центральное направление всего правого глаза.
    forward = directions.mean(axis=0)
    forward /= np.linalg.norm(forward)

    # Строим локальные оси камеры.
    up_hint = np.array(
        [0.0, 1.0, 0.0]
    )

    right = np.cross(
        up_hint,
        forward,
    )

    if np.linalg.norm(right) < 1e-6:
        up_hint = np.array(
            [1.0, 0.0, 0.0]
        )

        right = np.cross(
            up_hint,
            forward,
        )

    right /= np.linalg.norm(right)

    up = np.cross(
        forward,
        right,
    )

    # --------------------------------------------------
    # 3D direction -> azimuthal projection
    # --------------------------------------------------

    forward_component = np.clip(
        directions @ forward,
        -1.0,
        1.0,
    )

    theta = np.arccos(
        forward_component
    )

    side_x = directions @ right
    side_y = directions @ up

    side_length = np.sqrt(
        side_x**2
        + side_y**2
    )

    safe_length = np.where(
        side_length > 1e-8,
        side_length,
        1.0,
    )

    # Направление от центра на плоскости.
    plane_x = side_x / safe_length
    plane_y = side_y / safe_length

    # Азимутальная equidistant projection:
    # расстояние от центра пропорционально углу зрения.
    theta_max = float(
        theta.max()
    )

    if theta_max < 1e-8:
        theta_max = 1.0

    radius = theta / theta_max

    sample_x = (
        plane_x
        * radius
        * eye_aspect
    )

    sample_y = (
        plane_y
        * radius
    )

    # --------------------------------------------------
    # Выходной овал
    # --------------------------------------------------

    grid_x = np.linspace(
        -eye_aspect,
        eye_aspect,
        width,
        dtype=np.float32,
    )

    grid_y = np.linspace(
        -1.0,
        1.0,
        height,
        dtype=np.float32,
    )

    view = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    # --------------------------------------------------
    # Voronoi внутри овальной развёртки
    # --------------------------------------------------

    for row, y in enumerate(grid_y):
        inside = (
            (grid_x / eye_aspect) ** 2
            + y**2
            <= 1.0
        )

        x = grid_x[inside]

        if len(x) == 0:
            continue

        dx = (
            x[:, None]
            - sample_x[None, :]
        )

        dy = (
            y
            - sample_y[None, :]
        )

        distance2 = (
            dx**2
            + dy**2
        )

        nearest = np.argmin(
            distance2,
            axis=1,
        )

        view[
            row,
            inside,
        ] = brightness[
            nearest
        ]

    return view