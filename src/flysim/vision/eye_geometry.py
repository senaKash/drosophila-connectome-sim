from dataclasses import dataclass

import numpy as np


@dataclass
class EyeGeometry:
    """
    Геометрия фасеток правого глаза.

    hex1, hex2  -> исходные hex-координаты
    x2d, y2d    -> плоская карта
    x3d, y3d, z3d -> изогнутая поверхность глаза

    dx, dy, dz  -> направление взгляда каждой фасетки
    """
    hex1: np.ndarray
    hex2: np.ndarray
    x2d: np.ndarray
    y2d: np.ndarray
    x3d: np.ndarray
    y3d: np.ndarray
    z3d: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray


def hex_to_2d(
    hex1: np.ndarray,
    hex2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Перевод hex-координат в плоские 2D-координаты.
    """
    hex1 = np.asarray(hex1, dtype=float)
    hex2 = np.asarray(hex2, dtype=float)

    x = hex1 + 0.5 * (hex2 % 2)
    y = hex2 * np.sqrt(3) / 2.0

    return x, y


def normalize_2d(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Центрируем и нормализуем координаты.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    scale = max(
        x.max() - x.min(),
        y.max() - y.min(),
    )

    if scale > 0:
        x = x / scale
        y = y / scale

    return x, y


def curve_to_eye_surface(
    x: np.ndarray,
    y: np.ndarray,
    depth: float = 0.9,
    horizontal_scale: float = 1.15,
    vertical_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Превращает плоскую карту в выпуклую поверхность глаза.
    """
    x = np.asarray(x, dtype=float) * horizontal_scale
    y = np.asarray(y, dtype=float) * vertical_scale

    r2 = x**2 + y**2

    z = 1.0 - depth * r2
    z = z - 0.12 * x**2

    return x, y, z


def compute_view_directions(
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    eye_center: tuple[float, float, float] = (0.0, 0.0, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Для каждой фасетки считаем направление взгляда.

    Простая идея:
    фасетка смотрит наружу от условного центра глаза.
    """
    cx, cy, cz = eye_center

    vx = np.asarray(x3d, dtype=float) - cx
    vy = np.asarray(y3d, dtype=float) - cy
    vz = np.asarray(z3d, dtype=float) - cz

    norms = np.sqrt(vx**2 + vy**2 + vz**2)
    norms = np.where(norms == 0.0, 1.0, norms)

    dx = vx / norms
    dy = vy / norms
    dz = vz / norms

    return dx, dy, dz


def build_eye_geometry(
    hex1: np.ndarray,
    hex2: np.ndarray,
) -> EyeGeometry:
    """
    Полный пайплайн:
    hex -> 2D -> normalized -> curved 3D -> view directions
    """
    x2d_raw, y2d_raw = hex_to_2d(hex1, hex2)
    x2d, y2d = normalize_2d(x2d_raw, y2d_raw)

    x3d, y3d, z3d = curve_to_eye_surface(x2d, y2d)
    dx, dy, dz = compute_view_directions(x3d, y3d, z3d)

    return EyeGeometry(
        hex1=np.asarray(hex1),
        hex2=np.asarray(hex2),
        x2d=x2d,
        y2d=y2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        dx=dx,
        dy=dy,
        dz=dz,
    )