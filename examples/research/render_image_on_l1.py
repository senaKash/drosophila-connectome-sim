import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# --------------------------------------------------
# Пути
# --------------------------------------------------

ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

# Сюда положи любую тестовую PNG-картинку.
# Например:
# data/input/test.png
IMAGE_PATH = "data/input/test.png"

# Размер квадратного canvas, в который будем вписывать картинку.
# Чем больше, тем чуть точнее sampling.
CANVAS_SIZE = 512


# --------------------------------------------------
# Загрузка right-eye L1 карты
# --------------------------------------------------

def load_right_eye_l1() -> pd.DataFrame:
    """
    Загружает L1-нейроны и оставляет только правый глаз.

    Мы уже проверили:
    - 892 L1-нейрона правого глаза
    - 892 уникальные hex-позиции
    - 1 нейрон на 1 позицию

    Это делает right-eye L1 удобным "экраном"
    для первой visual demo.
    """
    df = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "bodyId",
            "type",
            "instance",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    l1 = df[
        (df["type"] == "L1")
        & df["assignedOlHex1"].notna()
        & df["assignedOlHex2"].notna()
    ].copy()

    l1["assignedOlHex1"] = l1["assignedOlHex1"].astype(int)
    l1["assignedOlHex2"] = l1["assignedOlHex2"].astype(int)

    right_eye = l1[
        l1["somaSide"] == "R"
    ].copy()

    return right_eye


def add_plot_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Превращает дискретные hex-координаты в обычные x/y
    для отрисовки через matplotlib.

    Это пока визуальная раскладка hex-grid:
    - каждый второй ряд слегка смещён;
    - вертикальный шаг = sqrt(3) / 2.
    """
    df = df.copy()

    df["x"] = (
        df["assignedOlHex1"]
        + 0.5 * (df["assignedOlHex2"] % 2)
    )

    df["y"] = (
        df["assignedOlHex2"]
        * (math.sqrt(3) / 2)
    )

    return df


def add_normalized_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует x/y в диапазон [0, 1].

    Это нужно, чтобы потом можно было брать
    любую картинку и спрашивать:
    какая яркость находится в этой точке?
    """
    df = df.copy()

    x_min = df["x"].min()
    x_max = df["x"].max()

    y_min = df["y"].min()
    y_max = df["y"].max()

    df["xn"] = (df["x"] - x_min) / (x_max - x_min)
    df["yn"] = (df["y"] - y_min) / (y_max - y_min)

    return df


# --------------------------------------------------
# Подготовка изображения
# --------------------------------------------------

def load_and_prepare_image(
    image_path: str,
    canvas_size: int = 512,
) -> np.ndarray:
    """
    Загружает изображение, переводит в grayscale
    и вписывает в квадратный canvas без искажения пропорций.

    Почему не просто resize в квадрат:
    - иначе лицо/силуэт растянется;
    - для Bad Apple это особенно нежелательно.

    Возвращает массив float в диапазоне [0, 1]:
    0.0 = чёрный
    1.0 = белый
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(path).convert("L")

    # Исходные размеры
    src_w, src_h = image.size

    # Масштабируем так, чтобы картинка влезла в квадрат,
    # сохраняя пропорции.
    scale = min(
        canvas_size / src_w,
        canvas_size / src_h,
    )

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    # Чёрный квадратный canvas
    canvas = Image.new(
        mode="L",
        size=(canvas_size, canvas_size),
        color=0,
    )

    # Центрируем изображение на canvas
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2

    canvas.paste(resized, (offset_x, offset_y))

    # В float [0, 1]
    canvas_array = np.asarray(canvas, dtype=np.float32) / 255.0

    return canvas_array


# --------------------------------------------------
# Sampling яркости в hex-позициях
# --------------------------------------------------

def sample_image_at_hex_positions(
    df: pd.DataFrame,
    image_array: np.ndarray,
) -> np.ndarray:
    """
    Для каждой hex-позиции берёт яркость из подготовленного изображения.

    Мы используем xn/yn из диапазона [0, 1],
    переводим их в координаты пикселя и просто берём
    значение ближайшего пикселя.

    Для первой demo этого более чем достаточно.
    """
    height, width = image_array.shape

    xn = df["xn"].to_numpy()
    yn = df["yn"].to_numpy()

    # Переводим [0, 1] -> индексы пикселей
    px = np.rint(xn * (width - 1)).astype(int)
    py = np.rint(yn * (height - 1)).astype(int)

    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    brightness = image_array[py, px]

    return brightness


# --------------------------------------------------
# Визуализация
# --------------------------------------------------

def plot_source_image(image_array: np.ndarray) -> None:
    """
    Показывает подготовленное grayscale-изображение.
    Это удобно для контроля:
    - не перевёрнуто ли оно;
    - не слишком ли маленькое;
    - нормально ли отцентровалось.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(
        image_array,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    plt.title("Prepared source image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_hex_image(
    df: pd.DataFrame,
    brightness: np.ndarray,
) -> None:
    """
    Рисует изображение на right-eye L1 hex-grid.

    Здесь каждая hex-ячейка = один реальный L1-нейрон.
    """
    plt.figure(figsize=(10, 10))

    plt.scatter(
        df["x"],
        df["y"],
        c=brightness,
        s=140,
        marker="h",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        linewidths=0.4,
    )

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    plt.title("Image projected onto MaleCNS right-eye L1 hex grid")
    plt.xlabel("hex x")
    plt.ylabel("hex y")

    plt.colorbar(label="brightness")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    # 1. Загружаем карту правого глаза
    right_eye = load_right_eye_l1()
    right_eye = add_plot_coordinates(right_eye)
    right_eye = add_normalized_coordinates(right_eye)

    print("RIGHT EYE L1")
    print("Neurons:", len(right_eye))
    print()

    positions = (
        right_eye
        .groupby([
            "assignedOlHex1",
            "assignedOlHex2",
        ])
        .size()
    )

    print("Unique hex positions:", len(positions))
    print("Max neurons per position:", positions.max())
    print()

    # 2. Загружаем и готовим изображение
    image_array = load_and_prepare_image(
        IMAGE_PATH,
        canvas_size=CANVAS_SIZE,
    )

    print("Prepared image shape:", image_array.shape)
    print("Image min brightness:", float(image_array.min()))
    print("Image max brightness:", float(image_array.max()))
    print()

    # 3. Сэмплируем яркость в hex-позициях
    brightness = sample_image_at_hex_positions(
        right_eye,
        image_array,
    )

    print("Hex brightness statistics:")
    print("Min:", float(brightness.min()))
    print("Mean:", float(brightness.mean()))
    print("Max:", float(brightness.max()))
    print()

    # 4. Показываем исходную подготовленную картинку
    plot_source_image(image_array)

    # 5. Показываем картинку на глазе мухи
    plot_hex_image(right_eye, brightness)


if __name__ == "__main__":
    main()