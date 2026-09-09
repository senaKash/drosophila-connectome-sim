import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Официальный файл аннотаций MaleCNS.
ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


def load_right_eye_l1() -> pd.DataFrame:
    """
    Загружает L1-нейроны и оставляет только правый глаз.

    Почему L1:
    - у нас для него получилась почти идеальная карта;
    - 892 нейрона;
    - 892 уникальные hex-позиции;
    - 1 нейрон на 1 позицию.

    Возвращает DataFrame с координатами правого глаза.
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
    Преобразует дискретные hex-координаты в x/y для matplotlib.

    assignedOlHex1 / assignedOlHex2:
    - это не обычные декартовы координаты,
      а индексы ячеек в hex-grid.

    Чтобы красиво нарисовать соты, смещаем каждый второй ряд.
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
    Нормализует координаты в диапазон [0, 1].

    Это удобно: дальше можно описывать тестовые изображения
    в абстрактных координатах, не завися от конкретного размера карты.
    """
    df = df.copy()

    x_min = df["x"].min()
    x_max = df["x"].max()

    y_min = df["y"].min()
    y_max = df["y"].max()

    df["xn"] = (df["x"] - x_min) / (x_max - x_min)
    df["yn"] = (df["y"] - y_min) / (y_max - y_min)

    return df


def build_test_pattern(df: pd.DataFrame) -> np.ndarray:
    """
    Строит простой асимметричный чёрно-белый паттерн.

    Нам нужен НЕ симметричный рисунок, чтобы можно было глазами
    понять, не перевёрнута ли картинка и не отражена ли по горизонтали.

    Здесь делаем:
    1. вертикальную белую полосу слева;
    2. верхнюю горизонтальную полосу;
    3. среднюю горизонтальную полосу;
    4. яркое пятно справа снизу.

    Это похоже на грубую букву "F" + отдельную точку,
    чтобы сразу было видно ориентацию.
    """
    xn = df["xn"].to_numpy()
    yn = df["yn"].to_numpy()

    # Чёрный фон.
    brightness = np.zeros(len(df), dtype=float)

    # Вертикальная полоса слева.
    left_bar = (
        (xn >= 0.08)
        & (xn <= 0.18)
        & (yn >= 0.15)
        & (yn <= 0.85)
    )

    # Верхняя полоса.
    top_bar = (
        (xn >= 0.08)
        & (xn <= 0.68)
        & (yn >= 0.12)
        & (yn <= 0.22)
    )

    # Средняя полоса.
    middle_bar = (
        (xn >= 0.08)
        & (xn <= 0.52)
        & (yn >= 0.42)
        & (yn <= 0.52)
    )

    # Яркое пятно справа снизу.
    # Нужен асимметричный маркер ориентации.
    spot = (
        ((xn - 0.82) ** 2 + (yn - 0.78) ** 2)
        <= 0.05 ** 2
    )

    brightness[left_bar] = 1.0
    brightness[top_bar] = 1.0
    brightness[middle_bar] = 1.0
    brightness[spot] = 1.0

    return brightness


def plot_pattern(df: pd.DataFrame, brightness: np.ndarray) -> None:
    """
    Рисует hex-grid правого глаза,
    где цвет каждой ячейки — яркость тестового изображения.
    """
    plt.figure(figsize=(10, 10))

    plt.scatter(
        df["x"],
        df["y"],
        c=brightness,
        s=130,
        marker="h",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    plt.title("Test pattern on MaleCNS right-eye L1 hex grid")
    plt.xlabel("hex x")
    plt.ylabel("hex y")

    plt.colorbar(label="brightness")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # 1. Загружаем правый глаз L1.
    right_eye = load_right_eye_l1()

    # 2. Добавляем обычные x/y для рисования.
    right_eye = add_plot_coordinates(right_eye)

    # 3. Добавляем нормализованные координаты [0, 1].
    right_eye = add_normalized_coordinates(right_eye)

    print("RIGHT EYE L1")
    print("Neurons:", len(right_eye))
    print()

    # Контроль: должна быть почти идеальная карта.
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

    # 4. Строим тестовый паттерн.
    brightness = build_test_pattern(right_eye)

    print("White hexes:", int(np.count_nonzero(brightness > 0.5)))
    print("Black hexes:", int(np.count_nonzero(brightness <= 0.5)))

    # 5. Рисуем.
    plot_pattern(right_eye, brightness)


if __name__ == "__main__":
    main()