import math

import matplotlib.pyplot as plt
import pandas as pd


# Путь к официальным аннотациям MaleCNS.
ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


def main() -> None:
    # Загружаем только те колонки, которые нужны для L1-карты.
    #
    # bodyId:
    #   реальный идентификатор нейрона в MaleCNS.
    #
    # type:
    #   тип нейрона, например L1, L2, Mi1, Tm2.
    #
    # instance:
    #   экземпляр типа с указанием стороны, например L1_L / L1_R.
    #
    # somaSide:
    #   сторона тела/мозга, к которой относится сома: L или R.
    #
    # assignedOlHex1 / assignedOlHex2:
    #   дискретные координаты optic-lobe hex column.
    df = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "bodyId",
            "type",
            "instance",
            "somaSide",
            "rootSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    # Оставляем только L1-нейроны,
    # для которых известны обе optic-hex координаты.
    l1 = df[
        (df["type"] == "L1")
        & df["assignedOlHex1"].notna()
        & df["assignedOlHex2"].notna()
    ].copy()

    # Координаты в Feather хранятся как float,
    # потому что колонка содержит NaN.
    #
    # После фильтрации NaN уже нет,
    # поэтому спокойно переводим их в int.
    l1["assignedOlHex1"] = (
        l1["assignedOlHex1"]
        .astype(int)
    )

    l1["assignedOlHex2"] = (
        l1["assignedOlHex2"]
        .astype(int)
    )

    # --------------------------------------------------
    # Общая статистика по L1
    # --------------------------------------------------

    print("L1 neurons:", len(l1))
    print()

    print("rootSide:")
    print(
        l1["rootSide"]
        .value_counts(dropna=False)
    )
    print()

    # Считаем, сколько L1-нейронов попадает
    # в каждую уникальную hex-позицию.
    positions = (
        l1
        .groupby([
            "assignedOlHex1",
            "assignedOlHex2",
        ])
        .size()
    )

    print(
        "Unique hex positions:",
        len(positions),
    )

    print(
        "Max L1 neurons at one position:",
        positions.max(),
    )

    print()

    # --------------------------------------------------
    # Проверяем, как указана сторона глаза
    # --------------------------------------------------

    print("instance examples:")

    print(
        l1[
            [
                "bodyId",
                "instance",
                "somaSide",
                "assignedOlHex1",
                "assignedOlHex2",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()

    print("somaSide:")

    print(
        l1["somaSide"]
        .value_counts(dropna=False)
    )

    print()

    # --------------------------------------------------
    # Проверяем позиции, где есть два L1
    # --------------------------------------------------
    #
    # До разделения глаз многие hex-позиции
    # имеют по два L1-нейрона:
    #
    # один для левого глаза,
    # один для правого глаза.
    #
    # Это полезная диагностическая проверка.

    print("Duplicate hex positions:")

    duplicate_positions = (
        l1
        .groupby([
            "assignedOlHex1",
            "assignedOlHex2",
        ])
        .filter(
            lambda group: len(group) == 2
        )
    )

    print(
        duplicate_positions[
            [
                "bodyId",
                "instance",
                "somaSide",
                "assignedOlHex1",
                "assignedOlHex2",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()

    # --------------------------------------------------
    # Выбираем правый глаз
    # --------------------------------------------------
    #
    # В наших данных:
    #
    # R -> 892 L1
    # L -> 875 L1
    #
    # Правый глаз выглядит наиболее полным,
    # поэтому пока используем именно его
    # как будущую входную visual grid.

    right_eye = l1[
        l1["somaSide"] == "R"
    ].copy()

    # Снова считаем число нейронов
    # в каждой hex-позиции,
    # но уже только для одного глаза.
    right_positions = (
        right_eye
        .groupby([
            "assignedOlHex1",
            "assignedOlHex2",
        ])
        .size()
    )

    print("RIGHT EYE")

    print(
        "L1 neurons:",
        len(right_eye),
    )

    print(
        "Unique hex positions:",
        len(right_positions),
    )

    print(
        "Max neurons at one position:",
        right_positions.max(),
    )

    print()

    # --------------------------------------------------
    # Преобразуем hex-координаты в обычные x/y
    # --------------------------------------------------
    #
    # assignedOlHex1 / assignedOlHex2 —
    # дискретные координаты hex grid.
    #
    # Чтобы нарисовать их через matplotlib,
    # мы немного смещаем каждый второй ряд.
    #
    # Это пока именно визуальная раскладка,
    # а не доказанная физическая геометрия глаза.

    x = (
        right_eye["assignedOlHex1"]
        + 0.5
        * (
            right_eye["assignedOlHex2"]
            % 2
        )
    )

    # Вертикальное расстояние между рядами
    # регулярной hex-сетки:
    #
    # sqrt(3) / 2
    y = (
        right_eye["assignedOlHex2"]
        * (
            math.sqrt(3)
            / 2
        )
    )

    # --------------------------------------------------
    # Рисуем карту
    # --------------------------------------------------

    plt.figure(
        figsize=(10, 10)
    )

    # marker="h" рисует шестиугольный маркер.
    #
    # Каждая точка здесь —
    # один реальный L1-нейрон правого глаза.
    plt.scatter(
        x,
        y,
        s=100,
        marker="h",
    )

    # Чтобы hex-сетка не растягивалась
    # по одной из осей.
    plt.gca().set_aspect("equal")

    # Инвертируем Y, чтобы ориентация
    # была ближе к обычной картинке:
    # верх находится сверху окна.
    plt.gca().invert_yaxis()

    plt.title(
        "MaleCNS right eye — L1 visual columns"
    )

    plt.xlabel("hex x")
    plt.ylabel("hex y")

    plt.tight_layout()

    # Открываем интерактивное окно matplotlib.
    plt.show()


if __name__ == "__main__":
    main()