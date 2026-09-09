import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flysim.connectome.loader import load_malecns_feather


WEIGHTS_PATH = (
    "data/raw/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


# Насколько сильно изгибаем сетку вокруг сферического глаза.
#
# 60-70 градусов уже дают заметную выпуклость,
# но не превращают глаз в почти полный шар.
HORIZONTAL_ANGLE_DEGREES = 70.0
VERTICAL_ANGLE_DEGREES = 70.0


def load_full_right_eye_grid() -> pd.DataFrame:
    """
    Загружает полную L1 retinotopic grid правого глаза.

    Для неё у нас:
        892 L1 neurons
        892 unique hex positions
        1 L1 neuron per position

    Используем её как полный пространственный каркас глаза.
    """
    annotations = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    l1 = annotations[
        (annotations["type"] == "L1")
        & (annotations["somaSide"] == "R")
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].copy()

    l1["assignedOlHex1"] = (
        l1["assignedOlHex1"]
        .astype(int)
    )

    l1["assignedOlHex2"] = (
        l1["assignedOlHex2"]
        .astype(int)
    )

    l1 = l1.rename(
        columns={
            "assignedOlHex1": "hex1",
            "assignedOlHex2": "hex2",
        }
    )

    return l1


def reconstruct_observed_columns() -> pd.DataFrame:
    """
    Восстанавливает high-confidence R1-R6 visual columns.

    Для каждого R1-R6:
        strongest L1 target
        strongest L2 target

    Если оба target указывают на один OlHex,
    считаем позицию подтверждённой.
    """
    annotations = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "bodyId",
            "type",
            "instance",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    r1r6 = annotations[
        (annotations["type"] == "R1-R6")
        & (annotations["instance"] == "R1-R6_R")
    ].copy()

    r_body_ids = r1r6["bodyId"].to_numpy(
        dtype=np.int64
    )

    targets = annotations[
        annotations["type"].isin(["L1", "L2"])
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].copy()

    print("Loading MaleCNS connectivity...")

    neuron_index, connectivity = load_malecns_feather(
        weights_path=WEIGHTS_PATH,
        annotations_path=ANNOTATIONS_PATH,

        # Здесь structural mapping,
        # поэтому берём все связи >= 1.
        min_weight=1,
    )

    print("Loaded.")
    print()

    body_ids = np.asarray(
        neuron_index.body_ids,
        dtype=np.int64,
    )

    source_body_ids = body_ids[
        connectivity.source_indices
    ]

    target_body_ids = body_ids[
        connectivity.target_indices
    ]

    source_mask = np.isin(
        source_body_ids,
        r_body_ids,
    )

    edges = pd.DataFrame(
        {
            "r_bodyId": source_body_ids[source_mask],
            "target_bodyId": target_body_ids[source_mask],
            "weight": connectivity.weights[source_mask],
        }
    )

    target_info = targets[
        [
            "bodyId",
            "type",
            "assignedOlHex1",
            "assignedOlHex2",
        ]
    ].rename(
        columns={
            "bodyId": "target_bodyId",
            "type": "target_type",
        }
    )

    edges = edges.merge(
        target_info,
        on="target_bodyId",
        how="inner",
    )

    strongest = (
        edges
        .sort_values(
            "weight",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "r_bodyId",
                "target_type",
            ]
        )
    )

    l1 = strongest[
        strongest["target_type"] == "L1"
    ].copy()

    l2 = strongest[
        strongest["target_type"] == "L2"
    ].copy()

    l1 = l1.rename(
        columns={
            "assignedOlHex1": "l1_hex1",
            "assignedOlHex2": "l1_hex2",
        }
    )

    l2 = l2.rename(
        columns={
            "assignedOlHex1": "l2_hex1",
            "assignedOlHex2": "l2_hex2",
        }
    )

    comparison = l1[
        [
            "r_bodyId",
            "l1_hex1",
            "l1_hex2",
        ]
    ].merge(
        l2[
            [
                "r_bodyId",
                "l2_hex1",
                "l2_hex2",
            ]
        ],
        on="r_bodyId",
        how="inner",
    )

    comparison["same_hex"] = (
        (comparison["l1_hex1"] == comparison["l2_hex1"])
        & (comparison["l1_hex2"] == comparison["l2_hex2"])
    )

    mapped = comparison[
        comparison["same_hex"]
    ].copy()

    mapped["hex1"] = (
        mapped["l1_hex1"]
        .astype(int)
    )

    mapped["hex2"] = (
        mapped["l1_hex2"]
        .astype(int)
    )

    # На одной visual column может быть несколько R1-R6.
    observed = (
        mapped
        .groupby([
            "hex1",
            "hex2",
        ])
        .size()
        .reset_index(
            name="r1r6_count"
        )
    )

    return observed


def build_eye_map(
    full_grid: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Объединяет:
        полную L1 spatial grid
        +
        observed R1-R6 columns

    Для каждой из 892 колонок получаем:
        observed = True / False
        r1r6_count
    """
    eye = full_grid.merge(
        observed,
        on=[
            "hex1",
            "hex2",
        ],
        how="left",
    )

    eye["observed"] = (
        eye["r1r6_count"]
        .notna()
    )

    eye["r1r6_count"] = (
        eye["r1r6_count"]
        .fillna(0)
        .astype(int)
    )

    return eye


def add_flat_coordinates(
    eye: pd.DataFrame,
) -> pd.DataFrame:
    """
    Сначала строим обычные координаты hex-grid.

    Они останутся нашей научной внутренней системой координат.
    """
    eye = eye.copy()

    eye["flat_x"] = (
        eye["hex1"]
        + 0.5 * (
            eye["hex2"] % 2
        )
    )

    eye["flat_y"] = (
        eye["hex2"]
        * (
            math.sqrt(3) / 2
        )
    )

    return eye


def project_to_curved_eye(
    eye: pd.DataFrame,
) -> pd.DataFrame:
    """
    Проецирует плоскую retinotopic grid
    на выпуклый участок сферы.

    ВАЖНО:
    это визуализационная геометрия,
    а не измеренная 3D-форма глаза MaleCNS.

    OlHex остаётся истинной системой координат симуляции.
    Сферическая поверхность нужна только для красивого отображения.
    """
    eye = eye.copy()

    x = eye["flat_x"].to_numpy()
    y = eye["flat_y"].to_numpy()

    # Центрируем плоскую карту.
    x_center = (x.min() + x.max()) / 2
    y_center = (y.min() + y.max()) / 2

    x_half_range = (
        x.max() - x.min()
    ) / 2

    y_half_range = (
        y.max() - y.min()
    ) / 2

    # Нормализованные координаты примерно [-1, 1].
    u = (
        (x - x_center)
        / x_half_range
    )

    v = (
        (y - y_center)
        / y_half_range
    )

    # Переводим координаты в углы сферической поверхности.
    horizontal_angle = math.radians(
        HORIZONTAL_ANGLE_DEGREES
    )

    vertical_angle = math.radians(
        VERTICAL_ANGLE_DEGREES
    )

    longitude = (
        u
        * horizontal_angle
        / 2
    )

    latitude = (
        -v
        * vertical_angle
        / 2
    )

    # Единичная сфера.
    #
    # Z направлен наружу к зрителю.
    eye["X"] = (
        np.cos(latitude)
        * np.sin(longitude)
    )

    eye["Y"] = np.sin(latitude)

    eye["Z"] = (
        np.cos(latitude)
        * np.cos(longitude)
    )

    return eye


def plot_compound_eye(
    eye: pd.DataFrame,
) -> None:
    """
    Первый curved compound-eye prototype.

    Сейчас каждая visual column рисуется как hex-marker.

    Позже для cinematic renderer:
    - заменим marker на настоящую 3D-фасетку;
    - добавим поверхность;
    - свет;
    - материал;
    - перспективную камеру;
    - активность Bad Apple.
    """
    observed = eye[
        eye["observed"]
    ]

    missing = eye[
        ~eye["observed"]
    ]

    fig = plt.figure(
        figsize=(11, 10)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Missing retinal coverage.
    ax.scatter(
        missing["X"],
        missing["Y"],
        missing["Z"],
        marker="h",
        s=55,
        c="lightgray",
        alpha=0.5,
        label="missing / to reconstruct",
    )

    # Observed columns.
    scatter = ax.scatter(
        observed["X"],
        observed["Y"],
        observed["Z"],
        marker="h",
        s=70,
        c=observed["r1r6_count"],
        alpha=0.95,
        label="observed R1-R6",
    )

    ax.set_title(
        "MaleCNS right-eye retinotopic map\n"
        "projected onto a curved compound-eye visualization"
    )

    # В финальном красивом рендере осей не будет.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect(
        (
            1,
            1,
            0.55,
        )
    )

    # Камера смотрит немного сбоку,
    # чтобы выпуклость сразу была видна.
    ax.view_init(
        elev=12,
        azim=-65,
    )

    ax.legend()

    fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.7,
        label="mapped R1-R6 per observed column",
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    full_grid = load_full_right_eye_grid()

    observed = reconstruct_observed_columns()

    eye = build_eye_map(
        full_grid,
        observed,
    )

    eye = add_flat_coordinates(
        eye
    )

    eye = project_to_curved_eye(
        eye
    )

    total = len(eye)

    observed_count = int(
        eye["observed"].sum()
    )

    missing_count = (
        total - observed_count
    )

    print("CURVED RIGHT-EYE PROTOTYPE")
    print("Total columns:", total)
    print("Observed:", observed_count)
    print("Missing:", missing_count)

    print(
        "Coverage:",
        observed_count / total * 100,
        "%",
    )

    plot_compound_eye(
        eye
    )


if __name__ == "__main__":
    main()