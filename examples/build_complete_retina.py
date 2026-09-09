import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flysim.connectome.loader import load_malecns_feather
from flysim.vision.retina import build_complete_retina


WEIGHTS_PATH = (
    "data/raw/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


def load_full_right_eye_columns() -> pd.DataFrame:
    """
    Получает полный spatial grid правого глаза.

    Используем L1, потому что мы уже проверили:

        892 L1 neurons
        892 unique hex positions
        1 L1 neuron per position

    Здесь L1 нужен только как источник полной
    retinotopic geometry глаза.
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

    full_columns = l1[
        [
            "assignedOlHex1",
            "assignedOlHex2",
        ]
    ].rename(
        columns={
            "assignedOlHex1": "hex1",
            "assignedOlHex2": "hex2",
        }
    )

    full_columns["hex1"] = full_columns["hex1"].astype(int)
    full_columns["hex2"] = full_columns["hex2"].astype(int)

    full_columns = (
        full_columns
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return full_columns


def reconstruct_real_r1r6() -> pd.DataFrame:
    """
    Восстанавливает реальные R1-R6 retinotopic positions.

    Метод уже проверен ранее:

        R1-R6
          ├── strongest L1 -> hex
          └── strongest L2 -> hex

    Если L1 и L2 дают одинаковую координату,
    считаем mapping high-confidence.

    В предыдущем анализе agreement был ~99.81%.
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

    # Настоящие R1-R6 правого глаза.
    r1r6 = annotations[
        (annotations["type"] == "R1-R6")
        & (annotations["instance"] == "R1-R6_R")
    ].copy()

    r_body_ids = r1r6["bodyId"].to_numpy(
        dtype=np.int64
    )

    # L1/L2 дают нам spatial coordinates.
    targets = annotations[
        annotations["type"].isin(["L1", "L2"])
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].copy()

    print("Loading MaleCNS connectivity...")

    neuron_index, connectivity = load_malecns_feather(
        weights_path=WEIGHTS_PATH,
        annotations_path=ANNOTATIONS_PATH,

        # Для structural mapping используем все связи.
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

    # Только связи:
    #
    # R1-R6_R -> ...
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

    # Для каждого R1-R6 отдельно находим
    # strongest L1 и strongest L2 target.
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

    # Оставляем только случаи,
    # где две независимые ветви согласны.
    same_hex = (
        (comparison["l1_hex1"] == comparison["l2_hex1"])
        & (comparison["l1_hex2"] == comparison["l2_hex2"])
    )

    mapped = comparison[
        same_hex
    ].copy()

    observed_receptors = pd.DataFrame(
        {
            "r_bodyId": mapped["r_bodyId"].astype(int),
            "hex1": mapped["l1_hex1"].astype(int),
            "hex2": mapped["l1_hex2"].astype(int),
        }
    )

    return observed_receptors


def build_column_summary(
    retina: pd.DataFrame,
) -> pd.DataFrame:
    """
    Строит одну строку на visual column.

    Нам нужно понять:

        observed
        или
        reconstructed

    и сколько receptors приходится на колонку.
    """
    summary = (
        retina
        .groupby(
            ["hex1", "hex2"],
            as_index=False,
        )
        .agg(
            receptor_count=(
                "receptor_id",
                "count",
            ),
            virtual_count=(
                "is_virtual",
                "sum",
            ),
        )
    )

    # Если virtual_count == 0:
    # колонка полностью основана на реальных R1-R6.
    summary["is_reconstructed"] = (
        summary["virtual_count"] > 0
    )

    return summary


def add_plot_coordinates(
    columns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hex coordinates -> обычные x/y для debug-графика.
    """
    columns = columns.copy()

    columns["x"] = (
        columns["hex1"]
        + 0.5 * (
            columns["hex2"] % 2
        )
    )

    columns["y"] = (
        columns["hex2"]
        * (math.sqrt(3) / 2)
    )

    return columns


def plot_complete_retina(
    columns: pd.DataFrame,
) -> None:
    """
    Показывает полный reconstructed retinal layer.

    Тёмные точки:
        observed columns — реальные R1-R6.

    Светлые:
        reconstructed columns — virtual R1-R6.

    Пока это debug-представление.
    """
    observed = columns[
        ~columns["is_reconstructed"]
    ]

    reconstructed = columns[
        columns["is_reconstructed"]
    ]

    plt.figure(
        figsize=(10, 10)
    )

    plt.scatter(
        reconstructed["x"],
        reconstructed["y"],
        s=130,
        marker="h",
        alpha=0.45,
        label="virtual / reconstructed",
    )

    plt.scatter(
        observed["x"],
        observed["y"],
        s=130,
        marker="h",
        label="real R1-R6",
    )

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    plt.title(
        "Complete right-eye retinal input\n"
        "real + reconstructed R1-R6"
    )

    plt.xlabel("hex x")
    plt.ylabel("hex y")

    plt.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    # --------------------------------------------------
    # 1. Полные 892 visual columns
    # --------------------------------------------------

    full_columns = load_full_right_eye_columns()

    # --------------------------------------------------
    # 2. Реальные high-confidence R1-R6
    # --------------------------------------------------

    observed_receptors = reconstruct_real_r1r6()

    # --------------------------------------------------
    # 3. Полный retinal layer
    # --------------------------------------------------
    #
    # Для missing visual columns создаём по 6
    # virtual R1-R6.
    #
    # Почему 6:
    # в observed данных это самый частый
    # R1-R6 motif на одну visual column.

    retina = build_complete_retina(
        full_columns=full_columns,
        observed_receptors=observed_receptors,
        virtual_receptors_per_column=6,
    )

    # --------------------------------------------------
    # 4. Статистика
    # --------------------------------------------------

    column_summary = build_column_summary(
        retina
    )

    real_receptors = retina[
        ~retina["is_virtual"]
    ]

    virtual_receptors = retina[
        retina["is_virtual"]
    ]

    observed_columns = column_summary[
        ~column_summary["is_reconstructed"]
    ]

    reconstructed_columns = column_summary[
        column_summary["is_reconstructed"]
    ]

    print("COMPLETE RETINAL INPUT")
    print("----------------------")

    print(
        "Visual columns:",
        len(column_summary),
    )

    print(
        "Observed columns:",
        len(observed_columns),
    )

    print(
        "Reconstructed columns:",
        len(reconstructed_columns),
    )

    print()

    print(
        "Real R1-R6:",
        len(real_receptors),
    )

    print(
        "Virtual R1-R6:",
        len(virtual_receptors),
    )

    print(
        "Total retinal receptors:",
        len(retina),
    )

    print()

    print("Real coverage:")

    coverage = (
        len(observed_columns)
        / len(column_summary)
        * 100
    )

    print(
        f"{coverage:.3f}%"
    )

    # --------------------------------------------------
    # 5. Визуализация
    # --------------------------------------------------

    column_summary = add_plot_coordinates(
        column_summary
    )

    plot_complete_retina(
        column_summary
    )


if __name__ == "__main__":
    main()