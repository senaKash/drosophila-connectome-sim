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


def load_right_eye_l1_map() -> pd.DataFrame:
    """
    Загружает полную L1-карту правого глаза.

    Мы уже знаем, что для правого глаза:
    - 892 L1 neurons
    - 892 unique hex positions
    - 1 neuron per position

    Это наша полная spatial grid из 892 visual columns.
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

    l1["assignedOlHex1"] = l1["assignedOlHex1"].astype(int)
    l1["assignedOlHex2"] = l1["assignedOlHex2"].astype(int)

    l1 = l1.rename(
        columns={
            "assignedOlHex1": "hex1",
            "assignedOlHex2": "hex2",
        }
    )

    return l1


def reconstruct_observed_r1r6_columns() -> pd.DataFrame:
    """
    Восстанавливает те visual columns,
    для которых есть high-confidence R1-R6 mapping.

    Идея:
    - берём R1-R6 правого глаза;
    - смотрим их strongest target среди L1;
    - смотрим их strongest target среди L2;
    - если hex(L1) == hex(L2), считаем mapping надёжным.
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

    # R1-R6 правого глаза.
    r1r6 = annotations[
        (annotations["type"] == "R1-R6")
        & (annotations["instance"] == "R1-R6_R")
    ].copy()

    r_body_ids = r1r6["bodyId"].to_numpy(dtype=np.int64)

    # L1/L2 с известными координатами.
    targets = annotations[
        annotations["type"].isin(["L1", "L2"])
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].copy()

    print("Loading MaleCNS connectivity...")

    neuron_index, connectivity = load_malecns_feather(
        weights_path=WEIGHTS_PATH,
        annotations_path=ANNOTATIONS_PATH,
        min_weight=1,  # structural mapping
    )

    print("Loaded.")
    print()

    body_ids = np.asarray(neuron_index.body_ids, dtype=np.int64)

    source_body_ids = body_ids[connectivity.source_indices]
    target_body_ids = body_ids[connectivity.target_indices]

    # Оставляем только исходящие связи от R1-R6.
    source_mask = np.isin(source_body_ids, r_body_ids)

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

    # Для каждого R1-R6 берём strongest L1 и strongest L2.
    strongest = (
        edges
        .sort_values("weight", ascending=False)
        .drop_duplicates(subset=["r_bodyId", "target_type"])
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
        ["r_bodyId", "l1_hex1", "l1_hex2"]
    ].merge(
        l2[
            ["r_bodyId", "l2_hex1", "l2_hex2"]
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

    mapped["hex1"] = mapped["l1_hex1"].astype(int)
    mapped["hex2"] = mapped["l1_hex2"].astype(int)

    # Сколько R1-R6 попадает в каждую observed column.
    observed_columns = (
        mapped
        .groupby(["hex1", "hex2"])
        .size()
        .reset_index(name="r1r6_count")
    )

    return observed_columns


def add_plot_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует hex1/hex2 в обычные x/y для matplotlib.
    """
    df = df.copy()

    df["x"] = (
        df["hex1"]
        + 0.5 * (df["hex2"] % 2)
    )

    df["y"] = (
        df["hex2"]
        * (math.sqrt(3) / 2)
    )

    return df


def build_status_map(
    full_l1: pd.DataFrame,
    observed_columns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Объединяет:
    - полную карту 892 колонок
    - список observed колонок с R1-R6

    Результат:
    для каждой колонки знаем, observed она или missing.
    """
    full_l1 = full_l1.copy()

    observed_keys = observed_columns[
        ["hex1", "hex2", "r1r6_count"]
    ].copy()

    merged = full_l1.merge(
        observed_keys,
        on=["hex1", "hex2"],
        how="left",
    )

    merged["observed"] = merged["r1r6_count"].notna()
    merged["r1r6_count"] = merged["r1r6_count"].fillna(0).astype(int)

    return merged


def plot_flat_map(status_map: pd.DataFrame) -> None:
    """
    Плоская debug-карта:
    - серые hex = missing
    - цветные hex = observed
    """
    plt.figure(figsize=(10, 10))

    missing = status_map[~status_map["observed"]]
    observed = status_map[status_map["observed"]]

    # Сначала рисуем всё missing серым фоном.
    plt.scatter(
        missing["x"],
        missing["y"],
        s=130,
        marker="h",
        c="lightgray",
        edgecolors="black",
        linewidths=0.3,
        label="missing",
    )

    # Потом observed поверх.
    scatter = plt.scatter(
        observed["x"],
        observed["y"],
        s=130,
        marker="h",
        c=observed["r1r6_count"],
        edgecolors="black",
        linewidths=0.3,
        label="observed",
    )

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    plt.title("Observed vs missing retinal coverage on right-eye L1 grid")
    plt.xlabel("hex x")
    plt.ylabel("hex y")

    plt.legend()
    plt.colorbar(
        scatter,
        label="mapped R1-R6 per observed column",
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    # 1. Полная L1-сетка правого глаза.
    full_l1 = load_right_eye_l1_map()

    # 2. Observed R1-R6 columns.
    observed_columns = reconstruct_observed_r1r6_columns()

    # 3. Объединяем.
    status_map = build_status_map(full_l1, observed_columns)

    # 4. Добавляем координаты для plot.
    status_map = add_plot_coordinates(status_map)

    observed_count = int(status_map["observed"].sum())
    missing_count = int((~status_map["observed"]).sum())
    total_count = len(status_map)

    print("RIGHT-EYE VISUAL COLUMNS")
    print("Total:", total_count)
    print("Observed:", observed_count)
    print("Missing:", missing_count)
    print("Coverage:", observed_count / total_count * 100, "%")
    print()

    print("Observed R1-R6 count distribution:")
    print(
        status_map.loc[
            status_map["observed"],
            "r1r6_count",
        ]
        .value_counts()
        .sort_index()
    )

    # 5. Рисуем.
    plot_flat_map(status_map)


if __name__ == "__main__":
    main()