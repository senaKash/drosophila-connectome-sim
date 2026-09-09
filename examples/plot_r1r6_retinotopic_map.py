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


def main() -> None:
    # --------------------------------------------------
    # 1. Загружаем аннотации
    # --------------------------------------------------

    annotations = pd.read_feather(
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

    # Фоторецепторы R1-R6 правого глаза.
    r1r6 = annotations[
        (annotations["type"] == "R1-R6")
        & (annotations["instance"] == "R1-R6_R")
    ].copy()

    r_body_ids = r1r6["bodyId"].to_numpy(
        dtype=np.int64
    )

    # L1 и L2 с известными hex-координатами.
    targets = annotations[
        annotations["type"].isin(["L1", "L2"])
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].copy()

    print("R1-R6 right eye:", len(r1r6))
    print()

    # --------------------------------------------------
    # 2. Загружаем структурный connectome
    # --------------------------------------------------
    #
    # Здесь min_weight=1:
    # нас интересует анатомическое соответствие,
    # а не симуляционный порог weight >= 5.

    print("Loading MaleCNS...")

    neuron_index, connectivity = load_malecns_feather(
        weights_path=WEIGHTS_PATH,
        annotations_path=ANNOTATIONS_PATH,
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

    # --------------------------------------------------
    # 3. Оставляем исходящие связи R1-R6
    # --------------------------------------------------

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

    # Добавляем информацию о target-нейронах.
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

    # --------------------------------------------------
    # 4. Находим strongest L1 и strongest L2
    #    для каждого R1-R6
    # --------------------------------------------------

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
            "target_bodyId": "l1_bodyId",
            "assignedOlHex1": "l1_hex1",
            "assignedOlHex2": "l1_hex2",
        }
    )

    l2 = l2.rename(
        columns={
            "target_bodyId": "l2_bodyId",
            "assignedOlHex1": "l2_hex1",
            "assignedOlHex2": "l2_hex2",
        }
    )

    # --------------------------------------------------
    # 5. Соединяем две независимые оценки позиции
    # --------------------------------------------------

    comparison = l1[
        [
            "r_bodyId",
            "l1_bodyId",
            "l1_hex1",
            "l1_hex2",
        ]
    ].merge(
        l2[
            [
                "r_bodyId",
                "l2_bodyId",
                "l2_hex1",
                "l2_hex2",
            ]
        ],
        on="r_bodyId",
        how="inner",
    )

    # Высокая уверенность:
    # L1 и L2 должны указывать на один hex.
    comparison["same_hex"] = (
        (comparison["l1_hex1"] == comparison["l2_hex1"])
        & (comparison["l1_hex2"] == comparison["l2_hex2"])
    )

    mapped = comparison[
        comparison["same_hex"]
    ].copy()

    # Поскольку координаты совпадают,
    # можно использовать координаты L1.
    mapped["hex1"] = mapped["l1_hex1"].astype(int)
    mapped["hex2"] = mapped["l1_hex2"].astype(int)

    print(
        "High-confidence mapped R1-R6:",
        len(mapped),
    )

    # --------------------------------------------------
    # 6. Сколько R1-R6 соответствует каждой колонке
    # --------------------------------------------------

    hex_counts = (
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

    print(
        "Mapped hex positions:",
        len(hex_counts),
    )

    print()
    print("R1-R6 per mapped hex:")
    print(
        hex_counts["r1r6_count"]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Mean R1-R6 per hex:",
        hex_counts["r1r6_count"].mean(),
    )

    print(
        "Max R1-R6 per hex:",
        hex_counts["r1r6_count"].max(),
    )

    # --------------------------------------------------
    # 7. Превращаем OlHex в координаты для matplotlib
    # --------------------------------------------------

    hex_counts["x"] = (
        hex_counts["hex1"]
        + 0.5 * (hex_counts["hex2"] % 2)
    )

    hex_counts["y"] = (
        hex_counts["hex2"]
        * (math.sqrt(3) / 2)
    )

    # --------------------------------------------------
    # 8. Рисуем восстановленную retinotopic map
    # --------------------------------------------------
    #
    # Один hex:
    #     одна восстановленная visual column.
    #
    # Цвет:
    #     сколько R1-R6 мы смогли привязать
    #     к этой колонке через L1/L2.

    plt.figure(
        figsize=(10, 10)
    )

    scatter = plt.scatter(
        hex_counts["x"],
        hex_counts["y"],
        c=hex_counts["r1r6_count"],
        s=130,
        marker="h",
    )

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    plt.title(
        "MaleCNS R1-R6 retinotopic map\n"
        "reconstructed through L1/L2 connectivity"
    )

    plt.xlabel("hex x")
    plt.ylabel("hex y")

    plt.colorbar(
        scatter,
        label="mapped R1-R6 per visual column",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()