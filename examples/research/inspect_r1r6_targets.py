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
    # Аннотации
    # --------------------------------------------------

    annotations = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "bodyId",
            "type",
            "instance",
            "somaSide",
            "superclass",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    # R1-R6 правого глаза.
    r1r6 = annotations[
        (annotations["type"] == "R1-R6")
        & (annotations["instance"] == "R1-R6_R")
    ].copy()

    r_body_ids = r1r6["bodyId"].to_numpy(
        dtype=np.int64
    )

    print("R1-R6 right eye:", len(r1r6))
    print()

    # --------------------------------------------------
    # Загружаем структурный граф
    # --------------------------------------------------
    #
    # Здесь min_weight=1 намеренно:
    # сейчас исследуем анатомическую структуру,
    # а не запускаем динамику LIF.

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
    # Все исходящие связи R1-R6_R
    # --------------------------------------------------

    r_mask = np.isin(
        source_body_ids,
        r_body_ids,
    )

    target_ids = target_body_ids[r_mask]
    target_weights = connectivity.weights[r_mask]

    print(
        "All outgoing R1-R6 edges:",
        len(target_ids),
    )

    print(
        "Unique target neurons:",
        len(np.unique(target_ids)),
    )

    print()

    # --------------------------------------------------
    # Добавляем target annotations
    # --------------------------------------------------

    edges = pd.DataFrame(
        {
            "target_bodyId": target_ids,
            "weight": target_weights,
        }
    )

    target_annotations = annotations[
        [
            "bodyId",
            "type",
            "instance",
            "somaSide",
            "superclass",
            "assignedOlHex1",
            "assignedOlHex2",
        ]
    ].rename(
        columns={
            "bodyId": "target_bodyId",
        }
    )

    edges = edges.merge(
        target_annotations,
        on="target_bodyId",
        how="left",
    )

    # --------------------------------------------------
    # Какие типы получают R1-R6?
    # --------------------------------------------------

    print("TOP TARGET TYPES BY EDGE COUNT")

    type_edge_counts = (
        edges["type"]
        .fillna("<unknown>")
        .value_counts()
        .head(30)
    )

    print(type_edge_counts)
    print()

    # --------------------------------------------------
    # Сколько суммарного structural weight
    # идёт в каждый тип
    # --------------------------------------------------

    print("TOP TARGET TYPES BY TOTAL WEIGHT")

    total_weight_by_type = (
        edges
        .assign(
            target_type=edges["type"].fillna(
                "<unknown>"
            )
        )
        .groupby("target_type")["weight"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
    )

    print(total_weight_by_type)
    print()

    # --------------------------------------------------
    # Пространственное покрытие типов
    # --------------------------------------------------

    with_hex = edges[
        edges["assignedOlHex1"].notna()
        & edges["assignedOlHex2"].notna()
    ].copy()

    print(
        "Edges to targets with OlHex:",
        len(with_hex),
    )

    print()

    print("HEX COVERAGE BY TARGET TYPE")

    rows = []

    for target_type, group in with_hex.groupby(
        "type",
        dropna=False,
    ):
        unique_targets = group[
            "target_bodyId"
        ].nunique()

        unique_hex = (
            group[
                [
                    "assignedOlHex1",
                    "assignedOlHex2",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        total_weight = group["weight"].sum()

        rows.append(
            {
                "type": target_type,
                "target_neurons": unique_targets,
                "hex_positions": unique_hex,
                "total_weight": total_weight,
            }
        )

    coverage = pd.DataFrame(rows)

    coverage = coverage.sort_values(
        [
            "hex_positions",
            "total_weight",
        ],
        ascending=False,
    )

    print(
        coverage
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()