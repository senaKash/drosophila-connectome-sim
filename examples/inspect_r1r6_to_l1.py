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
    # Загружаем нужные аннотации
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

    # R1-R6 правого глаза.
    r1r6 = annotations[
        annotations["type"].eq("R1-R6")
        & annotations["instance"].eq("R1-R6_R")
    ].copy()

    # L1 правого глаза.
    l1 = annotations[
        annotations["type"].eq("L1")
        & annotations["somaSide"].eq("R")
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].copy()

    print("R1-R6 right-eye neurons:", len(r1r6))
    print("L1 right-eye neurons:", len(l1))
    print()

    r1r6_body_ids = set(
        r1r6["bodyId"].astype(int)
    )

    l1_body_ids = set(
        l1["bodyId"].astype(int)
    )

    # --------------------------------------------------
    # Загружаем реальный neuron-to-neuron graph
    # --------------------------------------------------

    print("Loading MaleCNS connectivity...")

    neuron_index, connectivity = load_malecns_feather(
        weights_path=WEIGHTS_PATH,
        annotations_path=ANNOTATIONS_PATH,
        #min_weight=5,
        min_weight=1,
    )

    print("Loaded.")
    print()

    # --------------------------------------------------
    # Переводим internal indices обратно в bodyId
    # --------------------------------------------------

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
    # Выбираем только:
    #
    # R1-R6_R -> L1_R
    # --------------------------------------------------

    r1r6_array = np.fromiter(
        r1r6_body_ids,
        dtype=np.int64,
    )

    l1_array = np.fromiter(
        l1_body_ids,
        dtype=np.int64,
    )

    mask = (
        np.isin(
            source_body_ids,
            r1r6_array,
        )
        & np.isin(
            target_body_ids,
            l1_array,
        )
    )

    edge_sources = source_body_ids[mask]
    edge_targets = target_body_ids[mask]
    edge_weights = connectivity.weights[mask]

    print("R1-R6 -> L1 edges:", len(edge_weights))
    print()

    if len(edge_weights) == 0:
        print("No matching edges found.")
        return

    # --------------------------------------------------
    # Статистика
    # --------------------------------------------------

    print("Weight statistics:")
    print("Min:   ", np.min(edge_weights))
    print("Median:", np.median(edge_weights))
    print("Mean:  ", np.mean(edge_weights))
    print("Max:   ", np.max(edge_weights))
    print()

    unique_r = np.unique(edge_sources)
    unique_l1 = np.unique(edge_targets)

    print(
        "R1-R6 neurons participating:",
        len(unique_r),
    )

    print(
        "L1 neurons receiving R1-R6:",
        len(unique_l1),
    )

    print()

    # --------------------------------------------------
    # Сколько L1 targets у каждого R1-R6
    # --------------------------------------------------

    edges_df = pd.DataFrame(
        {
            "r_bodyId": edge_sources,
            "l1_bodyId": edge_targets,
            "weight": edge_weights,
        }
    )

    target_counts = (
        edges_df
        .groupby("r_bodyId")["l1_bodyId"]
        .nunique()
    )

    print("L1 targets per R1-R6:")
    print("Min:   ", target_counts.min())
    print("Median:", target_counts.median())
    print("Mean:  ", target_counts.mean())
    print("Max:   ", target_counts.max())
    print()

    # --------------------------------------------------
    # Для первых R1-R6 показываем самый сильный L1 target
    # и его hex coordinate.
    # --------------------------------------------------

    strongest = (
        edges_df
        .sort_values(
            "weight",
            ascending=False,
        )
        .drop_duplicates(
            subset="r_bodyId",
        )
    )

    l1_coordinates = (
        l1[
            [
                "bodyId",
                "assignedOlHex1",
                "assignedOlHex2",
            ]
        ]
        .rename(
            columns={
                "bodyId": "l1_bodyId",
            }
        )
    )

    strongest = strongest.merge(
        l1_coordinates,
        on="l1_bodyId",
        how="left",
    )

    print("STRONGEST L1 TARGET EXAMPLES:")
    print(
        strongest[
            [
                "r_bodyId",
                "l1_bodyId",
                "weight",
                "assignedOlHex1",
                "assignedOlHex2",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()