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
        (annotations["type"] == "R1-R6")
        & (annotations["instance"] == "R1-R6_R")
    ].copy()

    r_body_ids = r1r6["bodyId"].to_numpy(
        dtype=np.int64
    )

    # L1/L2 правого глаза с известными координатами.
    targets = annotations[
        annotations["type"].isin(["L1", "L2"])
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].copy()

    print("R1-R6:", len(r1r6))
    print()

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

    # Только исходящие связи наших R1-R6.
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

    # --------------------------------------------------
    # Берём сильнейший L1 и сильнейший L2
    # для каждого R1-R6.
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
            "weight": "l1_weight",
            "assignedOlHex1": "l1_hex1",
            "assignedOlHex2": "l1_hex2",
        }
    )

    l2 = l2.rename(
        columns={
            "target_bodyId": "l2_bodyId",
            "weight": "l2_weight",
            "assignedOlHex1": "l2_hex1",
            "assignedOlHex2": "l2_hex2",
        }
    )

    comparison = l1[
        [
            "r_bodyId",
            "l1_bodyId",
            "l1_weight",
            "l1_hex1",
            "l1_hex2",
        ]
    ].merge(
        l2[
            [
                "r_bodyId",
                "l2_bodyId",
                "l2_weight",
                "l2_hex1",
                "l2_hex2",
            ]
        ],
        on="r_bodyId",
        how="inner",
    )

    # Совпадает ли восстановленная hex-позиция?
    comparison["same_hex"] = (
        (comparison["l1_hex1"] == comparison["l2_hex1"])
        & (comparison["l1_hex2"] == comparison["l2_hex2"])
    )

    same_count = int(
        comparison["same_hex"].sum()
    )

    print(
        "R1-R6 with both L1 and L2 targets:",
        len(comparison),
    )

    print(
        "Same L1/L2 hex:",
        same_count,
    )

    print(
        "Different L1/L2 hex:",
        len(comparison) - same_count,
    )

    if len(comparison) > 0:
        print(
            "Agreement:",
            same_count / len(comparison) * 100,
            "%",
        )

    print()

    print("EXAMPLES:")

    print(
        comparison[
            [
                "r_bodyId",
                "l1_bodyId",
                "l1_weight",
                "l1_hex1",
                "l1_hex2",
                "l2_bodyId",
                "l2_weight",
                "l2_hex1",
                "l2_hex2",
                "same_hex",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()