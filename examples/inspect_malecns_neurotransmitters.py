import pandas as pd
import numpy as np
from flysim.connectome.loader import load_malecns_feather

PATH = (
    "data/raw/"
    "body-neurotransmitters-male-cns-v1.0.feather"
)


df = pd.read_feather(PATH)

print("Rows:", len(df))

print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nData types:")
print(df.dtypes)

print("\nFirst 10 rows:")
print(df.head(10))

print()
print("Unique bodies:", df["body"].nunique())

print()
print("Rows per body:")
rows_per_body = df["body"].value_counts()

print("Min:   ", rows_per_body.min())
print("Median:", rows_per_body.median())
print("Mean:  ", rows_per_body.mean())
print("Max:   ", rows_per_body.max())

print()
print("Consensus neurotransmitters:")
print(
    df["consensus_nt"]
    .value_counts(dropna=False)
)

print()
print("Example body 10001:")
print(
    df[df["body"] == 10001].to_string(index=False)
)


ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

annotations = pd.read_feather(
    ANNOTATIONS_PATH,
    columns=[
        "bodyId",
        "superclass",
    ],
)

valid_annotations = annotations[
    annotations["superclass"].notna()
    & ~annotations["superclass"].str.contains(
        "tbc",
        case=False,
        na=False,
    )
]

valid_body_ids = valid_annotations["bodyId"]

valid_nt = df[
    df["body"].isin(valid_body_ids)
]

print()
print("VALID NEURONS")
print("Valid neurons:", len(valid_body_ids))
print("Found in neurotransmitter table:", len(valid_nt))
print(
    "Missing:",
    len(valid_body_ids) - len(valid_nt),
)

print()
print("Consensus neurotransmitters for valid neurons:")
print(
    valid_nt["consensus_nt"]
    .value_counts(dropna=False)
)

print()
print("Percentages:")
print(
    (
        valid_nt["consensus_nt"]
        .value_counts(normalize=True, dropna=False)
        * 100
    ).round(3)
)


WEIGHTS_PATH = (
    "data/raw/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)


print()
print("Loading filtered MaleCNS connectivity...")

neuron_index, connectivity = load_malecns_feather(
    weights_path=WEIGHTS_PATH,
    annotations_path=ANNOTATIONS_PATH,
    min_weight=5,
)


# consensus_nt для каждого внутреннего индекса нейрона
nt_by_body = (
    df
    .set_index("body")["consensus_nt"]
)

neuron_nt = (
    nt_by_body
    .reindex(neuron_index.body_ids)
    .fillna("missing")
    .to_numpy()
)


transmitters = [
    "acetylcholine",
    "gaba",
    "glutamate",
    "histamine",
    "dopamine",
    "octopamine",
    "serotonin",
    "unclear",
    "missing",
]


print()
print("OUTGOING CONNECTIONS BY TRANSMITTER")

total_edges = len(connectivity.weights)

for transmitter in transmitters:
    neuron_mask = neuron_nt == transmitter

    # Для каждого ребра смотрим neurotransmitter
    # его presynaptic/source нейрона.
    edge_mask = neuron_mask[
        connectivity.source_indices
    ]

    count = np.count_nonzero(edge_mask)
    percentage = count / total_edges * 100

    print(
        f"{transmitter:14s}: "
        f"{count:8d} "
        f"({percentage:6.3f}%)"
    )


annotations_receptors = pd.read_feather(
    ANNOTATIONS_PATH,
    columns=[
        "bodyId",
        "superclass",
        "receptorType",
    ],
)

valid_receptors = annotations_receptors[
    annotations_receptors["superclass"].notna()
    & ~annotations_receptors["superclass"].str.contains(
        "tbc",
        case=False,
        na=False,
    )
]

print()
print("RECEPTOR TYPES FOR VALID NEURONS")

print(
    valid_receptors["receptorType"]
    .value_counts(dropna=False)
    .head(30)
)

print()
print("Non-null receptorType:")
print(
    valid_receptors["receptorType"]
    .notna()
    .sum()
)

print(
    "Coverage:",
    valid_receptors["receptorType"]
    .notna()
    .mean()
    * 100,
    "%",
)

sign_by_transmitter = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unclear": 0,
    "missing": 0,
}

neuron_sign = np.array(
    [
        sign_by_transmitter.get(nt, 0)
        for nt in neuron_nt
    ],
    dtype=np.int8,
)

edge_signs = neuron_sign[
    connectivity.source_indices
]

print()
print("EDGE SIGN SUMMARY")

for sign, name in [
    (1, "excitatory"),
    (-1, "inhibitory"),
    (0, "unsigned"),
]:
    count = np.count_nonzero(
        edge_signs == sign
    )

    percentage = count / len(edge_signs) * 100

    print(
        f"{name:10s}: "
        f"{count:8d} "
        f"({percentage:6.3f}%)"
    )