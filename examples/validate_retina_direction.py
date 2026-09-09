from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATIONS = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

SKELETONS = Path(
    "data/raw/skeletons_r1r6_right"
)


def load_swc(path):
    data = np.loadtxt(
        path,
        comments="#",
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data


df = pd.read_feather(
    ANNOTATIONS,
    columns=[
        "bodyId",
        "type",
        "instance",
        "somaLocation",
    ],
)

known = df[
    (df["type"] == "R1-R6")
    & (df["instance"] == "R1-R6_R")
    & df["somaLocation"].notna()
]


print("KNOWN SOMA NEURONS:", len(known))
print()

for row in known.itertuples():
    data = load_swc(
        SKELETONS / f"{row.bodyId}.swc"
    )

    xyz = data[:, 2:5]
    radius = data[:, 5]

    soma = np.asarray(
        row.somaLocation,
        dtype=float,
    )

    nearest_index = np.argmin(
        np.linalg.norm(
            xyz - soma,
            axis=1,
        )
    )

    max_radius_index = np.argmax(radius)

    root_index = np.where(
        data[:, 6] == -1
    )[0][0]

    nearest_distance = np.linalg.norm(
        xyz[nearest_index] - soma
    )

    radius_distance = np.linalg.norm(
        xyz[max_radius_index] - soma
    )

    root_distance = np.linalg.norm(
        xyz[root_index] - soma
    )

    print(
        row.bodyId,
        "| nearest:",
        round(nearest_distance, 1),
        "| max radius:",
        round(radius_distance, 1),
        "| root:",
        round(root_distance, 1),
    )