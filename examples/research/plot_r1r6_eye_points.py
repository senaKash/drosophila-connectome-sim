from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANNOTATIONS = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
SKELETONS = Path("data/raw/skeletons_r1r6_right")


def load_swc(path):
    data = np.loadtxt(path, comments="#")

    ids = data[:, 0].astype(int)
    xyz = data[:, 2:5]
    parents = data[:, 6].astype(int)

    parent_ids = set(parents[parents != -1])

    tips = np.array([
        point
        for node_id, point in zip(ids, xyz)
        if node_id not in parent_ids
    ])

    return tips


df = pd.read_feather(
    ANNOTATIONS,
    columns=["bodyId", "type", "instance", "somaLocation"],
)

r1r6 = df[
    (df["type"] == "R1-R6")
    & (df["instance"] == "R1-R6_R")
].copy()


# Сначала используем 9 известных somaLocation,
# чтобы понять, где примерно находится сторона retina.

seed_points = []

for row in r1r6[r1r6["somaLocation"].notna()].itertuples():
    path = SKELETONS / f"{row.bodyId}.swc"

    if not path.exists():
        continue

    tips = load_swc(path)
    soma = np.asarray(row.somaLocation, dtype=float)

    seed_points.append(
        tips[np.argmin(np.linalg.norm(tips - soma, axis=1))]
    )

retina_center = np.mean(seed_points, axis=0)

print("Retina seed center:", retina_center)


# Для каждого R1-R6 выбираем конец,
# который ближе всего к retinal seed.

eye_points = []

for body_id in r1r6["bodyId"]:
    path = SKELETONS / f"{body_id}.swc"

    if not path.exists():
        continue

    tips = load_swc(path)

    if len(tips) == 0:
        continue

    eye_points.append(
        tips[
            np.argmin(
                np.linalg.norm(
                    tips - retina_center,
                    axis=1,
                )
            )
        ]
    )

eye_points = np.asarray(eye_points)

print("Eye points:", len(eye_points))


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    eye_points[:, 0],
    eye_points[:, 1],
    eye_points[:, 2],
    s=5,
)

ax.set_title("MaleCNS right-eye R1-R6 candidate retinal endpoints")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_box_aspect(np.ptp(eye_points, axis=0))

plt.tight_layout()
plt.show()