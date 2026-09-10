from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SKELETONS = Path("data/raw/skeletons_r1r6_right")


def load_eye_points():
    points = []

    for path in SKELETONS.glob("*.swc"):
        data = np.loadtxt(path, comments="#")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        ids = data[:, 0].astype(int)
        xyz = data[:, 2:5]
        parents = data[:, 6].astype(int)

        parent_ids = set(
            parents[parents != -1]
        )

        tips = np.array(
            [
                point
                for node_id, point in zip(ids, xyz)
                if node_id not in parent_ids
            ]
        )

        if len(tips) == 0:
            continue

        # Пока берём просто первый tip,
        # чтобы посмотреть общую форму данных.
        points.append(tips[0])

    return np.asarray(points)


points = load_eye_points()

print("Points:", len(points))

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
ax1.scatter(points[:, 0], points[:, 1], s=4)
ax1.set_title("XY")
ax1.set_aspect("equal")

ax2 = fig.add_subplot(132)
ax2.scatter(points[:, 0], points[:, 2], s=4)
ax2.set_title("XZ")
ax2.set_aspect("equal")

ax3 = fig.add_subplot(133)
ax3.scatter(points[:, 1], points[:, 2], s=4)
ax3.set_title("YZ")
ax3.set_aspect("equal")

plt.tight_layout()
plt.show()