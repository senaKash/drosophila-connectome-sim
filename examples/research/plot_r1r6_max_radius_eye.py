from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SKELETONS = Path(
    "data/raw/skeletons_r1r6_right"
)

points = []

for path in SKELETONS.glob("*.swc"):
    data = np.loadtxt(
        path,
        comments="#",
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # SWC:
    # x, y, z = columns 2:5
    # radius  = column 5
    max_radius_index = np.argmax(
        data[:, 5]
    )

    points.append(
        data[max_radius_index, 2:5]
    )


points = np.asarray(points)

print("Points:", len(points))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    s=6,
)

ax.set_title(
    "R1-R6 max-radius nodes"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_box_aspect(
    np.ptp(points, axis=0)
)

plt.tight_layout()
plt.show()