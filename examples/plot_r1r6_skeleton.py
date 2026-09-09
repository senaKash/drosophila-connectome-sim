from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SWC_PATH = Path(
    "data/raw/skeletons/28592.swc"
)


def load_swc(path: Path):
    data = np.loadtxt(
        path,
        comments="#",
    )

    node_id = data[:, 0].astype(int)
    xyz = data[:, 2:5]
    parent_id = data[:, 6].astype(int)

    return node_id, xyz, parent_id


def main():
    node_id, xyz, parent_id = load_swc(
        SWC_PATH
    )

    index_by_id = {
        node: index
        for index, node in enumerate(node_id)
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    for i, parent in enumerate(parent_id):
        if parent == -1:
            continue

        parent_index = index_by_id[parent]

        points = xyz[
            [parent_index, i]
        ]

        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            linewidth=0.8,
        )

    ax.set_title(
        "MaleCNS R1-R6 skeleton — bodyId 28592"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ranges = np.ptp(
        xyz,
        axis=0,
    )

    ax.set_box_aspect(ranges)

    print("SWC nodes:", len(xyz))

    print(
        "X:",
        xyz[:, 0].min(),
        "->",
        xyz[:, 0].max(),
    )

    print(
        "Y:",
        xyz[:, 1].min(),
        "->",
        xyz[:, 1].max(),
    )

    print(
        "Z:",
        xyz[:, 2].min(),
        "->",
        xyz[:, 2].max(),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()