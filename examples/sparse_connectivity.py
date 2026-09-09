import numpy as np

from flysim.simulation.connectivity import SparseConnectivity


connectivity = SparseConnectivity(
    size=4,

    source_indices=np.array([
        0,  # A -> B
        0,  # A -> D
        2,  # C -> B
    ]),

    target_indices=np.array([
        1,
        3,
        1,
    ]),

    weights=np.array([
        2.0,
        5.0,
        3.0,
    ]),
)


spikes = np.array([
    True,   # A
    False,  # B
    True,   # C
    False,  # D
])


inputs = connectivity.propagate(spikes)


print("spikes:")
print(spikes)

print()

print("inputs:")
print(inputs)