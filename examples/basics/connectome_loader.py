import numpy as np

from flysim.connectome.loader import build_connectivity


neuron_index, connectivity = build_connectivity(
    source_body_ids=[
        1001,
        1001,
        7002,
    ],
    target_body_ids=[
        5007,
        9004,
        5007,
    ],
    weights=[
        2.0,
        5.0,
        3.0,
    ],
)


print("Neuron mapping:")

for i in range(len(neuron_index)):
    print(
        neuron_index.to_body_id(i),
        "->",
        i,
    )


print("\nInternal connections:")

for source, target, weight in zip(
    connectivity.source_indices,
    connectivity.target_indices,
    connectivity.weights,
):
    print(
        f"{source} -> {target}, weight={weight}"
    )


print("\nPropagation test:")

# Спайкнули bodyId 1001 и 7002
spikes = np.array([
    True,   # 1001
    False,  # 5007
    True,   # 7002
    False,  # 9004
])

inputs = connectivity.propagate(spikes)

print(inputs)