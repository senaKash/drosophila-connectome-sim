import numpy as np

from flysim.connectome.loader import load_malecns_feather
from flysim.connectome.neurotransmitters import load_neuron_signs
from flysim.simulation.weight_transform import (
    build_synaptic_currents,
)


WEIGHTS_PATH = (
    "data/raw/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

NEUROTRANSMITTERS_PATH = (
    "data/raw/"
    "body-neurotransmitters-male-cns-v1.0.feather"
)


print("Loading real MaleCNS...")

neuron_index, connectivity = load_malecns_feather(
    weights_path=WEIGHTS_PATH,
    annotations_path=ANNOTATIONS_PATH,
    min_weight=5,
)

print("Loading neurotransmitters...")

neuron_signs = load_neuron_signs(
    path=NEUROTRANSMITTERS_PATH,
    neuron_index=neuron_index,
)


synaptic_scale = 2.0

print("Building synaptic currents...")

currents = build_synaptic_currents(
    connectivity=connectivity,
    neuron_signs=neuron_signs,
    synaptic_scale=synaptic_scale,
)


print()
print("REAL MALECNS SYNAPTIC MODEL")
print("---------------------------")
print("Neurons:     ", len(neuron_index))
print("Connections: ", len(currents))
print("Scale:       ", synaptic_scale, "nA")

print()
print("Excitatory edges:", np.count_nonzero(currents > 0))
print("Inhibitory edges:", np.count_nonzero(currents < 0))
print("Unsigned edges:  ", np.count_nonzero(currents == 0))

print()
print("Current statistics:")
print("Min:    ", np.min(currents), "nA")
print("Median: ", np.median(currents), "nA")
print("Mean:   ", np.mean(currents), "nA")
print("Max:    ", np.max(currents), "nA")


print()
print("FIRST REAL CONNECTIONS")

shown = 0

for i in range(len(currents)):
    if currents[i] == 0:
        continue

    source_index = int(
        connectivity.source_indices[i]
    )

    target_index = int(
        connectivity.target_indices[i]
    )

    source_body = neuron_index.to_body_id(
        source_index
    )

    target_body = neuron_index.to_body_id(
        target_index
    )

    print(
        f"{source_body} -> {target_body} | "
        f"raw={connectivity.weights[i]:.0f} | "
        f"sign={neuron_signs[source_index]:+d} | "
        f"current={currents[i]:+.6f} nA"
    )

    shown += 1

    if shown == 10:
        break