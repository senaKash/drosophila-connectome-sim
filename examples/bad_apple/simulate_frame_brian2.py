from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image

from brian2 import (
    Mohm,
    Network,
    NeuronGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    ms,
    mV,
    nA,
    start_scope,
)

from flysim.connectome.loader import (
    build_soma_mapping,
    load_malecns_feather,
)
from flysim.connectome.neurotransmitters import load_neuron_signs
from flysim.rendering.brain_activity import BrainActivityRenderer
from flysim.simulation.weight_transform import build_synaptic_currents
from flysim.vision.spatial_input import (
    build_external_inputs,
    build_spatial_stimulus,
)


ANNOTATIONS = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "data/raw/body-neurotransmitters-male-cns-v1.0.feather"
FRAME = "data/input/test.png"

SIMULATION_MS = 100
STEP_MS = 25

# Экспериментальные параметры модели, не физиологические измерения.
STIMULUS_GAIN = 1.6
SYNAPTIC_SCALE = 10.0
TAU_SYN_MS = 5.0


def load_stimulus(neuron_index):
    annotations = pd.read_feather(
        ANNOTATIONS,
        columns=["bodyId", "somaLocation"],
    )

    _, network_indices, points = build_soma_mapping(
        annotations,
        neuron_index,
    )

    with Image.open(FRAME) as image:
        frame = np.asarray(
            image.convert("L"),
            dtype=np.float32,
        ) / 255.0

    stimulus = build_spatial_stimulus(
        points,
        frame,
        gain=STIMULUS_GAIN,
    )

    inputs = build_external_inputs(
        stimulus,
        network_indices,
        len(neuron_index),
    )

    return inputs, network_indices, points


def prepare_weights(connectivity, neuron_index):
    signs = load_neuron_signs(
        NEUROTRANSMITTERS,
        neuron_index,
    )

    weights = build_synaptic_currents(
        connectivity,
        signs,
        synaptic_scale=SYNAPTIC_SCALE,
    )

    connectivity.weights = weights.astype(
        np.float32,
        copy=False,
    )


def build_network(size, connectivity, inputs):
    equations = """
    dv/dt = (v_rest - v + resistance * (I_ext + I_syn)) / tau_m : volt
    dI_syn/dt = -I_syn / tau_syn : amp
    I_ext : amp
    """

    neurons = NeuronGroup(
        size,
        equations,
        threshold="v >= v_threshold",
        reset="v = v_reset",
        method="euler",
        namespace={
            "v_rest": -65 * mV,
            "v_reset": -65 * mV,
            "v_threshold": -50 * mV,
            "tau_m": 20 * ms,
            "tau_syn": TAU_SYN_MS * ms,
            "resistance": 10 * Mohm,
        },
    )

    neurons.v = -65 * mV
    neurons.I_ext = inputs * nA

    synapses = Synapses(
        neurons,
        neurons,
        model="w : amp",
        on_pre="I_syn_post += w",
    )

    synapses.connect(
        i=connectivity.source_indices,
        j=connectivity.target_indices,
    )
    synapses.w = connectivity.weights * nA

    return neurons, synapses


def simulate(neurons, synapses):
    monitor = SpikeMonitor(neurons, record=False)
    network = Network(neurons, synapses, monitor)

    previous_counts = np.zeros(
        len(neurons),
        dtype=np.int32,
    )

    snapshots = []

    for time_ms in range(
        STEP_MS,
        SIMULATION_MS + 1,
        STEP_MS,
    ):
        network.run(STEP_MS * ms)

        counts = np.asarray(monitor.count)
        new_spikes = counts - previous_counts
        previous_counts[:] = counts

        snapshots.append((
            time_ms,
            np.asarray(neurons.v / mV, dtype=np.float32).copy(),
            (counts > 0).copy(),
            (new_spikes > 0).copy(),
        ))

        print(
            f"{time_ms - STEP_MS + 1:3d}-{time_ms:3d} ms: "
            f"{new_spikes.sum()} spikes"
        )

    return monitor, snapshots


def main():
    start_scope()
    defaultclock.dt = 1 * ms
    started = perf_counter()

    print("Loading MaleCNS...")

    neuron_index, connectivity = load_malecns_feather(
        weights_path=WEIGHTS,
        annotations_path=ANNOTATIONS,
        min_weight=5,
    )

    prepare_weights(connectivity, neuron_index)

    inputs, network_indices, points = load_stimulus(
        neuron_index
    )

    print("Building Brian2 network...")

    neurons, synapses = build_network(
        len(neuron_index),
        connectivity,
        inputs,
    )

    print("Neurons:", len(neuron_index))
    print("Synapses:", len(connectivity.weights))
    print("Stimulated:", np.count_nonzero(inputs > 0))

    monitor, snapshots = simulate(
        neurons,
        synapses,
    )

    counts = np.asarray(monitor.count)
    direct_input = inputs > 0
    downstream = ~direct_input

    print()
    print("Total spikes:", int(counts.sum()))
    print(
        "Downstream spikes:",
        int(counts[downstream].sum()),
    )
    print(
        "Downstream neurons:",
        np.count_nonzero(counts[downstream]),
    )
    print(
        f"Total runtime: {perf_counter() - started:.2f} s"
    )

    renderer = BrainActivityRenderer(
        points=points,
        camera_points=points,
        network_indices=network_indices,
    )

    for time_ms, voltage, ever_spiked, recent_spikes in snapshots:
        print(f"Rendering {time_ms} ms...")

        colors = renderer.build_colors(
            voltage=voltage,
            direct_input=direct_input,
            ever_spiked=ever_spiked,
            recent_spikes=recent_spikes,
        )

        renderer.render(colors)


if __name__ == "__main__":
    main()