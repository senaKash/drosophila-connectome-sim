from pathlib import Path
from datetime import timedelta
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image

from brian2 import Network, SpikeMonitor, defaultclock, ms, mV, nA, start_scope

from flysim.connectome.loader import build_soma_mapping, load_malecns_feather
from flysim.rendering.brain_activity import BrainActivityRenderer
from flysim.vision.spatial_input import build_external_inputs, build_spatial_stimulus

from simulate_frame_brian2 import (
    ANNOTATIONS,
    WEIGHTS,
    STIMULUS_GAIN,
    build_network,
    prepare_weights,
)


INPUT = Path("data/input/bad_apple_frames")
OUTPUT = Path("results/bad_apple_frames")

START_FRAME = 1
FRAME_DURATION_MS = 25


def load_frame(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L"),
            dtype=np.float32,
        ) / 255.0


def main():
    start_scope()
    defaultclock.dt = 1 * ms
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading MaleCNS...")

    neuron_index, connectivity = load_malecns_feather(
        weights_path=WEIGHTS,
        annotations_path=ANNOTATIONS,
        min_weight=5,
    )

    prepare_weights(connectivity, neuron_index)

    annotations = pd.read_feather(
        ANNOTATIONS,
        columns=["bodyId", "somaLocation"],
    )

    _, network_indices, points = build_soma_mapping(
        annotations,
        neuron_index,
    )

    neurons, synapses = build_network(
        len(neuron_index),
        connectivity,
        np.zeros(len(neuron_index), dtype=np.float32),
    )

    monitor = SpikeMonitor(neurons, record=False)
    network = Network(neurons, synapses, monitor)

    previous_counts = np.zeros(
        len(neuron_index),
        dtype=np.int32,
    )

    renderer = BrainActivityRenderer(
        points=points,
        camera_points=points,
        network_indices=network_indices,
    )

    frame_paths = sorted(
        path
        for path in INPUT.glob("*.png")
        if path.stem.isdigit() and int(path.stem) >= START_FRAME
    )

    if not frame_paths:
        raise RuntimeError(f"No frames found in {INPUT}")

    total_frames = len(frame_paths)
    render_started = perf_counter()

    print(f"Frames to process: {total_frames}")

    for number, source in enumerate(frame_paths, start=1):
        frame_index = int(source.stem)

        frame = load_frame(source)

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

        # Меняем только внешний stimulus.
        # Состояние мозга между кадрами сохраняется.
        neurons.I_ext = inputs * nA

        network.run(FRAME_DURATION_MS * ms)

        counts = np.asarray(monitor.count)
        new_spikes = counts - previous_counts
        previous_counts[:] = counts

        colors = renderer.build_colors(
            voltage=np.asarray(neurons.v / mV),
            direct_input=inputs > 0,
            ever_spiked=new_spikes > 0,
            recent_spikes=new_spikes > 0,
        )

        target = OUTPUT / f"{frame_index:06d}.png"
        renderer.render(colors, output_path=target)

        elapsed = perf_counter() - render_started
        remaining = (elapsed / number) * (total_frames - number)

        print(
            f"Frame {number}/{total_frames} "
            f"(source {frame_index}) | "
            f"spikes: {int(new_spikes.sum())} | "
            f"elapsed: {timedelta(seconds=int(elapsed))} | "
            f"ETA: {timedelta(seconds=int(remaining))}"
        )

    total_time = perf_counter() - render_started
    print(f"Done in {timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    main()