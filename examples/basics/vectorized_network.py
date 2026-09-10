import numpy as np

from flysim.simulation.population import LIFPopulation
from flysim.simulation.connectivity import SparseConnectivity
from flysim.simulation.vectorized_network import VectorizedNetwork


population = LIFPopulation(size=3)

connectivity = SparseConnectivity(
    size=3,
    source_indices=np.array([
        0,  # neuron 0 -> neuron 1
    ]),
    target_indices=np.array([
        1,
    ]),
    weights=np.array([
        2.0,
    ]),
)

network = VectorizedNetwork(
    population=population,
    connectivity=connectivity,
)


for t in range(35):

    external_inputs = np.array([
        2.0,  # постоянно стимулируем neuron 0
        0.0,
        0.0,
    ])

    input_before_step = network.pending_inputs.copy()

    spikes = network.step(
        external_inputs=external_inputs,
        dt=1.0,
    )

    print(
        f"time={t:2d} ms | "
        f"V={network.population.voltage.round(2)} | "
        f"spikes={spikes} | "
        f"input={input_before_step} | "
        f"next={network.pending_inputs}"
    )