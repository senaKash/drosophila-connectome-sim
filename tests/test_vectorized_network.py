import numpy as np

from flysim.simulation.population import LIFPopulation
from flysim.simulation.connectivity import SparseConnectivity
from flysim.simulation.vectorized_network import VectorizedNetwork


def test_spike_creates_pending_input():
    population = LIFPopulation(size=2)

    connectivity = SparseConnectivity(
        size=2,
        source_indices=np.array([0]),
        target_indices=np.array([1]),
        weights=np.array([2.0]),
    )

    network = VectorizedNetwork(
        population=population,
        connectivity=connectivity,
    )

    while True:
        spikes = network.step(
            external_inputs=np.array([2.0, 0.0])
        )

        if spikes[0]:
            break

    assert network.pending_inputs[1] == 2.0


def test_pending_input_affects_target_on_next_step():
    population = LIFPopulation(size=2)

    connectivity = SparseConnectivity(
        size=2,
        source_indices=np.array([0]),
        target_indices=np.array([1]),
        weights=np.array([2.0]),
    )

    network = VectorizedNetwork(
        population=population,
        connectivity=connectivity,
    )

    while True:
        spikes = network.step(
            external_inputs=np.array([2.0, 0.0])
        )

        if spikes[0]:
            break

    voltage_before = population.voltage[1]

    network.step()

    assert population.voltage[1] > voltage_before


def test_network_rejects_size_mismatch():
    population = LIFPopulation(size=3)

    connectivity = SparseConnectivity(
        size=2,
        source_indices=np.array([0]),
        target_indices=np.array([1]),
        weights=np.array([1.0]),
    )

    try:
        VectorizedNetwork(
            population=population,
            connectivity=connectivity,
        )

        assert False, "Expected ValueError"

    except ValueError:
        pass