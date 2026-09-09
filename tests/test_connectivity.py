import numpy as np

from flysim.simulation.connectivity import SparseConnectivity


def test_sparse_connectivity_propagates_spikes():
    connectivity = SparseConnectivity(
        size=4,
        source_indices=np.array([0, 0, 2]),
        target_indices=np.array([1, 3, 1]),
        weights=np.array([2.0, 5.0, 3.0]),
    )

    spikes = np.array([
        True,
        False,
        True,
        False,
    ])

    inputs = connectivity.propagate(spikes)

    expected = np.array([
        0.0,
        5.0,
        0.0,
        5.0,
    ])

    np.testing.assert_array_equal(
        inputs,
        expected,
    )


def test_inactive_neurons_produce_no_input():
    connectivity = SparseConnectivity(
        size=3,
        source_indices=np.array([0, 1]),
        target_indices=np.array([2, 2]),
        weights=np.array([2.0, 3.0]),
    )

    spikes = np.array([
        False,
        False,
        False,
    ])

    inputs = connectivity.propagate(spikes)

    np.testing.assert_array_equal(
        inputs,
        np.zeros(3),
    )


def test_multiple_connections_to_same_target_are_summed():
    connectivity = SparseConnectivity(
        size=3,
        source_indices=np.array([0, 1]),
        target_indices=np.array([2, 2]),
        weights=np.array([2.0, 3.0]),
    )

    spikes = np.array([
        True,
        True,
        False,
    ])

    inputs = connectivity.propagate(spikes)

    assert inputs[2] == 5.0