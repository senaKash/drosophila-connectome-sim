import numpy as np

from flysim.simulation.connectivity import SparseConnectivity
from flysim.simulation.weight_transform import (
    normalize_incoming_weights,
)

from flysim.simulation.weight_transform import (
    normalize_incoming_weights,
    apply_neurotransmitter_signs,
)

from flysim.simulation.weight_transform import (
    normalize_incoming_weights,
    apply_neurotransmitter_signs,
    build_signed_normalized_weights,
)

from flysim.simulation.weight_transform import (
    normalize_incoming_weights,
    apply_neurotransmitter_signs,
    build_signed_normalized_weights,
    scale_synaptic_weights,
    build_synaptic_currents,
)

def test_normalize_incoming_weights():
    connectivity = SparseConnectivity(
        size=4,
        source_indices=np.array([0, 1, 2]),
        target_indices=np.array([3, 3, 3]),
        weights=np.array([10.0, 20.0, 70.0]),
    )

    normalized = normalize_incoming_weights(
        connectivity
    )

    np.testing.assert_allclose(
        normalized,
        np.array([
            0.1,
            0.2,
            0.7,
        ]),
    )

def test_apply_neurotransmitter_signs():
    normalized_weights = np.array([
        0.1,
        0.2,
        0.7,
    ])

    source_indices = np.array([
        0,
        1,
        2,
    ])

    neuron_signs = np.array([
        1,
        -1,
        1,
        0,
    ])

    signed = apply_neurotransmitter_signs(
        normalized_weights=normalized_weights,
        source_indices=source_indices,
        neuron_signs=neuron_signs,
    )

    np.testing.assert_allclose(
        signed,
        np.array([
            0.1,
            -0.2,
            0.7,
        ]),
    )

def test_build_signed_normalized_weights():
    connectivity = SparseConnectivity(
        size=4,
        source_indices=np.array([0, 1, 2]),
        target_indices=np.array([3, 3, 3]),
        weights=np.array([10.0, 20.0, 70.0]),
    )

    neuron_signs = np.array([
        1,
        -1,
        1,
        0,
    ])

    weights = build_signed_normalized_weights(
        connectivity=connectivity,
        neuron_signs=neuron_signs,
    )

    np.testing.assert_allclose(
        weights,
        np.array([
            0.1,
            -0.2,
            0.7,
        ]),
    )


    def test_build_synaptic_currents():
        connectivity = SparseConnectivity(
            size=4,
            source_indices=np.array([0, 1, 2]),
            target_indices=np.array([3, 3, 3]),
            weights=np.array([10.0, 20.0, 70.0]),
        )

        neuron_signs = np.array([
            1,
            -1,
            1,
            0,
        ])

        currents = build_synaptic_currents(
            connectivity=connectivity,
            neuron_signs=neuron_signs,
            synaptic_scale=2.0,
        )

        np.testing.assert_allclose(
            currents,
            np.array([
                0.2,
                -0.4,
                1.4,
            ]),
        )