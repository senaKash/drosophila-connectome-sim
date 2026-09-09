import numpy as np

from flysim.simulation.connectivity import SparseConnectivity


def normalize_incoming_weights(
    connectivity: SparseConnectivity,
) -> np.ndarray:
    raw_weights = connectivity.weights

    incoming_totals = np.bincount(
        connectivity.target_indices,
        weights=raw_weights,
        minlength=connectivity.size,
    )

    normalized_weights = (
        raw_weights
        / incoming_totals[connectivity.target_indices]
    )

    return normalized_weights


def apply_neurotransmitter_signs(
    normalized_weights: np.ndarray,
    source_indices: np.ndarray,
    neuron_signs: np.ndarray,
) -> np.ndarray:
    edge_signs = neuron_signs[source_indices]

    return normalized_weights * edge_signs

def build_signed_normalized_weights(
    connectivity: SparseConnectivity,
    neuron_signs: np.ndarray,
) -> np.ndarray:
    normalized_weights = normalize_incoming_weights(
        connectivity
    )

    return apply_neurotransmitter_signs(
        normalized_weights=normalized_weights,
        source_indices=connectivity.source_indices,
        neuron_signs=neuron_signs,
    )

def build_synaptic_currents(
    connectivity: SparseConnectivity,
    neuron_signs: np.ndarray,
    synaptic_scale: float,
) -> np.ndarray:
    signed_weights = build_signed_normalized_weights(
        connectivity=connectivity,
        neuron_signs=neuron_signs,
    )

    return scale_synaptic_weights(
        signed_weights=signed_weights,
        synaptic_scale=synaptic_scale,
    )

def scale_synaptic_weights(
    signed_weights: np.ndarray,
    synaptic_scale: float,
) -> np.ndarray:
    if synaptic_scale < 0:
        raise ValueError(
            "synaptic_scale must be non-negative"
        )

    return signed_weights * synaptic_scale