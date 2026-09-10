import numpy as np


def build_spatial_stimulus(points, frame, gain=1.0):
    """Sample one grayscale frame on the existing fixed X/Y projection."""
    center = np.median(points, axis=0)

    p_low = np.percentile(points, 5, axis=0)
    p_high = np.percentile(points, 95, axis=0)

    robust_span = p_high - p_low
    half = robust_span.max() * 0.55 / 2

    # Координаты нейронов на нашей фиксированной X/Y-проекции
    u = (points[:, 0] - (center[0] - half)) / (2 * half)
    v = (points[:, 1] - (center[1] - half)) / (2 * half)

    # Нейроны за пределами картинки ток не получают
    visible = (
        (u >= 0.0)
        & (u <= 1.0)
        & (v >= 0.0)
        & (v <= 1.0)
    )

    current = np.zeros(len(points), dtype=np.float32)

    h, w = frame.shape

    x = (u[visible] * (w - 1)).astype(int)
    y = ((1.0 - v[visible]) * (h - 1)).astype(int)

    brightness = frame[y, x]

    current[visible] = brightness * gain

    return current


def build_external_inputs(stimulus, network_indices, network_size):
    """Place soma-subset currents into a full network-sized input array."""
    stimulus = np.asarray(stimulus)
    network_indices = np.asarray(network_indices)
    if stimulus.ndim != 1 or network_indices.shape != stimulus.shape:
        raise ValueError("stimulus and network_indices must be matching 1D arrays")
    if not np.issubdtype(network_indices.dtype, np.integer):
        raise ValueError("network_indices must be integers")
    if np.any(network_indices < 0) or np.any(network_indices >= network_size):
        raise ValueError("network_indices must refer to neurons in the network")

    # Match VectorizedNetwork's float64 input to avoid conversion on every step.
    external_inputs = np.zeros(network_size, dtype=float)
    external_inputs[network_indices] = stimulus
    return external_inputs
