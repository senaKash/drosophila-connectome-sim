import numpy as np
import pandas as pd

from flysim.connectome.loader import build_soma_mapping
from flysim.connectome.neuron_index import NeuronIndex
from flysim.vision.image_input import sample_image_brightness
from flysim.vision.spatial_input import build_external_inputs


def test_sample_image_brightness():
    image = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.2, 0.7, 0.9],
        ]
    )

    columns = pd.DataFrame(
        {
            "hex1": [10, 11, 12],
            "hex2": [20, 20, 20],
            "xn": [0.0, 0.5, 1.0],
            "yn": [0.0, 0.0, 1.0],
        }
    )

    result = sample_image_brightness(
        image=image,
        columns=columns,
    )

    np.testing.assert_allclose(
        result["brightness"],
        np.array([
            0.0,
            0.5,
            0.9,
        ]),
    )


def test_stimulus_is_scattered_in_network_index_order():
    external_inputs = build_external_inputs(
        stimulus=np.array([1.6, 0.4, 0.8], dtype=np.float32),
        network_indices=np.array([4, 0, 2]),
        network_size=6,
    )

    np.testing.assert_allclose(external_inputs, [0.4, 0.0, 0.8, 0.0, 1.6, 0.0])


def test_neurons_without_soma_receive_zero_external_input():
    neuron_index = NeuronIndex([30, 10, 20, 40])
    annotations = pd.DataFrame(
        {"bodyId": [20, 99, 30], "somaLocation": [[2, 2, 2], [9, 9, 9], None]}
    )
    _, network_indices, _ = build_soma_mapping(annotations, neuron_index)

    external_inputs = build_external_inputs([1.6], network_indices, len(neuron_index))

    np.testing.assert_allclose(external_inputs, [0.0, 0.0, 1.6, 0.0])
    assert len(neuron_index) == 4


def test_external_inputs_have_full_network_size_even_for_empty_subset():
    neuron_index = NeuronIndex([70, 10, 30, 90, 50])
    external_inputs = build_external_inputs(
        stimulus=np.empty(0, dtype=np.float32),
        network_indices=np.empty(0, dtype=np.int32),
        network_size=len(neuron_index),
    )

    assert external_inputs.shape == (len(neuron_index),)
    assert external_inputs.dtype == np.dtype(float)
    assert not np.any(external_inputs)
