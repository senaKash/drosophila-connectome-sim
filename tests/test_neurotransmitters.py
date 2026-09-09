import pandas as pd

from flysim.connectome.neuron_index import NeuronIndex
from flysim.connectome.neurotransmitters import (
    load_neuron_signs,
)


def test_load_neuron_signs(tmp_path):
    path = tmp_path / "neurotransmitters.feather"

    df = pd.DataFrame(
        {
            "body": [
                1001,
                1002,
                1003,
                1004,
            ],
            "consensus_nt": [
                "acetylcholine",
                "gaba",
                "glutamate",
                "dopamine",
            ],
        }
    )

    df.to_feather(path)

    neuron_index = NeuronIndex(
        [
            1001,
            1002,
            1003,
            1004,
            9999,
        ]
    )

    signs = load_neuron_signs(
        path=str(path),
        neuron_index=neuron_index,
    )

    assert signs.tolist() == [
        1,
        -1,
        -1,
        0,
        0,
    ]