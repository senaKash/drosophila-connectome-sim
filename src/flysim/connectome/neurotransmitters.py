import numpy as np
import pandas as pd

from flysim.connectome.neuron_index import NeuronIndex


TRANSMITTER_SIGNS = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unclear": 0,
}


def load_neuron_signs(
    path: str,
    neuron_index: NeuronIndex,
) -> np.ndarray:
    neurotransmitters = pd.read_feather(
        path,
        columns=[
            "body",
            "consensus_nt",
        ],
    )

    nt_by_body = neurotransmitters.set_index(
        "body"
    )["consensus_nt"]

    neuron_nt = (
        nt_by_body
        .reindex(neuron_index.body_ids)
        .fillna("missing")
    )

    signs = np.array(
        [
            TRANSMITTER_SIGNS.get(nt, 0)
            for nt in neuron_nt
        ],
        dtype=np.int8,
    )

    return signs