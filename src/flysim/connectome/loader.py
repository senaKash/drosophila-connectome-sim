import numpy as np
import csv


from flysim.connectome.neuron_index import NeuronIndex
from flysim.simulation.connectivity import SparseConnectivity


def build_connectivity(
    source_body_ids,
    target_body_ids,
    weights,
) -> tuple[NeuronIndex, SparseConnectivity]:

    source_body_ids = np.asarray(
        source_body_ids,
        dtype=np.int64,
    )

    target_body_ids = np.asarray(
        target_body_ids,
        dtype=np.int64,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if not (
        len(source_body_ids)
        == len(target_body_ids)
        == len(weights)
    ):
        raise ValueError(
            "source_body_ids, target_body_ids and weights "
            "must have the same length"
        )

    # Собираем все уникальные bodyId,
    # встречающиеся в таблице связей.
    body_ids = np.unique(
        np.concatenate([
            source_body_ids,
            target_body_ids,
        ])
    )

    neuron_index = NeuronIndex(
        body_ids.tolist()
    )

    # Переводим реальные bodyId
    # во внутренние индексы 0..N-1.
    source_indices = np.array(
        [
            neuron_index.to_index(body_id)
            for body_id in source_body_ids
        ],
        dtype=int,
    )

    target_indices = np.array(
        [
            neuron_index.to_index(body_id)
            for body_id in target_body_ids
        ],
        dtype=int,
    )

    connectivity = SparseConnectivity(
        size=len(neuron_index),
        source_indices=source_indices,
        target_indices=target_indices,
        weights=weights,
    )

    return neuron_index, connectivity

def load_connectivity_csv(path: str):
    source_body_ids = []
    target_body_ids = []
    weights = []

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "source_bodyId",
            "target_bodyId",
            "weight",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain columns: "
                "source_bodyId, target_bodyId, weight"
            )

        for row in reader:
            source_body_ids.append(
                int(row["source_bodyId"])
            )

            target_body_ids.append(
                int(row["target_bodyId"])
            )

            weights.append(
                float(row["weight"])
            )

    return build_connectivity(
        source_body_ids=source_body_ids,
        target_body_ids=target_body_ids,
        weights=weights,
    )