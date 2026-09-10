import ast

import numpy as np
import csv
import pyarrow.feather as feather


from flysim.connectome.neuron_index import NeuronIndex, build_malecns_neuron_index
from flysim.simulation.connectivity import SparseConnectivity


def build_soma_mapping(
    annotations,
    neuron_index: NeuronIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned body_ids, network_indices and soma points (M, 3).

    annotations is a DataFrame with bodyId and somaLocation columns.
    Rows retain annotation order; network_indices always come from the
    supplied NeuronIndex. Neurons outside that index and rows without a
    somaLocation are omitted. The full network index is never modified.
    """
    required_columns = {"bodyId", "somaLocation"}
    if not required_columns.issubset(annotations.columns):
        raise ValueError("annotations must contain bodyId and somaLocation")

    soma_rows = annotations[
        annotations["bodyId"].isin(neuron_index.body_ids)
        & annotations["somaLocation"].notna()
    ]

    if soma_rows["bodyId"].duplicated().any():
        raise ValueError("somaLocation must be unique per network bodyId")

    body_ids = soma_rows["bodyId"].to_numpy(dtype=np.int64)
    network_indices = np.array(
        [neuron_index.to_index(body_id) for body_id in body_ids],
        dtype=int,
    )

    points = np.empty((len(body_ids), 3), dtype=float)
    for i, value in enumerate(soma_rows["somaLocation"]):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        location = np.asarray(value, dtype=float)
        if location.shape != (3,) or not np.isfinite(location).all():
            raise ValueError(
                f"somaLocation for bodyId {body_ids[i]} "
                "must contain three finite coordinates"
            )
        points[i] = location

    return body_ids, network_indices, points


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

def load_malecns_feather(
    weights_path: str,
    annotations_path: str,
    min_weight: int = 5,
) -> tuple[NeuronIndex, SparseConnectivity]:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    # --------------------------------------------------
    # 1. Получаем список валидных нейронов
    # --------------------------------------------------

    annotations = pd.read_feather(
        annotations_path,
        columns=["bodyId", "superclass"],
    )

    neuron_index = build_malecns_neuron_index(annotations)
    valid_body_ids = np.asarray(neuron_index.body_ids, dtype=np.int64)

    valid_body_ids_arrow = pa.array(valid_body_ids)

    source_chunks = []
    target_chunks = []
    weight_chunks = []

    # --------------------------------------------------
    # 2. Читаем граф по частям
    # --------------------------------------------------

    with pa.memory_map(weights_path, "r") as source:
        reader = ipc.open_file(source)

        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)

            body_pre = batch.column(
                batch.schema.get_field_index("body_pre")
            )

            body_post = batch.column(
                batch.schema.get_field_index("body_post")
            )

            weight = batch.column(
                batch.schema.get_field_index("weight")
            )

            valid_pre = pc.is_in(
                body_pre,
                value_set=valid_body_ids_arrow,
            )

            valid_post = pc.is_in(
                body_post,
                value_set=valid_body_ids_arrow,
            )

            strong_enough = pc.greater_equal(
                weight,
                min_weight,
            )

            mask = pc.and_(
                pc.and_(valid_pre, valid_post),
                strong_enough,
            )

            filtered_pre = pc.filter(
                body_pre,
                mask,
            ).to_numpy()

            filtered_post = pc.filter(
                body_post,
                mask,
            ).to_numpy()

            filtered_weight = pc.filter(
                weight,
                mask,
            ).to_numpy()

            # valid_body_ids отсортирован.
            # Поэтому searchsorted очень быстро переводит
            # реальные bodyId -> индексы 0..N-1.
            source_indices = np.searchsorted(
                valid_body_ids,
                filtered_pre,
            )

            target_indices = np.searchsorted(
                valid_body_ids,
                filtered_post,
            )

            source_chunks.append(source_indices)
            target_chunks.append(target_indices)
            weight_chunks.append(filtered_weight)

    # --------------------------------------------------
    # 3. Склеиваем обработанные batch'и
    # --------------------------------------------------

    source_indices = np.concatenate(source_chunks)
    target_indices = np.concatenate(target_chunks)
    weights = np.concatenate(weight_chunks)

    connectivity = SparseConnectivity(
        size=len(neuron_index),
        source_indices=source_indices,
        target_indices=target_indices,
        weights=weights,
    )

    return neuron_index, connectivity
