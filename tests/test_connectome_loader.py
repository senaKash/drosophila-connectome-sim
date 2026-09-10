from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flysim.connectome.loader import (
    build_connectivity,
    build_soma_mapping,
    load_connectivity_csv,
    load_malecns_feather,
)
from flysim.connectome.neuron_index import NeuronIndex


@pytest.fixture
def soma_mapping_case():
    # Network order deliberately differs from annotation and bodyId order.
    neuron_index = NeuronIndex([7002, 1001, 9004, 5007, 8000])
    annotations = pd.DataFrame(
        {
            "bodyId": [5007, 9999, 1001, 7002, 9004],
            "somaLocation": [
                np.array([5, 6, 7]),
                [99, 99, 99],
                None,
                "[7, 8, 9]",
                (9, 10, 11),
            ],
        },
        index=[40, 10, 70, 30, 20],
    )
    return annotations, neuron_index


def test_soma_mapping_points_and_indices_have_same_length(soma_mapping_case):
    body_ids, network_indices, points = build_soma_mapping(*soma_mapping_case)

    assert len(points) == len(network_indices) == len(body_ids) == 3
    assert points.shape == (3, 3)
    np.testing.assert_array_equal(points, [[5, 6, 7], [7, 8, 9], [9, 10, 11]])


def test_soma_mapping_network_indices_are_valid(soma_mapping_case):
    _, neuron_index = soma_mapping_case
    _, network_indices, _ = build_soma_mapping(*soma_mapping_case)

    assert np.issubdtype(network_indices.dtype, np.integer)
    assert np.all(network_indices >= 0)
    assert np.all(network_indices < len(neuron_index))


def test_soma_mapping_body_ids_match_existing_index(soma_mapping_case):
    _, neuron_index = soma_mapping_case
    body_ids, network_indices, _ = build_soma_mapping(*soma_mapping_case)

    np.testing.assert_array_equal(body_ids, [5007, 7002, 9004])
    np.testing.assert_array_equal(network_indices, [3, 0, 2])
    for body_id, network_index in zip(body_ids, network_indices):
        assert neuron_index.to_index(body_id) == network_index
        assert neuron_index.to_body_id(network_index) == body_id


def test_soma_mapping_excludes_neurons_outside_network(soma_mapping_case):
    _, neuron_index = soma_mapping_case
    body_ids, network_indices, points = build_soma_mapping(*soma_mapping_case)

    assert 9999 not in body_ids
    assert set(body_ids).issubset(neuron_index.body_ids)
    assert not np.any(np.all(points == [99, 99, 99], axis=1))
    assert len(np.unique(network_indices)) == len(points)


def test_soma_mapping_keeps_full_network_and_annotations(soma_mapping_case):
    annotations, neuron_index = soma_mapping_case
    annotations_before = annotations.copy(deep=True)
    body_ids_before = neuron_index.body_ids
    indices_before = neuron_index.id_to_index.copy()

    body_ids, _, _ = build_soma_mapping(annotations, neuron_index)

    # 1001 has no soma; 8000 has no annotation row. Both remain in the network.
    assert 1001 not in body_ids
    assert 8000 not in body_ids
    assert neuron_index.body_ids == body_ids_before
    assert neuron_index.id_to_index == indices_before
    assert len(neuron_index) == 5
    pd.testing.assert_frame_equal(annotations, annotations_before)


@pytest.mark.parametrize("body_ids", [[1001], [], [9998]])
def test_soma_mapping_handles_no_renderable_neurons(body_ids):
    annotations = pd.DataFrame(
        {"bodyId": [1001, 9999], "somaLocation": [None, [1, 2, 3]]}
    )
    neuron_index = NeuronIndex(body_ids)

    mapped_ids, network_indices, points = build_soma_mapping(annotations, neuron_index)

    assert mapped_ids.shape == network_indices.shape == (0,)
    assert points.shape == (0, 3)
    assert neuron_index.body_ids == tuple(body_ids)


@pytest.mark.parametrize("location", [[1, 2], [1, 2, 3, 4], [1, np.nan, 3]])
def test_soma_mapping_rejects_invalid_coordinates(location):
    annotations = pd.DataFrame({"bodyId": [1001], "somaLocation": [location]})

    with pytest.raises(ValueError, match="three finite coordinates"):
        build_soma_mapping(annotations, NeuronIndex([1001]))


def test_soma_mapping_rejects_duplicate_network_somas():
    annotations = pd.DataFrame(
        {"bodyId": [1001, 1001], "somaLocation": [[1, 2, 3], [4, 5, 6]]}
    )

    with pytest.raises(ValueError, match="unique per network bodyId"):
        build_soma_mapping(annotations, NeuronIndex([1001]))


def test_build_connectivity():
    neuron_index, connectivity = build_connectivity(
        source_body_ids=[
            1001,
            1001,
            7002,
        ],
        target_body_ids=[
            5007,
            9004,
            5007,
        ],
        weights=[
            2.0,
            5.0,
            3.0,
        ],
    )

    assert len(neuron_index) == 4

    np.testing.assert_array_equal(
        connectivity.source_indices,
        np.array([0, 0, 2]),
    )

    np.testing.assert_array_equal(
        connectivity.target_indices,
        np.array([1, 3, 1]),
    )

    np.testing.assert_array_equal(
        connectivity.weights,
        np.array([2.0, 5.0, 3.0]),
    )


def test_load_connectivity_csv(tmp_path: Path):
    csv_path = tmp_path / "connectome.csv"

    csv_path.write_text(
        (
            "source_bodyId,target_bodyId,weight\n"
            "1001,5007,2.0\n"
            "1001,9004,5.0\n"
            "7002,5007,3.0\n"
        ),
        encoding="utf-8",
    )

    neuron_index, connectivity = load_connectivity_csv(
        str(csv_path)
    )

    assert len(neuron_index) == 4

    np.testing.assert_array_equal(
        connectivity.weights,
        np.array([2.0, 5.0, 3.0]),
    )


def test_csv_requires_expected_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"

    csv_path.write_text(
        (
            "source,target,value\n"
            "1001,5007,2.0\n"
        ),
        encoding="utf-8",
    )

    try:
        load_connectivity_csv(str(csv_path))

        assert False, "Expected ValueError"

    except ValueError:
        pass


def test_load_malecns_feather_filters_graph(tmp_path: Path):
    annotations_path = tmp_path / "annotations.feather"
    weights_path = tmp_path / "weights.feather"

    annotations = pd.DataFrame(
        {
            "bodyId": [
                1001,
                5007,
                7002,
                9004,
                9999,
            ],
            "superclass": [
                "sensory",
                "sensory",
                "sensory",
                "sensory",
                "tbc",
            ],
        }
    )

    weights = pd.DataFrame(
        {
            "body_pre": [
                1001,
                1001,
                7002,
                7002,
                9999,
                5007,
            ],
            "body_post": [
                5007,
                9004,
                5007,
                9004,
                1001,
                9999,
            ],
            "weight": [
                2,
                5,
                3,
                7,
                10,
                10,
            ],
        }
    )

    annotations.to_feather(annotations_path)
    weights.to_feather(weights_path)

    neuron_index, connectivity = load_malecns_feather(
        weights_path=str(weights_path),
        annotations_path=str(annotations_path),
        min_weight=5,
    )

    assert len(neuron_index) == 4

    assert neuron_index.body_ids == (
        1001,
        5007,
        7002,
        9004,
    )

    np.testing.assert_array_equal(
        connectivity.source_indices,
        np.array([0, 2]),
    )

    np.testing.assert_array_equal(
        connectivity.target_indices,
        np.array([3, 3]),
    )

    np.testing.assert_array_equal(
        connectivity.weights,
        np.array([5.0, 7.0]),
    )
