from pathlib import Path

import numpy as np
import pandas as pd

from flysim.connectome.loader import (
    build_connectivity,
    load_connectivity_csv,
    load_malecns_feather,
)

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