from pathlib import Path

import numpy as np

from flysim.connectome.loader import (
    build_connectivity,
    load_connectivity_csv,
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