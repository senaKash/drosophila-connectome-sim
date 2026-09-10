import pandas as pd
import pytest

from flysim.connectome.loader import load_malecns_feather
from flysim.connectome.neuron_index import NeuronIndex, build_malecns_neuron_index


def test_body_id_to_index():
    index = NeuronIndex([
        1001,
        5007,
        9004,
    ])

    assert index.to_index(1001) == 0
    assert index.to_index(5007) == 1
    assert index.to_index(9004) == 2


def test_index_to_body_id():
    index = NeuronIndex([
        1001,
        5007,
        9004,
    ])

    assert index.to_body_id(0) == 1001
    assert index.to_body_id(1) == 5007
    assert index.to_body_id(2) == 9004


def test_duplicate_body_ids_are_rejected():
    try:
        NeuronIndex([
            1001,
            1001,
        ])

        assert False, "Expected ValueError"

    except ValueError:
        pass


def test_unknown_body_id_is_rejected():
    index = NeuronIndex([
        1001,
        5007,
    ])

    try:
        index.to_index(9999)

        assert False, "Expected KeyError"

    except KeyError:
        pass


@pytest.mark.parametrize("min_weight", [1, 5, 100])
def test_annotation_index_matches_full_malecns_loader(tmp_path, min_weight):
    annotations = pd.DataFrame(
        {
            "bodyId": [9004, 7002, 1001, 5007, 7002, 9998, 9999],
            "superclass": ["sensory"] * 5 + ["unknown_TbC", None],
        }
    )
    weights = pd.DataFrame(
        {
            "body_pre": [1001, 5007, 9999],
            "body_post": [5007, 7002, 1001],
            "weight": [5, 2, 10],
        }
    )
    annotations_path = tmp_path / "annotations.feather"
    weights_path = tmp_path / "weights.feather"
    annotations.to_feather(annotations_path)
    weights.to_feather(weights_path)

    annotation_index = build_malecns_neuron_index(annotations)
    loaded_index, connectivity = load_malecns_feather(
        weights_path=str(weights_path),
        annotations_path=str(annotations_path),
        min_weight=min_weight,
    )

    assert annotation_index.body_ids == loaded_index.body_ids == (1001, 5007, 7002, 9004)
    assert annotation_index.id_to_index == loaded_index.id_to_index
    assert connectivity.size == len(annotation_index)
