from flysim.connectome.neuron_index import NeuronIndex


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