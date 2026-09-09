import pandas as pd

from flysim.vision.columns import (
    add_normalized_coordinates,
    build_right_eye_columns,
)


def test_build_right_eye_columns():
    annotations = pd.DataFrame(
        {
            "type": ["L1", "L1", "L1", "L2"],
            "somaSide": ["R", "R", "L", "R"],
            "assignedOlHex1": [10, 11, 12, 13],
            "assignedOlHex2": [20, 21, 22, 23],
        }
    )

    columns = build_right_eye_columns(annotations)

    assert len(columns) == 2
    assert columns["hex1"].tolist() == [10, 11]
    assert columns["hex2"].tolist() == [20, 21]


def test_add_normalized_coordinates():
    columns = pd.DataFrame(
        {
            "hex1": [10, 20],
            "hex2": [10, 20],
        }
    )

    result = add_normalized_coordinates(columns)

    assert result["xn"].min() == 0.0
    assert result["xn"].max() == 1.0
    assert result["yn"].min() == 0.0
    assert result["yn"].max() == 1.0