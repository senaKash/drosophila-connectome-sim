import numpy as np
import pandas as pd

from flysim.vision.image_input import sample_image_brightness
from flysim.vision.retina import (
    assign_column_brightness,
    build_complete_retina,
)


def test_image_brightness_reaches_retinal_receptors():
    image = np.array(
        [
            [0.0, 1.0],
            [0.25, 0.75],
        ]
    )

    columns = pd.DataFrame(
        {
            "hex1": [10, 11],
            "hex2": [20, 20],
            "xn": [0.0, 1.0],
            "yn": [0.0, 1.0],
        }
    )

    observed_receptors = pd.DataFrame(
        {
            "r_bodyId": [1001, 1002],
            "hex1": [10, 10],
            "hex2": [20, 20],
        }
    )

    retina = build_complete_retina(
        full_columns=columns[["hex1", "hex2"]],
        observed_receptors=observed_receptors,
        virtual_receptors_per_column=6,
    )

    column_brightness = sample_image_brightness(
        image=image,
        columns=columns,
    )

    stimulated = assign_column_brightness(
        retina=retina,
        column_brightness=column_brightness,
    )

    real = stimulated[
        ~stimulated["is_virtual"]
    ]

    virtual = stimulated[
        stimulated["is_virtual"]
    ]

    assert len(real) == 2
    assert len(virtual) == 6

    np.testing.assert_allclose(
        real["brightness"],
        0.0,
    )

    np.testing.assert_allclose(
        virtual["brightness"],
        0.75,
    )