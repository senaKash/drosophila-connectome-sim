import numpy as np
import pandas as pd

from flysim.vision.image_input import sample_image_brightness


def test_sample_image_brightness():
    image = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.2, 0.7, 0.9],
        ]
    )

    columns = pd.DataFrame(
        {
            "hex1": [10, 11, 12],
            "hex2": [20, 20, 20],
            "xn": [0.0, 0.5, 1.0],
            "yn": [0.0, 0.0, 1.0],
        }
    )

    result = sample_image_brightness(
        image=image,
        columns=columns,
    )

    np.testing.assert_allclose(
        result["brightness"],
        np.array([
            0.0,
            0.5,
            0.9,
        ]),
    )