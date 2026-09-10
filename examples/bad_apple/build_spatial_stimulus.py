import ast

import numpy as np
import pandas as pd
from PIL import Image

from flysim.vision.spatial_input import build_spatial_stimulus


ANNOTATIONS = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

FRAME_PATH = "data/input/test.png"

# Пока просто тестовый коэффициент.
# Это не физиологическое значение.
STIMULUS_GAIN = 1.0


def parse_location(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)

    return np.asarray(
        ast.literal_eval(str(value)),
        dtype=float,
    )


df = pd.read_feather(
    ANNOTATIONS,
    columns=["bodyId", "somaLocation"],
)

df = df[df["somaLocation"].notna()].copy()

points = np.stack(
    df["somaLocation"]
    .map(parse_location)
    .to_numpy()
)

frame = Image.open(FRAME_PATH).convert("L")
frame = np.asarray(frame, dtype=np.float32) / 255.0

input_current = build_spatial_stimulus(
    points,
    frame,
    gain=STIMULUS_GAIN,
)

print("Neurons:", len(input_current))
print("Current min:", input_current.min())
print("Current mean:", input_current.mean())
print("Current max:", input_current.max())
print(
    "Stimulated neurons:",
    np.count_nonzero(input_current > 0.05),
)
