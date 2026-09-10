import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flysim.connectome.loader import build_soma_mapping
from flysim.connectome.neuron_index import build_malecns_neuron_index


ANNOTATIONS = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

print("Loading MaleCNS network index...")
df = pd.read_feather(
    ANNOTATIONS,
    columns=[
        "bodyId",
        "somaLocation",
        "superclass",
    ],
)

neuron_index = build_malecns_neuron_index(df)
df = df[df["somaLocation"].notna()].copy()


def parse_location(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)

    return np.asarray(
        ast.literal_eval(str(value)),
        dtype=float,
    )


# Keep the original soma cloud for unchanged camera framing.
camera_points = np.stack(
    df["somaLocation"]
    .map(parse_location)
    .to_numpy()
)

body_ids, network_indices, points = build_soma_mapping(
    annotations=df,
    neuron_index=neuron_index,
)

print("Network neurons:", len(neuron_index))
print("Brain/CNS soma points:", len(points))


# -----------------------------
# 1. Центр сцены
# -----------------------------
center = np.median(camera_points, axis=0)

# -----------------------------
# 2. Убираем влияние выбросов
# Берём "типичный" размер сцены через перцентили
# -----------------------------
p_low = np.percentile(camera_points, 5, axis=0)
p_high = np.percentile(camera_points, 95, axis=0)

robust_span = p_high - p_low
max_span = robust_span.max()

# -----------------------------
# 3. Управление зумом
# МЕНЬШЕ число -> БЛИЖЕ камера
# Попробуй 0.8, 0.6, 0.45
# -----------------------------
zoom = 0.55
half = max_span * zoom / 2


fig = plt.figure(
    figsize=(10, 8),
    facecolor="black",
)

ax = fig.add_subplot(
    111,
    projection="3d",
    facecolor="black",
)

ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    s=3,
    c="#3fa7ff",
    alpha=0.9,
)

# Ортографическая проекция — без лишней перспективы
ax.set_proj_type("ortho")

# Нормальные пропорции по осям
ax.set_box_aspect(robust_span)

# Вид спереди / сверху можно менять
# сверху:
ax.view_init(elev=90, azim=-90)

# Если хочешь "фронтальный" ракурс, попробуй:
# ax.view_init(elev=0, azim=-90)

# ЯВНЫЙ ЗУМ
ax.set_xlim(center[0] - half, center[0] + half)
ax.set_ylim(center[1] - half, center[1] + half)
ax.set_zlim(center[2] - half, center[2] + half)

ax.set_axis_off()

plt.tight_layout()
plt.show()
