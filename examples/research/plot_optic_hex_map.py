import math

import matplotlib.pyplot as plt
import pandas as pd


ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


df = pd.read_feather(
    ANNOTATIONS_PATH,
    columns=[
        "bodyId",
        "type",
        "assignedOlHex1",
        "assignedOlHex2",
    ],
)

with_hex = df[
    df["assignedOlHex1"].notna()
    & df["assignedOlHex2"].notna()
].copy()

with_hex["assignedOlHex1"] = with_hex["assignedOlHex1"].astype(int)
with_hex["assignedOlHex2"] = with_hex["assignedOlHex2"].astype(int)

# Сколько нейронов попадает в каждую hex-координату
hex_counts = (
    with_hex
    .groupby(["assignedOlHex1", "assignedOlHex2"])
    .size()
    .reset_index(name="count")
)

# Превращаем дискретные hex-координаты в x,y для картинки
# Простая "сотовая" укладка
x = []
y = []

for _, row in hex_counts.iterrows():
    h1 = row["assignedOlHex1"]
    h2 = row["assignedOlHex2"]

    px = h1 + 0.5 * (h2 % 2)
    py = h2 * (math.sqrt(3) / 2)

    x.append(px)
    y.append(py)

plt.figure(figsize=(10, 10))
plt.scatter(
    x,
    y,
    s=120,
    c=hex_counts["count"],
    marker="h",
)

plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()

plt.title("MaleCNS optic hex map (occupied columns)")
plt.xlabel("hex x")
plt.ylabel("hex y")

plt.colorbar(label="neurons in column")
plt.tight_layout()
plt.show()