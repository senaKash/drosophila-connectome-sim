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
        "superclass",
        "assignedOlHex1",
        "assignedOlHex2",
    ],
)


with_hex = df[
    df["assignedOlHex1"].notna()
    & df["assignedOlHex2"].notna()
]


print("All annotation rows:", len(df))
print("Rows with optic hex coordinates:", len(with_hex))

print()
print("Hex1:")
print("Min:", with_hex["assignedOlHex1"].min())
print("Max:", with_hex["assignedOlHex1"].max())
print("Unique:", with_hex["assignedOlHex1"].nunique())

print()
print("Hex2:")
print("Min:", with_hex["assignedOlHex2"].min())
print("Max:", with_hex["assignedOlHex2"].max())
print("Unique:", with_hex["assignedOlHex2"].nunique())

print()
print("First 20 neurons with coordinates:")

print(
    with_hex[
        [
            "bodyId",
            "type",
            "superclass",
            "assignedOlHex1",
            "assignedOlHex2",
        ]
    ]
    .head(20)
    .to_string(index=False)
)