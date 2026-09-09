import pandas as pd


PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


df = pd.read_feather(PATH)

print("ANNOTATION COLUMNS")
print("------------------")

for column in df.columns:
    print(column)

print()
print("GEOMETRY-LIKE COLUMNS")
print("---------------------")

for column in ["somaLocation", "tosomaLocation"]:
    print()
    print(column)
    print("dtype:", df[column].dtype)
    print("non-null:", df[column].notna().sum())

    print(
        df[column]
        .dropna()
        .head(10)
        .to_string(index=False)
    )

r1r6 = df[
    (df["type"] == "R1-R6")
    & (df["instance"] == "R1-R6_R")
    & df["somaLocation"].notna()
].copy()

print()
print("RIGHT-EYE R1-R6 SOMA GEOMETRY")
print("----------------------------")
print("Neurons:", len(r1r6))

print()
print(
    r1r6[
        [
            "bodyId",
            "instance",
            "somaLocation",
        ]
    ]
    .head(10)
    .to_string(index=False)
)