import pandas as pd


PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

df = pd.read_feather(PATH)

print("Rows:", len(df))
print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nFirst 5 rows:")
print(df.head())