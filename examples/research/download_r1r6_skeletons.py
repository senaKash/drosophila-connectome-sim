from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


ANNOTATIONS = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
OUTPUT = Path("data/raw/skeletons_r1r6_right")

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "segmentation/skeletons-malecns/skeletons-swc"
)


df = pd.read_feather(
    ANNOTATIONS,
    columns=["bodyId", "type", "instance"],
)

body_ids = df[
    (df["type"] == "R1-R6")
    & (df["instance"] == "R1-R6_R")
]["bodyId"].astype(int)

OUTPUT.mkdir(parents=True, exist_ok=True)

for i, body_id in enumerate(body_ids, start=1):
    path = OUTPUT / f"{body_id}.swc"

    if not path.exists():
        try:
            urlretrieve(
                f"{BASE_URL}/{body_id}.swc",
                path,
            )
        except Exception:
            print("failed:", body_id)

    if i % 100 == 0:
        print(f"{i}/{len(body_ids)}")