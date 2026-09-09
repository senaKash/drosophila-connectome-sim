import pandas as pd


ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


def main() -> None:
    df = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "bodyId",
            "type",
            "flywireType",
            "instance",
            "somaSide",
            "superclass",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    print("All annotations:", len(df))
    print()

    # --------------------------------------------------
    # Ищем R-типы ВО ВСЕЙ таблице,
    # а не только среди нейронов с hex coordinates.
    # --------------------------------------------------

    type_text = df["type"].fillna("").astype(str)
    flywire_text = df["flywireType"].fillna("").astype(str)
    instance_text = df["instance"].fillna("").astype(str)

    r_mask = (
        type_text.str.match(r"^R[0-9]", case=False)
        | flywire_text.str.match(r"^R[0-9]", case=False)
        | instance_text.str.match(r"^R[0-9]", case=False)
    )

    r_candidates = df[r_mask].copy()

    print("R candidates:", len(r_candidates))
    print()

    print("TYPE VALUES:")
    print(
        r_candidates["type"]
        .value_counts(dropna=False)
        .head(50)
    )
    print()

    print("FLYWIRE TYPE VALUES:")
    print(
        r_candidates["flywireType"]
        .value_counts(dropna=False)
        .head(50)
    )
    print()

    print("INSTANCE VALUES:")
    print(
        r_candidates["instance"]
        .value_counts(dropna=False)
        .head(50)
    )
    print()

    print("EXAMPLES:")
    print(
        r_candidates[
            [
                "bodyId",
                "type",
                "flywireType",
                "instance",
                "somaSide",
                "superclass",
                "assignedOlHex1",
                "assignedOlHex2",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )
    print()

    # --------------------------------------------------
    # Проверяем наличие hex coordinates
    # --------------------------------------------------

    with_hex = r_candidates[
        r_candidates["assignedOlHex1"].notna()
        & r_candidates["assignedOlHex2"].notna()
    ]

    print("R candidates with hex coordinates:", len(with_hex))

    print(
        "R candidates without hex coordinates:",
        len(r_candidates) - len(with_hex),
    )

    if len(with_hex) > 0:
        print()
        print("R WITH HEX:")
        print(
            with_hex[
                [
                    "bodyId",
                    "type",
                    "instance",
                    "somaSide",
                    "assignedOlHex1",
                    "assignedOlHex2",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()