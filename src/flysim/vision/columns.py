import math

import pandas as pd


def build_right_eye_columns(
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Получает visual columns правого глаза из L1.
    """
    l1 = annotations[
        (annotations["type"] == "L1")
        & (annotations["somaSide"] == "R")
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ]

    columns = l1[
        [
            "assignedOlHex1",
            "assignedOlHex2",
        ]
    ].rename(
        columns={
            "assignedOlHex1": "hex1",
            "assignedOlHex2": "hex2",
        }
    )

    columns = columns.drop_duplicates().copy()

    columns["hex1"] = columns["hex1"].astype(int)
    columns["hex2"] = columns["hex2"].astype(int)

    return columns.reset_index(drop=True)


def add_normalized_coordinates(
    columns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет координаты xn, yn в диапазоне 0..1
    для выборки пикселей изображения.
    """
    result = columns.copy()

    x = result["hex1"] + 0.5 * (result["hex2"] % 2)
    y = result["hex2"] * (math.sqrt(3) / 2)

    result["xn"] = (x - x.min()) / (x.max() - x.min())
    result["yn"] = (y - y.min()) / (y.max() - y.min())

    return result