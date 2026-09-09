import pandas as pd


def build_complete_retina(
    full_columns: pd.DataFrame,
    observed_receptors: pd.DataFrame,
    virtual_receptors_per_column: int = 6,
) -> pd.DataFrame:
    """
    Строит полный retinal input layer.

    full_columns:
        Все visual columns глаза.
        Ожидаются:
            hex1
            hex2

    observed_receptors:
        Реальные R1-R6, для которых retinotopic position
        была восстановлена через MaleCNS.

        Ожидаются:
            r_bodyId
            hex1
            hex2

    Для колонок без реальных R1-R6 создаются
    virtual receptors.

    Реальные MaleCNS bodyId не подменяются.
    У виртуальных рецепторов отдельные строковые ID.
    """
    if virtual_receptors_per_column <= 0:
        raise ValueError(
            "virtual_receptors_per_column must be positive"
        )

    required_full = {
        "hex1",
        "hex2",
    }

    required_observed = {
        "r_bodyId",
        "hex1",
        "hex2",
    }

    if not required_full.issubset(full_columns.columns):
        raise ValueError(
            "full_columns must contain hex1 and hex2"
        )

    if not required_observed.issubset(
        observed_receptors.columns
    ):
        raise ValueError(
            "observed_receptors must contain "
            "r_bodyId, hex1 and hex2"
        )

    rows = []

    # --------------------------------------------------
    # Быстрый lookup:
    #
    # (hex1, hex2) -> реальные R1-R6 bodyId
    # --------------------------------------------------

    observed_by_column = {}

    for (hex1, hex2), group in observed_receptors.groupby(
        ["hex1", "hex2"]
    ):
        observed_by_column[
            (int(hex1), int(hex2))
        ] = group["r_bodyId"].astype(int).tolist()

    # --------------------------------------------------
    # Проходим по всем visual columns
    # --------------------------------------------------

    for row in full_columns.itertuples(index=False):
        hex1 = int(row.hex1)
        hex2 = int(row.hex2)

        key = (hex1, hex2)

        real_body_ids = observed_by_column.get(
            key,
            [],
        )

        # ----------------------------------------------
        # Если реальные R1-R6 есть — используем их.
        # ----------------------------------------------

        if real_body_ids:
            for body_id in real_body_ids:
                rows.append(
                    {
                        "hex1": hex1,
                        "hex2": hex2,
                        "receptor_id": str(body_id),
                        "real_body_id": body_id,
                        "is_virtual": False,
                    }
                )

        # ----------------------------------------------
        # Если данных нет — создаём synthetic input.
        # ----------------------------------------------

        else:
            for receptor_number in range(
                virtual_receptors_per_column
            ):
                virtual_id = (
                    f"VR1-R6_R_"
                    f"{hex1}_"
                    f"{hex2}_"
                    f"{receptor_number}"
                )

                rows.append(
                    {
                        "hex1": hex1,
                        "hex2": hex2,
                        "receptor_id": virtual_id,
                        "real_body_id": None,
                        "is_virtual": True,
                    }
                )

    return pd.DataFrame(rows)

def assign_column_brightness(
    retina: pd.DataFrame,
    column_brightness: pd.DataFrame,
) -> pd.DataFrame:
    """
    Назначает яркость visual column всем retinal receptors
    внутри этой колонки.

    retina ожидает:
        hex1
        hex2
        receptor_id
        real_body_id
        is_virtual

    column_brightness ожидает:
        hex1
        hex2
        brightness

    brightness:
        0.0 = чёрный
        1.0 = белый

    Например:

        hex (10, 20)
        brightness = 0.8

    Все реальные или virtual R1-R6 в этой колонке
    получат brightness = 0.8.
    """

    required_retina = {
        "hex1",
        "hex2",
        "receptor_id",
        "is_virtual",
    }

    required_brightness = {
        "hex1",
        "hex2",
        "brightness",
    }

    if not required_retina.issubset(retina.columns):
        raise ValueError(
            "retina must contain "
            "hex1, hex2, receptor_id and is_virtual"
        )

    if not required_brightness.issubset(
        column_brightness.columns
    ):
        raise ValueError(
            "column_brightness must contain "
            "hex1, hex2 and brightness"
        )

    # Для одной visual column должно быть
    # ровно одно значение яркости.
    if column_brightness.duplicated(
        subset=["hex1", "hex2"]
    ).any():
        raise ValueError(
            "column_brightness contains duplicate hex positions"
        )

    brightness_values = (
        column_brightness["brightness"]
        .to_numpy(dtype=float)
    )

    # Пока stimulus задаём в нормализованном диапазоне [0, 1].
    if (
        (brightness_values < 0.0).any()
        or (brightness_values > 1.0).any()
    ):
        raise ValueError(
            "brightness must be between 0 and 1"
        )

    stimulated = retina.merge(
        column_brightness[
            [
                "hex1",
                "hex2",
                "brightness",
            ]
        ],
        on=[
            "hex1",
            "hex2",
        ],
        how="left",

        # Многие receptors принадлежат одной visual column.
        validate="many_to_one",
    )

    # Полный retinal layer должен иметь stimulus
    # для каждой из 892 visual columns.
    if stimulated["brightness"].isna().any():
        missing = (
            stimulated[
                stimulated["brightness"].isna()
            ][
                ["hex1", "hex2"]
            ]
            .drop_duplicates()
        )

        raise ValueError(
            "Missing brightness for retinal columns: "
            f"{missing.to_dict(orient='records')}"
        )

    return stimulated