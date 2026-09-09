import pandas as pd

from flysim.vision.retina import (
    assign_column_brightness,
    build_complete_retina,
)

def test_build_complete_retina():
    full_columns = pd.DataFrame(
        {
            "hex1": [
                10,
                11,
            ],
            "hex2": [
                20,
                20,
            ],
        }
    )

    # Для первой колонки есть два настоящих R1-R6.
    # Для второй данных нет.
    observed_receptors = pd.DataFrame(
        {
            "r_bodyId": [
                1001,
                1002,
            ],
            "hex1": [
                10,
                10,
            ],
            "hex2": [
                20,
                20,
            ],
        }
    )

    retina = build_complete_retina(
        full_columns=full_columns,
        observed_receptors=observed_receptors,
        virtual_receptors_per_column=6,
    )

    real = retina[
        ~retina["is_virtual"]
    ]

    virtual = retina[
        retina["is_virtual"]
    ]

    # Два настоящих R1-R6.
    assert len(real) == 2

    assert real["real_body_id"].tolist() == [
        1001,
        1002,
    ]

    # В отсутствующей колонке создали 6 virtual R1-R6.
    assert len(virtual) == 6

    assert set(
        virtual["hex1"]
    ) == {11}

    assert set(
        virtual["hex2"]
    ) == {20}

    # В итоге обе visual columns представлены.
    assert (
        retina[
            ["hex1", "hex2"]
        ]
        .drop_duplicates()
        .shape[0]
        == 2
    )


def test_assign_column_brightness():
    full_columns = pd.DataFrame(
        {
            "hex1": [10, 11],
            "hex2": [20, 20],
        }
    )

    observed_receptors = pd.DataFrame(
        {
            "r_bodyId": [
                1001,
                1002,
            ],
            "hex1": [
                10,
                10,
            ],
            "hex2": [
                20,
                20,
            ],
        }
    )

    retina = build_complete_retina(
        full_columns=full_columns,
        observed_receptors=observed_receptors,
        virtual_receptors_per_column=6,
    )

    column_brightness = pd.DataFrame(
        {
            "hex1": [10, 11],
            "hex2": [20, 20],
            "brightness": [
                1.0,
                0.25,
            ],
        }
    )

    stimulated = assign_column_brightness(
        retina=retina,
        column_brightness=column_brightness,
    )

    real = stimulated[
        ~stimulated["is_virtual"]
    ]

    virtual = stimulated[
        stimulated["is_virtual"]
    ]

    # Оба реальных рецептора находятся в первой колонке.
    assert real["brightness"].tolist() == [
        1.0,
        1.0,
    ]

    # Все 6 virtual receptors второй колонки
    # получают одну и ту же яркость.
    assert virtual["brightness"].tolist() == [
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    ]