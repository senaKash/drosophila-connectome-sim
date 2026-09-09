import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

WEIGHTS_PATH = (
    "data/raw/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)


# --------------------------------------------------
# 1. Получаем bodyId валидных нейронов
# --------------------------------------------------

annotations = pd.read_feather(
    ANNOTATIONS_PATH,
    columns=["bodyId", "superclass"],
)

valid_annotations = annotations[
    annotations["superclass"].notna()
    & ~annotations["superclass"].str.contains(
        "tbc",
        case=False,
        na=False,
    )
]

valid_body_ids = valid_annotations["bodyId"].to_numpy(
    dtype=np.int64
)

print("Annotation rows:", len(annotations))
print("Valid annotation neurons:", len(valid_body_ids))


# Arrow будет использовать этот набор для фильтрации.
valid_body_ids_arrow = pa.array(valid_body_ids)


# --------------------------------------------------
# 2. Читаем огромный граф ПО BATCH'АМ
# --------------------------------------------------

valid_edges = 0
thresholded_edges = 0

valid_neurons = set()
thresholded_neurons = set()


with pa.memory_map(WEIGHTS_PATH, "r") as source:
    reader = ipc.open_file(source)

    print("Record batches:", reader.num_record_batches)

    for batch_index in range(reader.num_record_batches):

        batch = reader.get_batch(batch_index)

        body_pre = batch.column(
            batch.schema.get_field_index("body_pre")
        )

        body_post = batch.column(
            batch.schema.get_field_index("body_post")
        )

        weight = batch.column(
            batch.schema.get_field_index("weight")
        )

        # source должен быть валидным нейроном
        valid_pre = pc.is_in(
            body_pre,
            value_set=valid_body_ids_arrow,
        )

        # target тоже должен быть валидным нейроном
        valid_post = pc.is_in(
            body_post,
            value_set=valid_body_ids_arrow,
        )

        valid_mask = pc.and_(
            valid_pre,
            valid_post,
        )

        filtered_pre = pc.filter(
            body_pre,
            valid_mask,
        )

        filtered_post = pc.filter(
            body_post,
            valid_mask,
        )

        filtered_weight = pc.filter(
            weight,
            valid_mask,
        )

        valid_edges += len(filtered_pre)

        valid_neurons.update(
            filtered_pre.to_numpy()
        )

        valid_neurons.update(
            filtered_post.to_numpy()
        )


        # ------------------------------------------
        # Порог weight >= 5
        # ------------------------------------------

        threshold_mask = pc.greater_equal(
            filtered_weight,
            5,
        )

        threshold_pre = pc.filter(
            filtered_pre,
            threshold_mask,
        )

        threshold_post = pc.filter(
            filtered_post,
            threshold_mask,
        )

        thresholded_edges += len(threshold_pre)

        thresholded_neurons.update(
            threshold_pre.to_numpy()
        )

        thresholded_neurons.update(
            threshold_post.to_numpy()
        )


        if batch_index % 100 == 0:
            print(
                f"Processed "
                f"{batch_index}/{reader.num_record_batches}"
            )


print()
print("VALID GRAPH")
print("Edges:", valid_edges)
print("Neurons:", len(valid_neurons))

print()
print("THRESHOLDED GRAPH (weight >= 5)")
print("Edges:", thresholded_edges)
print("Neurons:", len(thresholded_neurons))