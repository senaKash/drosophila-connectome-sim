import pyarrow as pa
import pyarrow.ipc as ipc


PATH = (
    "data/raw/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)


with pa.memory_map(PATH, "r") as source:
    reader = ipc.open_file(source)

    print("Schema:")
    print(reader.schema)

    print("\nNumber of record batches:")
    print(reader.num_record_batches)

    print("\nFirst 5 rows:")
    first_batch = reader.get_batch(0)
    print(first_batch.slice(0, 5).to_pandas())