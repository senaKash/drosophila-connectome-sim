from flysim.connectome.loader import load_malecns_feather


WEIGHTS_PATH = (
    "data/raw/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


print("Loading MaleCNS...")

neuron_index, connectivity = load_malecns_feather(
    weights_path=WEIGHTS_PATH,
    annotations_path=ANNOTATIONS_PATH,
    min_weight=5,
)

print()
print("Loaded.")
print("Neuron population:", len(neuron_index))
print("Connections:", len(connectivity.weights))

print()
print("First 5 connections:")

for i in range(5):
    source_index = int(connectivity.source_indices[i])
    target_index = int(connectivity.target_indices[i])

    source_body_id = neuron_index.to_body_id(source_index)
    target_body_id = neuron_index.to_body_id(target_index)

    weight = connectivity.weights[i]

    print(
        f"{source_body_id} -> "
        f"{target_body_id}, "
        f"weight={weight}"
    )