from flysim.connectome.loader import load_connectivity_csv


neuron_index, connectivity = load_connectivity_csv(
    "examples/data/mini_connectome.csv"
)

print("Neuron count:", len(neuron_index))

print("Connections:")

for source, target, weight in zip(
    connectivity.source_indices,
    connectivity.target_indices,
    connectivity.weights,
):
    print(
        f"{source} -> {target}, weight={weight}"
    )