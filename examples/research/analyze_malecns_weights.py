import numpy as np

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

_, connectivity = load_malecns_feather(
    weights_path=WEIGHTS_PATH,
    annotations_path=ANNOTATIONS_PATH,
    min_weight=5,
)

weights = connectivity.weights

print()
print("Connections:", len(weights))

print()
print("Weight statistics:")
print("Min:       ", np.min(weights))
print("Mean:      ", np.mean(weights))
print("Median:    ", np.median(weights))
print("75%:       ", np.percentile(weights, 75))
print("90%:       ", np.percentile(weights, 90))
print("95%:       ", np.percentile(weights, 95))
print("99%:       ", np.percentile(weights, 99))
print("99.9%:     ", np.percentile(weights, 99.9))
print("Max:       ", np.max(weights))

print()
print("Weight ranges:")

ranges = [
    (5, 9),
    (10, 19),
    (20, 49),
    (50, 99),
    (100, 199),
    (200, 499),
    (500, 999),
    (1000, None),
]

for lower, upper in ranges:
    if upper is None:
        count = np.sum(weights >= lower)
        label = f">= {lower}"
    else:
        count = np.sum(
            (weights >= lower)
            & (weights <= upper)
        )
        label = f"{lower:4d} - {upper:4d}"

    percent = count / len(weights) * 100

    print(
        f"{label}: "
        f"{count:8d} "
        f"({percent:6.3f}%)"
    )

print()
print("Neuron connectivity statistics:")

size = connectivity.size

in_degree = np.bincount(
    connectivity.target_indices,
    minlength=size,
)

out_degree = np.bincount(
    connectivity.source_indices,
    minlength=size,
)

incoming_weight = np.bincount(
    connectivity.target_indices,
    weights=weights,
    minlength=size,
)

# Нейроны без входящих/исходящих связей пока
# исключаем из статистики, чтобы нули не искажали медиану.
nonzero_in_degree = in_degree[in_degree > 0]
nonzero_out_degree = out_degree[out_degree > 0]
nonzero_incoming_weight = incoming_weight[
    incoming_weight > 0
]


def print_stats(name, values):
    print()
    print(name)
    print("Count:     ", len(values))
    print("Mean:      ", np.mean(values))
    print("Median:    ", np.median(values))
    print("90%:       ", np.percentile(values, 90))
    print("95%:       ", np.percentile(values, 95))
    print("99%:       ", np.percentile(values, 99))
    print("99.9%:     ", np.percentile(values, 99.9))
    print("Max:       ", np.max(values))


print_stats(
    "IN-DEGREE",
    nonzero_in_degree,
)

print_stats(
    "OUT-DEGREE",
    nonzero_out_degree,
)

print_stats(
    "TOTAL INCOMING WEIGHT",
    nonzero_incoming_weight,
)