from flysim.connectome.neuron_index import NeuronIndex


body_ids = [
    720575940123456789,
    720575940987654321,
    720575941111111111,
]

index = NeuronIndex(body_ids)


for body_id in body_ids:
    internal_index = index.to_index(body_id)

    print(
        f"bodyId={body_id} -> index={internal_index}"
    )


print()

for internal_index in range(len(index)):
    body_id = index.to_body_id(internal_index)

    print(
        f"index={internal_index} -> bodyId={body_id}"
    )