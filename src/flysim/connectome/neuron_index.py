class NeuronIndex:
    def __init__(self, body_ids):
        body_ids = [int(body_id) for body_id in body_ids]

        if len(body_ids) != len(set(body_ids)):
            raise ValueError("body_ids must be unique")

        # index -> bodyId
        self.body_ids = tuple(body_ids)

        # bodyId -> index
        self.id_to_index = {
            body_id: index
            for index, body_id in enumerate(body_ids)
        }

    def __len__(self) -> int:
        return len(self.body_ids)

    def to_index(self, body_id: int) -> int:
        try:
            return self.id_to_index[int(body_id)]
        except KeyError:
            raise KeyError(f"Unknown bodyId: {body_id}")

    def to_body_id(self, index: int) -> int:
        if index < 0 or index >= len(self.body_ids):
            raise IndexError(f"Neuron index out of range: {index}")

        return self.body_ids[index]