import numpy as np


class SparseConnectivity:
    def __init__(
        self,
        size: int,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        weights: np.ndarray,
    ):
        self.size = size

        self.source_indices = np.asarray(source_indices, dtype=int)
        self.target_indices = np.asarray(target_indices, dtype=int)
        self.weights = np.asarray(weights, dtype=float)

        if not (
            len(self.source_indices)
            == len(self.target_indices)
            == len(self.weights)
        ):
            raise ValueError(
                "source_indices, target_indices and weights "
                "must have the same length"
            )

    def propagate(self, spikes: np.ndarray) -> np.ndarray:
        spikes = np.asarray(spikes, dtype=bool)

        if spikes.shape != (self.size,):
            raise ValueError(
                f"spikes must have shape ({self.size},)"
            )

        # Какие связи сейчас активны?
        active_connections = spikes[self.source_indices]

        # Веса только активных связей
        active_weights = self.weights * active_connections

        # Будущие входы всех нейронов
        inputs = np.zeros(self.size, dtype=float)

        # Складываем веса в соответствующие target-нейроны
        np.add.at(
            inputs,
            self.target_indices,
            active_weights,
        )

        return inputs