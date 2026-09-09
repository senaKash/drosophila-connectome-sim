import numpy as np

from flysim.simulation.population import LIFPopulation
from flysim.simulation.connectivity import SparseConnectivity


class VectorizedNetwork:
    def __init__(
        self,
        population: LIFPopulation,
        connectivity: SparseConnectivity,
    ):
        if population.size != connectivity.size:
            raise ValueError(
                "population and connectivity must have the same size"
            )

        self.population = population
        self.connectivity = connectivity

        # Входы, которые придут нейронам на следующем шаге
        self.pending_inputs = np.zeros(
            population.size,
            dtype=float,
        )

    def step(
        self,
        external_inputs: np.ndarray | None = None,
        dt: float = 1.0,
    ) -> np.ndarray:

        if external_inputs is None:
            external_inputs = np.zeros(
                self.population.size,
                dtype=float,
            )

        external_inputs = np.asarray(
            external_inputs,
            dtype=float,
        )

        if external_inputs.shape != (self.population.size,):
            raise ValueError(
                f"external_inputs must have shape "
                f"({self.population.size},)"
            )

        # Текущий вход = внешний вход + сигналы от прошлого шага
        total_input = (
            external_inputs
            + self.pending_inputs
        )

        # Обновляем все нейроны сразу
        spikes = self.population.step(
            input_current=total_input,
            dt=dt,
        )

        # По spike считаем входы на следующий шаг
        self.pending_inputs = (
            self.connectivity.propagate(spikes)
        )

        return spikes