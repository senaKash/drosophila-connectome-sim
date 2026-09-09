import numpy as np


class LIFPopulation:
    def __init__(
        self,
        size: int,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        tau_m: float = 20.0,
        resistance: float = 10.0,
    ):
        if size <= 0:
            raise ValueError("size must be greater than 0")

        if tau_m <= 0:
            raise ValueError("tau_m must be greater than 0")

        self.size = size

        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.tau_m = tau_m
        self.resistance = resistance

        # Один массив хранит voltage всех нейронов.
        self.voltage = np.full(
            size,
            v_rest,
            dtype=float,
        )

    def step(
        self,
        input_current: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:

        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        input_current = np.asarray(
            input_current,
            dtype=float,
        )

        if input_current.shape != (self.size,):
            raise ValueError(
                f"input_current must have shape ({self.size},)"
            )

        # Та же самая LIF-математика, но сразу для всего массива.
        leak = -(self.voltage - self.v_rest)

        input_effect = (
            self.resistance * input_current
        )

        dv_dt = (
            leak + input_effect
        ) / self.tau_m

        self.voltage += dv_dt * dt

        # Получаем массив True/False для всех нейронов.
        spikes = self.voltage >= self.v_threshold

        # Сбрасываем только те нейроны, которые спайкнули.
        self.voltage[spikes] = self.v_reset

        return spikes