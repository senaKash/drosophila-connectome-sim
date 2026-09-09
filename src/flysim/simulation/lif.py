class LIFNeuron:
    def __init__(
        self,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        tau_m: float = 20.0,
        resistance: float = 10.0,
    ):
        """
        Parameters
        ----------
        v_rest : float
            Потенциал покоя, mV.

        v_reset : float
            Потенциал после spike, mV.

        v_threshold : float
            Порог срабатывания, mV.

        tau_m : float
            Временная константа мембраны, ms.

        resistance : float
            Сопротивление мембраны, MΩ.
        """

        if tau_m <= 0:
            raise ValueError("tau_m must be greater than 0")

        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.tau_m = tau_m
        self.resistance = resistance

        # Текущее состояние нейрона.
        # При создании нейрон находится в состоянии покоя.
        self.voltage = v_rest

    def step(self, input_current: float, dt: float = 1.0) -> bool:
        """
        Выполнить один шаг симуляции.

        Parameters
        ----------
        input_current : float
            Входной ток, nA.

        dt : float
            Длительность шага симуляции, ms.

        Returns
        -------
        bool
            True, если на этом шаге произошёл spike.
        """

        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        # Утечка: тянет напряжение обратно к v_rest.
        leak = -(self.voltage - self.v_rest)

        # Влияние входного тока.
        input_effect = self.resistance * input_current

        # Скорость изменения потенциала, mV/ms.
        dv_dt = (leak + input_effect) / self.tau_m

        # Метод Эйлера:
        # V_new = V_old + dV/dt * dt
        self.voltage += dv_dt * dt

        # Проверяем порог.
        if self.voltage >= self.v_threshold:
            self.voltage = self.v_reset
            return True

        return False