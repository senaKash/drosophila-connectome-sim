from flysim.simulation.lif import LIFNeuron
from flysim.simulation.synapse import Synapse


class Network:
    def __init__(self):
        # Имя нейрона -> объект нейрона
        self.neurons: dict[str, LIFNeuron] = {}

        # Входы, которые должны прийти нейронам на следующем шаге
        self.pending_inputs: dict[str, float] = {}

        # (отправитель, получатель, синапс)
        self.connections: list[tuple[str, str, Synapse]] = []

    def add_neuron(self, name: str, neuron: LIFNeuron) -> None:
        if name in self.neurons:
            raise ValueError(f"Neuron '{name}' already exists")

        self.neurons[name] = neuron
        self.pending_inputs[name] = 0.0

    def connect(self, source: str, target: str, weight: float) -> None:
        if source not in self.neurons:
            raise ValueError(f"Unknown source neuron: '{source}'")

        if target not in self.neurons:
            raise ValueError(f"Unknown target neuron: '{target}'")

        synapse = Synapse(weight=weight)

        self.connections.append(
            (source, target, synapse)
        )

    def step(
        self,
        external_inputs: dict[str, float] | None = None,
        dt: float = 1.0,
    ) -> dict[str, bool]:

        if external_inputs is None:
            external_inputs = {}

        spikes: dict[str, bool] = {}

        # 1. Обновляем все нейроны текущими входами
        for name, neuron in self.neurons.items():
            total_input = (
                self.pending_inputs[name]
                + external_inputs.get(name, 0.0)
            )

            spikes[name] = neuron.step(
                input_current=total_input,
                dt=dt,
            )

        # 2. Готовим входы на следующий шаг
        next_inputs = {
            name: 0.0
            for name in self.neurons
        }

        for source, target, synapse in self.connections:
            next_inputs[target] += synapse.transmit(
                spikes[source]
            )

        self.pending_inputs = next_inputs

        return spikes