from flysim.simulation.lif import LIFNeuron
from flysim.simulation.network import Network


def test_add_neuron():
    network = Network()

    neuron = LIFNeuron()

    network.add_neuron("A", neuron)

    assert "A" in network.neurons
    assert network.neurons["A"] is neuron


def test_connect_neurons():
    network = Network()

    network.add_neuron("A", LIFNeuron())
    network.add_neuron("B", LIFNeuron())

    network.connect("A", "B", weight=2.0)

    assert len(network.connections) == 1

    source, target, synapse = network.connections[0]

    assert source == "A"
    assert target == "B"
    assert synapse.weight == 2.0

def test_spike_creates_pending_input_for_target():
    network = Network()

    network.add_neuron("A", LIFNeuron())
    network.add_neuron("B", LIFNeuron())

    network.connect("A", "B", weight=2.0)

    # Доводим A до spike
    while True:
        spikes = network.step(
            external_inputs={"A": 2.0}
        )

        if spikes["A"]:
            break

    assert network.pending_inputs["B"] == 2.0


def test_pending_input_is_used_on_next_step():
    network = Network()

    network.add_neuron("A", LIFNeuron())
    network.add_neuron("B", LIFNeuron())

    network.connect("A", "B", weight=2.0)

    # Доводим A до spike
    while True:
        spikes = network.step(
            external_inputs={"A": 2.0}
        )

        if spikes["A"]:
            break

    voltage_before = network.neurons["B"].voltage

    # На следующем шаге B должен получить pending input
    network.step()

    assert network.neurons["B"].voltage > voltage_before