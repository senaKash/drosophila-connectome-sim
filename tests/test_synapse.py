from flysim.simulation.lif import LIFNeuron
from flysim.simulation.synapse import Synapse


def test_synapse_transmits_weight_on_spike():
    synapse = Synapse(weight=2.0)

    assert synapse.transmit(True) == 2.0
    assert synapse.transmit(False) == 0.0


def test_spike_from_one_neuron_affects_another():
    neuron_a = LIFNeuron()
    neuron_b = LIFNeuron()
    synapse = Synapse(weight=2.0)

    initial_voltage_b = neuron_b.voltage

    # Доводим A до первого spike
    while True:
        spike_a = neuron_a.step(input_current=2.0)

        if spike_a:
            break

    input_b = synapse.transmit(spike_a)
    neuron_b.step(input_current=input_b)

    assert neuron_b.voltage > initial_voltage_b