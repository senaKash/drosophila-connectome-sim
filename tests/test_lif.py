from flysim.simulation.lif import LIFNeuron


def test_neuron_starts_at_rest():
    neuron = LIFNeuron()

    assert neuron.voltage == neuron.v_rest


def test_neuron_spikes_and_resets():
    neuron = LIFNeuron()

    spike_happened = False

    for _ in range(100):
        if neuron.step(input_current=2.0):
            spike_happened = True
            break

    assert spike_happened is True
    assert neuron.voltage == neuron.v_reset


def test_neuron_does_not_spike_with_weak_input():
    neuron = LIFNeuron()

    for _ in range(100):
        spike = neuron.step(input_current=0.1)

        assert spike is False

def test_voltage_leaks_toward_rest():
    neuron = LIFNeuron()

    # Искусственно поднимаем потенциал выше состояния покоя.
    neuron.voltage = -55.0

    neuron.step(input_current=0.0)

    assert neuron.voltage < -55.0
    assert neuron.voltage > neuron.v_rest