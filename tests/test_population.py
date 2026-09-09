import numpy as np

from flysim.simulation.population import LIFPopulation


def test_population_starts_at_rest():
    population = LIFPopulation(size=3)

    assert np.all(population.voltage == -65.0)


def test_population_spike_and_reset():
    population = LIFPopulation(size=3)

    inputs = np.array([
        2.0,
        1.0,
        0.0,
    ])

    spike_happened = False

    for _ in range(100):
        spikes = population.step(inputs)

        if spikes[0]:
            spike_happened = True
            break

    assert spike_happened is True
    assert population.voltage[0] == population.v_reset


def test_population_handles_different_neuron_inputs():
    population = LIFPopulation(size=3)

    inputs = np.array([
        2.0,
        1.0,
        0.0,
    ])

    for _ in range(10):
        population.step(inputs)

    assert population.voltage[0] > population.voltage[1]
    assert population.voltage[1] > population.voltage[2]
    assert population.voltage[2] == population.v_rest


def test_population_rejects_wrong_input_shape():
    population = LIFPopulation(size=3)

    wrong_input = np.array([
        1.0,
        2.0,
    ])

    try:
        population.step(wrong_input)

        assert False, "Expected ValueError"

    except ValueError:
        pass