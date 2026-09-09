import numpy as np

from flysim.simulation.population import LIFPopulation


population = LIFPopulation(size=3)


for t in range(35):

    inputs = np.array([
        2.0,  # neuron 0
        1.0,  # neuron 1
        0.0,  # neuron 2
    ])

    spikes = population.step(
        input_current=inputs,
        dt=1.0,
    )

    print(
        f"time={t:2d} ms | "
        f"V={population.voltage.round(2)} | "
        f"spikes={spikes}"
    )