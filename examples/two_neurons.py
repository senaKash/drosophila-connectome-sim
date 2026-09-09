from flysim.simulation.lif import LIFNeuron
from flysim.simulation.synapse import Synapse


neuron_a = LIFNeuron()
neuron_b = LIFNeuron()

synapse = Synapse(weight=2.0)


for t in range(100):
    # На A постоянно подаём ток, чтобы он периодически спайкал
    spike_a = neuron_a.step(
        input_current=2.0,
        dt=1.0,
    )

    # Если A спайкнул синапс передаст ток в B
    input_b = synapse.transmit(spike_a)

    spike_b = neuron_b.step(
        input_current=input_b,
        dt=1.0,
    )

    print(
        f"time={t:3d} ms | "
        f"A={neuron_a.voltage:7.2f} mV spike={spike_a} | "
        f"B={neuron_b.voltage:7.2f} mV spike={spike_b}"
    )