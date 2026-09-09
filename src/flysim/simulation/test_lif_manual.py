from flysim.simulation.lif import LIFNeuron


neuron = LIFNeuron()

for t in range(100):
    spike = neuron.step(
        input_current=2.0,
        dt=1.0,
    )

    print(
        f"time={t:3d} ms | "
        f"voltage={neuron.voltage:7.2f} mV | "
        f"spike={spike}"
    )