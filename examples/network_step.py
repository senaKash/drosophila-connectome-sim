from flysim.simulation.lif import LIFNeuron
from flysim.simulation.network import Network


network = Network()

network.add_neuron("A", LIFNeuron())
network.add_neuron("B", LIFNeuron())

network.connect("A", "B", weight=2.0)


for t in range(35):
    # Что B собирается получить НА ЭТОМ шаге
    input_b = network.pending_inputs["B"]

    spikes = network.step(
        external_inputs={
            "A": 2.0,
        },
        dt=1.0,
    )

    print(
        f"time={t:2d} ms | "
        f"A={network.neurons['A'].voltage:6.2f} "
        f"spike={spikes['A']} | "
        f"B_input={input_b:3.1f} nA | "
        f"B={network.neurons['B'].voltage:6.2f} "
        f"spike={spikes['B']} | "
        f"next_B={network.pending_inputs['B']:3.1f} nA"
    )