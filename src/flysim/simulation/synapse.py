class Synapse:
    def __init__(self, weight: float):
        self.weight = weight

    def transmit(self, spike: bool) -> float:
        if spike:
            return self.weight

        return 0.0