import torch as th
from torch.utils.data import IterableDataset
import typing


ACTIVATION_GRADIENT_REGISTER = {
    th.tanh: lambda x: x.cosh() ** - 2,
}


class SimulatedDataset(IterableDataset):
    """
    A dataset that yields synthetic data from a simulator.

    Args:
        simulator: Callable to generate data.
        args: Positional arguments for the simulator.
        kwargs: Keyword arguments for the simulator.
        length: Maximum number of simulations.
    """
    def __init__(self, simulator: typing.Callable, args: typing.Iterable = None,
                 kwargs: typing.Mapping = None, length: int = None):
        super().__init__()
        self.simulator = simulator
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.length = length
        self.num_simulations = 0

    def __len__(self):
        if self.length is None:
            raise TypeError('length of simulated dataset has not been specified')
        return self.length

    def __iter__(self):
        while self.length is None or (self.num_simulations < self.length):
            simulation = self.simulator(*self.args, **self.kwargs)
            self.num_simulations += 1
            yield simulation
