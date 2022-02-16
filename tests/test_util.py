from density import util
import pytest
from torch import distributions as dists
from torch.utils.data import DataLoader


def test_simulated_dataset():
    batch_size = 16
    num_batches = 4
    dataset = util.SimulatedDataset(dists.Normal(0, 1).sample, ([7],))
    loader = DataLoader(dataset, batch_size=batch_size)
    for i, batch in enumerate(loader):
        assert batch.shape == (batch_size, 7)
        if i > num_batches:
            break
    assert dataset.num_simulations == batch_size * (num_batches + loader.prefetch_factor)
    with pytest.raises(TypeError):
        len(dataset)


def test_simulated_dataset_with_length():
    batch_size = 16
    dataset = util.SimulatedDataset(dists.Normal(0, 1).sample, ([7],), length=37)
    loader = DataLoader(dataset, batch_size=batch_size)
    batch_sizes = [batch.shape[0] for batch in loader]
    assert batch_sizes == [16, 16, 5]
    assert len(dataset) == dataset.length
