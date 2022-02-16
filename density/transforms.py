import torch as th
from torch import distributions
from torch.distributions import constraints
import typing
from .util import ACTIVATION_GRADIENT_REGISTER


class PermutationTransform(distributions.Transform):
    """
    Permute elements of a tensor along a given axis.

    Args:
        index:
        dim:
    """
    bijective = True
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, index: th.LongTensor, dim: int = -1, cache_size=0):
        self.index = index
        self.inverse_index = th.empty_like(self.index)
        self.inverse_index[self.index] = th.arange(self.index.numel())
        self.dim = dim
        super().__init__(cache_size)

    def _call(self, x: th.Tensor) -> th.Tensor:
        return th.index_select(x, self.dim, self.index)

    def _inverse(self, y: th.Tensor) -> th.Tensor:
        return th.index_select(y, self.dim, self.inverse_index)

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        if x.shape != y.shape:  # pragma: no cover
            raise ValueError("x and y must have the same shape.")
        dim = (x.dim() + self.dim) if self.dim < 0 else self.dim
        if dim < 0 or dim >= x.dim():  # pragma: no cover
            raise ValueError(f"{self.dim} is not a valid dimension for tensor with shape {x.shape}")
        return th.zeros([n for i, n in enumerate(x.shape) if i != dim], dtype=x.dtype)


class PlanarTransform(distributions.Transform):
    r"""
    Planar transformation.

    Args:
        weight:
        bias:
        scale:
        activation:
    """
    bijective = False
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, weight: th.Tensor, bias: th.Tensor, scale: th.Tensor,
                 activation: typing.Callable, activation_grad: typing.Callable = None,
                 cache_size: int = 0):
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.activation = activation
        self.activation_grad = activation_grad or ACTIVATION_GRADIENT_REGISTER.get(self.activation)
        if self.activation_grad is None:  # pragma: no cover
            raise RuntimeError(f"could not automatically determine gradient for {self.activation}")
        super().__init__(cache_size)

    def _linear(self, x: th.Tensor) -> th.Tensor:
        return (x @ self.weight + self.bias)[..., None]

    def _call(self, x: th.Tensor) -> th.Tensor:
        return x + self.scale * self.activation(self._linear(x))

    def log_abs_det_jacobian(self, x: th.Tensor, _) -> th.Tensor:
        activation_grad = self.activation_grad(self._linear(x)) * self.weight
        return th.log((1 + activation_grad @ self.scale).abs())  # TODO: numerical stability.


class FlowModule(th.nn.Module):
    def forward(self, *args, **kwargs) -> th.distributions.transforms.Transform:
        raise NotImplementedError
