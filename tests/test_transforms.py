from density import transforms
import numpy as np
import pytest
import torch as th
import torch.autograd.functional


DIMS = 5


@pytest.fixture(params=[(), (13,), (7, 19)])
def batch_shape(request) -> tuple:
    return request.param


@pytest.fixture(params=[transforms.PermutationTransform, transforms.PlanarTransform])
def transform(request) -> th.distributions.Transform:
    if request.param is transforms.PermutationTransform:
        index = th.randperm(DIMS)
        return transforms.PermutationTransform(index)
    if request.param is transforms.PlanarTransform:
        bias = th.randn(())
        weight = th.randn(DIMS)
        scale = th.randn(DIMS)
        return transforms.PlanarTransform(weight, bias, scale, th.tanh)
    else:
        raise NotImplementedError(request.param)


@pytest.fixture
def domain_tensor(transform: th.distributions.Transform, batch_shape: tuple) -> th.Tensor:
    x = th.randn(*batch_shape, DIMS)
    return th.distributions.transform_to(transform.domain)(x)


@pytest.fixture
def codomain_tensor(transform: th.distributions.Transform, domain_tensor: th.Tensor) -> th.Tensor:
    return transform(domain_tensor)


@pytest.fixture
def jacobian(transform: th.distributions.Transform, domain_tensor: th.Tensor, batch_shape: tuple) \
        -> th.Tensor:
    # Use the batch sum trick to evaluate the batched jacobian (https://bit.ly/3rOMq6F). However,
    # the jacobian will have shape (output_dim, *, input_dim), so we need to move the output
    # dimension to the second to last position.
    batch_dims = tuple(range(len(batch_shape)))
    jacobian: th.Tensor = torch.autograd.functional.jacobian(
        lambda x: transform(x).sum(batch_dims) if batch_dims else transform(x), domain_tensor,
    )
    return jacobian.moveaxis(0, -2)


def test_log_abs_det_jacobian(transform: th.distributions.Transform, batch_shape: tuple,
                              domain_tensor: th.Tensor, codomain_tensor: th.Tensor,
                              jacobian: th.Tensor):
    log_abs_det_jacobian: th.Tensor = transform.log_abs_det_jacobian(domain_tensor, codomain_tensor)
    assert log_abs_det_jacobian.shape == batch_shape
    # Validate against the autograd Jacobian.
    _, autograd_log_abs_det_jacobian = jacobian.slogdet()
    np.testing.assert_allclose(log_abs_det_jacobian, autograd_log_abs_det_jacobian, atol=1e-9)


def test_bijective_transform(transform: th.distributions.Transform, domain_tensor: th.Tensor,
                             codomain_tensor: th.Tensor):
    if not transform.bijective:
        pytest.skip(f"{transform} is not bijective")
    inv = transform.inv(codomain_tensor)
    np.testing.assert_allclose(inv, domain_tensor)


def test_dimension_preserving_transform(domain_tensor: th.Tensor, codomain_tensor: th.Tensor):
    assert domain_tensor.shape == codomain_tensor.shape
