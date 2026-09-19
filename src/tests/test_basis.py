import pytest
import torch

from FuncyFlows.base_measures.bases import CosineBasis, FourierBasis

BASES = [CosineBasis, FourierBasis]


def gram(basis, num_points):
    points, weights = basis.quadrature(num_points)
    basis_matrix = basis.evaluate(points)
    return basis_matrix.T @ (weights[:, None] * basis_matrix)


@pytest.mark.parametrize("basis_class", BASES)
@pytest.mark.parametrize("physical_dim, num_points", [(1, 801), (2, 301)])
def test_orthonormal(basis_class, physical_dim, num_points):
    torch.testing.assert_close(gram(basis_class(20, physical_dim), num_points),
                               torch.eye(20, dtype=torch.float64), atol=1e-8, rtol=0)


@pytest.mark.parametrize("basis_class", BASES)
def test_coarse_quadrature_fails(basis_class):
    assert (gram(basis_class(20, 1), 15) - torch.eye(20, dtype=torch.float64)).abs().max() > 1e-3


@pytest.mark.parametrize("basis_class", BASES)
def test_round_trip(basis_class):
    basis = basis_class(20, 1)
    points, weights = basis.quadrature(801)
    basis_matrix = basis.evaluate(points)
    coeffs = torch.randn(3, 20, dtype=torch.float64)
    torch.testing.assert_close(((coeffs @ basis_matrix.T) * weights) @ basis_matrix, coeffs, atol=1e-8, rtol=0)


@pytest.mark.parametrize("basis_class", BASES)
def test_laplacian_eigenfunctions_1d(basis_class):
    basis = basis_class(6, 1)
    points, _ = basis.quadrature(2001)
    step = points[1, 0] - points[0, 0]
    values = basis.evaluate(points)
    second_derivative = (values[2:] - 2 * values[1:-1] + values[:-2]) / step ** 2
    torch.testing.assert_close(second_derivative, -basis.laplacian_eigenvalues * values[1:-1], atol=1e-2, rtol=1e-3)