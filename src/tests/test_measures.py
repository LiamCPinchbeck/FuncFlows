import pytest
import torch

from FuncFlows.base_measures.bases import CosineBasis, FourierBasis
from FuncFlows.base_measures.gaussian_reference_measure import GaussianReferenceMeasure


@pytest.fixture(params=[CosineBasis, FourierBasis])
def measure(request):
    torch.manual_seed(0)
    return GaussianReferenceMeasure(request.param(20, 1), alpha=0.1, power=2.0)


def test_variances_decay(measure):
    assert measure.variances[0] == 1.0                       # constant mode: sigma_0 = 0
    assert (measure.variances[1:] <= measure.variances[:-1]).all()


def test_sample_variance(measure):
    coeffs = measure.sample(20000)
    torch.testing.assert_close(coeffs.var(0), measure.variances, atol=0.03, rtol=0.05)


def test_cameron_martin_norm_of_samples(measure):
    coeffs = measure.sample(20000)
    torch.testing.assert_close(measure.cameron_martin_norm2(coeffs).mean(), torch.tensor(20.0, dtype=coeffs.dtype),
                               atol=0.2, rtol=0)


def test_log_density_diff_antisymmetric(measure):
    coeffs_from, coeffs_to = measure.sample(5), measure.sample(5)
    torch.testing.assert_close(measure.log_density_diff(coeffs_from, coeffs_from), torch.zeros(5, dtype=coeffs_from.dtype))
    torch.testing.assert_close(measure.log_density_diff(coeffs_from, coeffs_to),
                               -measure.log_density_diff(coeffs_to, coeffs_from))


def test_log_density_diff_matches_torch(measure):
    coeffs_from, coeffs_to = measure.sample(5), measure.sample(5)
    normal = torch.distributions.MultivariateNormal(torch.zeros(20, dtype=coeffs_from.dtype), torch.diag(measure.variances))
    torch.testing.assert_close(measure.log_density_diff(coeffs_from, coeffs_to),
                               normal.log_prob(coeffs_from) - normal.log_prob(coeffs_to))


def test_in_support(measure):
    assert measure.in_support(measure.sample(5)).all()