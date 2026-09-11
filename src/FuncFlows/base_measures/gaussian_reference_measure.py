from .abstract_reference_measure import ReferenceMeasure
import torch


class GaussianReferenceMeasure(ReferenceMeasure):
    """N(0, C0) with C0 = (I - alpha Laplacian)^(-power), diagonal in the basis:
    variance_k = (1 + alpha * sigma_k)^(-power), sigma_k = basis.laplacian_eigenvalues[k]."""

    def __init__(self, basis, alpha=0.1, power=2.0, variances=None, dtype=torch.float64):
        super().__init__(basis, dtype)
        self.variances = (1 + alpha * basis.laplacian_eigenvalues) ** (-power) if variances is None else variances
        self.scale = self.variances.sqrt()

    def sample(self, num_samples):
        return self.scale * torch.randn(num_samples, self.num_functions, dtype=self.dtype)

    def log_density_diff(self, coeffs_from, coeffs_to):
        # only this simple because we assume the dimensions are independent
        return 0.5 * ((coeffs_to ** 2 - coeffs_from ** 2) / self.variances).sum(-1)


    # L^2 norm after whitening/weighting by the covariances
    def cameron_martin_norm2(self, coeffs):
        return (coeffs ** 2 / self.variances).sum(-1)