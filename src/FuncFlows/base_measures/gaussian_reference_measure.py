import torch

from .abstract_reference_measure import ReferenceMeasure


class GaussianReferenceMeasure(ReferenceMeasure):
    """N(0, C0) with C0 = (I - alpha Laplacian)^(-power), diagonal in the basis:
    variance_k = (1 + alpha * sigma_k)^(-power), sigma_k = basis.laplacian_eigenvalues[k].

    The measure lives wherever its variances live: pass variances already on a device (or call
    .move_to(device)) and sample() draws there. Previously sample() always drew on the CPU in the
    default dtype, which every GPU/MPS run then had to work around with a subclass.
    """

    def __init__(self, basis, alpha=0.1, power=2.0, variances=None, dtype=torch.float64):
        if variances is None:
            variances = ((1 + alpha * basis.laplacian_eigenvalues) ** (-power)).to(dtype)
        elif not torch.is_tensor(variances):
            variances = torch.as_tensor(variances, dtype=dtype)
        super().__init__(basis, variances.dtype)                    # dtype follows the variances
        self.variances, self.scale = variances, variances.sqrt()

    def move_to(self, device):
        self.variances, self.scale = self.variances.to(device), self.scale.to(device)
        return self

    def sample(self, num_samples):
        return self.scale * torch.randn(num_samples, self.num_functions,
                                        dtype=self.scale.dtype, device=self.scale.device)

    def log_density_diff(self, coeffs_from, coeffs_to):
        # only this simple because the modes are independent
        return 0.5 * ((coeffs_to ** 2 - coeffs_from ** 2) / self.variances).sum(-1)

    def cameron_martin_norm2(self, coeffs):
        """Squared Cameron-Martin norm: the L2 norm after whitening by the covariance."""
        return (coeffs ** 2 / self.variances).sum(-1)
