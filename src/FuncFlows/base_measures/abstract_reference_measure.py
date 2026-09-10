import torch


class ReferenceMeasure:
    """Probability measure on coefficient vectors [..., num_functions] in the given basis."""

    def __init__(self, basis, dtype=torch.float64):
        self.basis = basis
        self.num_functions = basis.num_functions
        self.dtype = dtype



    def sample(self, num_samples):
        """-> coeffs [num_samples, num_functions]"""
        raise NotImplementedError



    def log_density_diff(self, coeffs_from, coeffs_to):
        """-> log p(coeffs_from) - log p(coeffs_to), shape [...]. Thing transport needs."""
        raise NotImplementedError



    def in_support(self, coeffs):
        # always returns true as Gaussian/Student-t have full support. Have to adjust for bounded domains
        return torch.ones(coeffs.shape[:-1], dtype=torch.bool) 



    def evaluate(self, coeffs, points):
        """coeffs [..., num_functions] -> function values [..., num_points]"""
        return coeffs @ self.basis.evaluate(points).T