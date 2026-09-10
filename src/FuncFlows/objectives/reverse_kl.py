class ReverseKL:
    """E_{coeffs ~ mu0}[ log_rn_weight + Phi(f(coeffs)) ]  (= ELBO?)."""

    def __init__(self, transformation, potential, num_samples=30):
        self.transformation, self.potential, self.num_samples = transformation, potential, num_samples

    def __call__(self):
        coeffs = self.transformation.base_measure.sample(self.num_samples)
        coeffs_out, log_rn_weight = self.transformation.push_forward(coeffs)
        return (log_rn_weight + self.potential(coeffs_out)).mean()