
# This is here just as an example
    # it's just a gaussian likelihood but the coefficients are part of the call


class GaussianMisfit:
    """Phi(coeffs) = ½ ||data - forward_map(coeffs)||² / noise_std². forward_map: coeffs [..., r] -> [..., num_data]."""

    def __init__(self, forward_map, data, noise_std):
        self.forward_map, self.data, self.noise_std = forward_map, data, noise_std

    def __call__(self, coeffs):
        return 0.5 * ((self.data - self.forward_map(coeffs)) ** 2).sum(-1) / self.noise_std ** 2