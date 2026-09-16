import torch


class ReverseKL:
    """E_{coeffs ~ mu0}[ log_rn_weight + Phi(f(coeffs)) ]: KL(f#mu0 || posterior) up to the evidence.

    context, if given, is handed to the transformation unchanged (one context for the whole
    objective -- the single-observation posterior case).

    path_gradient=True uses the path-gradient (sticking-the-landing) estimator of Vaitl et al.
    2022. The reparameterised gradient splits into a path term and a term that differentiates the
    density at a FIXED sample; the second has zero mean and pure variance, and it vanishes only in
    expectation, so dropping it leaves an unbiased estimator with strictly less noise. The gain
    grows as the fit improves: in the Gaussian check in this package's notes the variance ratio is
    about 1.4x far from the optimum and over 1000x near it, which is exactly where the ordinary
    estimator stalls. It also skips the trace on the forward pass, so a step is no more expensive.
    """

    def __init__(self, transformation, potential, num_samples=30, context=None, path_gradient=False):
        self.transformation, self.potential, self.num_samples = transformation, potential, num_samples
        self.context, self.path_gradient = context, path_gradient

    def __call__(self):
        coeffs = self.transformation.base_measure.sample(self.num_samples)
        if not self.path_gradient:
            coeffs_out, log_rn_weight = self.transformation.push_forward(coeffs, self.context)
            return (log_rn_weight + self.potential(coeffs_out)).mean()
        coeffs_out = self.transformation.transport(coeffs, self.context)     # no trace needed
        value, slope = self.integrand_and_slope(coeffs_out.detach())
        surrogate = (slope * coeffs_out).sum(-1).mean()
        return value + surrogate - surrogate.detach()       # value reports, gradient is the path term

    def integrand_and_slope(self, points):
        """log (dq/dmu0)(x) + Phi(x) and its x-derivative, with the flow's parameters held fixed."""
        moving = [parameter for parameter in self.transformation.parameters()
                  if parameter.requires_grad]
        for parameter in moving:
            parameter.requires_grad_(False)
        try:
            with torch.enable_grad():
                state = points.detach().requires_grad_(True)
                integrand = (self.transformation.log_rn_at(state, self.context)
                             + self.potential(state))
                slope, = torch.autograd.grad(integrand.sum(), state)
        finally:
            for parameter in moving:
                parameter.requires_grad_(True)
        return integrand.mean().detach(), slope.detach()
