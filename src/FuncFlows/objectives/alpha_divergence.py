import math

import torch


class AlphaDivergence:
    """Mass-covering alternative to reverse KL: minimise log E_q[w^alpha] / alpha, w = ptilde / q.

    Reverse KL is mode-seeking -- it is finite only where q has mass, so a posterior mode that the
    flow never visits costs nothing and is never found. The alpha = 2 divergence of FAB (Midgley
    et al. 2023) is mass-covering instead: it is the variance of the importance weights, so it
    punishes exactly the regions where q is too narrow.

    This is FAB's divergence with its samples drawn from the flow by reparameterisation, NOT its
    annealed-importance-sampling bootstrap. The divergence changes where the optimum sits; AIS is
    what finds modes the flow has never sampled. Without it, a mode that the flow gives no mass at
    all still contributes nothing, so run this as a wider-posterior objective rather than as a
    mode-discovery one.

    log-sum-exp, so the weights never leave log space: w^2 overflows long before the fit is good.
    """

    def __init__(self, transformation, potential, num_samples=30, alpha=2.0, context=None):
        if alpha in (0.0, 1.0):
            raise ValueError("alpha = 0 and alpha = 1 are the KL limits; use ReverseKL")
        self.transformation, self.potential, self.num_samples = transformation, potential, num_samples
        self.alpha, self.context = alpha, context

    def __call__(self):
        coeffs = self.transformation.base_measure.sample(self.num_samples)
        coeffs_out, log_rn_weight = self.transformation.push_forward(coeffs, self.context)
        log_weight = -self.potential(coeffs_out) - log_rn_weight      # log ptilde/q, up to log Z
        # Renyi form: D_alpha = log E_q[w^alpha] / (alpha (alpha - 1)). The 1/(alpha - 1) factor
        # matters: for 0 < alpha < 1, E_q[w^alpha] is MAXIMISED at q = p (Jensen, concave), so
        # dividing by alpha alone would push the flow away from the posterior. For alpha = 2 this
        # is the previous 1/2 log E[w^2], unchanged.
        return (torch.logsumexp(self.alpha * log_weight, 0)
                - math.log(len(log_weight))) / (self.alpha * (self.alpha - 1))
