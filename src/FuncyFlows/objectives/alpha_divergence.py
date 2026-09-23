import math

import torch


class AlphaDivergence:
    """Renyi / alpha divergence between the flow q and the posterior, as a mass-covering
    alternative to reverse KL:

        D_alpha = log E_q[w^alpha] / (alpha (alpha - 1)),    w = ptilde / q.

    alpha = 2 is the chi^2 divergence of FAB (Midgley et al. 2023): the variance of the
    importance weights, which punishes exactly the regions where q is too narrow. 0 < alpha < 1
    sits between reverse KL (alpha -> 0) and the evidence (alpha -> 1 from below in the Li &
    Turner 2016 convention). The 1/(alpha - 1) is essential: for 0 < alpha < 1 the expectation is
    MAXIMISED at q = p, so dividing by alpha alone would push the flow away from the posterior.
    """

    def __init__(self, transformation, potential, num_samples=30, alpha=2.0, context=None,
                 prior_fraction=0.5):
        if alpha in (0.0, 1.0):
            raise ValueError("alpha = 0 and alpha = 1 are the KL limits; use ReverseKL")
        self.transformation, self.potential, self.num_samples = transformation, potential, num_samples
        self.alpha, self.context, self.prior_fraction = alpha, context, prior_fraction

    def __call__(self):
        flow, alpha = self.transformation, self.alpha
        num_prior = int(round(self.prior_fraction * self.num_samples))
        num_flow = self.num_samples - num_prior
        points, log_ratio = [], []                              # log_ratio = log dq/dmu0 at the point
        if num_flow:
            coeffs_out, log_rn = flow.push_forward(flow.base_measure.sample(num_flow), self.context)
            points.append(coeffs_out)
            log_ratio.append(log_rn)
        if num_prior:
            prior_points = flow.base_measure.sample(num_prior)
            points.append(prior_points)
            log_ratio.append(flow.log_rn_at(prior_points, self.context))
        points, log_ratio = torch.cat(points), torch.cat(log_ratio)
        log_potential = -self.potential(points)
        if num_prior == 0 or num_flow == 0:
            log_proposal = log_ratio if num_prior == 0 else torch.zeros_like(log_ratio)
        else:
            fraction = num_prior / self.num_samples
            log_proposal = torch.logaddexp(math.log(1 - fraction) + log_ratio,
                                           torch.full_like(log_ratio, math.log(fraction)))

        log_terms = alpha * log_potential + (1 - alpha) * log_ratio - log_proposal
        return (torch.logsumexp(log_terms, 0) - math.log(len(log_terms))) / (alpha * (alpha - 1))
