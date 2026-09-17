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

    Where the samples come from decides whether this is usable. Estimated with draws from q
    alone, E_q[w^alpha] for alpha > 1 only sees where q already has mass, and the gradient can
    make q ever narrower (log q -> infinity, w -> 0, loss -> -infinity) without the estimator
    ever noticing the posterior mass it left behind -- the true quantity is bounded below by
    Z^alpha, the estimate is not. FAB fixes this with AIS toward ptilde^alpha q^(1-alpha). The fix
    here is simpler: a defensive mixture proposal r = (1 - f) q + f mu0, evaluated by importance
    weights,

        E_q[w^alpha] = E_r[ ptilde^alpha q^(1-alpha) / r ],

    with everything written against the reference measure, so only log dq/dmu0 (the quantity the
    flow already computes) and Phi appear. The prior half covers the whole posterior support, so
    a q that collapses is caught by the prior draws it no longer covers. prior_fraction = 0 is
    the plain q-sample estimator, kept for comparison; the default 0.5 is what to use.

    log-sum-exp throughout, so the weights never leave log space.
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
            # single proposal (q or mu0): log of the integrand relative to that proposal
            log_proposal = log_ratio if num_prior == 0 else torch.zeros_like(log_ratio)
        else:
            fraction = num_prior / self.num_samples
            log_proposal = torch.logaddexp(math.log(1 - fraction) + log_ratio,
                                           torch.full_like(log_ratio, math.log(fraction)))
        # ptilde^alpha q^(1-alpha) / r, all relative to mu0: alpha (-Phi) + (1 - alpha) log_ratio - log_proposal
        log_terms = alpha * log_potential + (1 - alpha) * log_ratio - log_proposal
        return (torch.logsumexp(log_terms, 0) - math.log(len(log_terms))) / (alpha * (alpha - 1))
