import math

import torch


class ImportanceCorrection:
    """Turn an approximate posterior into an asymptotically exact one, and measure how good it was
    (Dax et al. 2023).

    The proposal is whatever produced the draws -- an amortised posterior, a reverse-KL fit, a
    guided sampler. The weights are

        log w = -Phi(x) - log d(q)/d(mu0)(x)

    i.e. the true unnormalised posterior over the proposal, both taken against the same reference
    measure so the Radon-Nikodym derivatives are the ones the flow already computes. Nothing is
    trained and nothing is assumed about the proposal beyond being able to evaluate its density.

    Three things come out:
      * a corrected posterior -- reweight, or resample, and every expectation is consistent;
      * `efficiency`, the effective sample size over the sample count, which is a HARD verdict on
        the proposal (Dax et al. report a median of about 10% and treat low values as failures);
      * `log_evidence`, unbiased up to the reference measure's own normalisation, so model
        comparison between structures comes free.

    The catch is honest: efficiency falls with dimension, and a proposal narrower than the
    posterior can score well on every self-consistency check and still be wrong -- which is
    exactly what the efficiency number exposes.
    """

    def __init__(self, flow, draws, potential, context=None):
        self.draws = draws
        with torch.no_grad():
            self.log_ratio = -potential(draws) - flow.log_rn_at(draws, context)
        self.weights = torch.softmax(self.log_ratio, 0)

    @property
    def efficiency(self):
        """Effective sample size over the sample count, in [1/count, 1]."""
        return (1 / (len(self.weights) * self.weights.pow(2).sum())).item()

    @property
    def log_evidence(self):
        """log mean w: the evidence, up to the reference measure's normalisation."""
        return (torch.logsumexp(self.log_ratio, 0) - math.log(len(self.log_ratio))).item()

    def mean(self):
        return self.weights @ self.draws

    def standard_deviation(self):
        centred = self.draws - self.mean()
        return (self.weights @ centred.pow(2)).sqrt()

    def resample(self, count=None):
        """Equally weighted draws, for anything downstream that cannot carry weights (plots)."""
        count = count or len(self.draws)
        offsets = (torch.rand((), device=self.weights.device, dtype=self.weights.dtype)
                   + torch.arange(count, device=self.weights.device, dtype=self.weights.dtype)) / count
        picked = torch.searchsorted(self.weights.cumsum(0), offsets).clamp(max=len(self.draws) - 1)
        return self.draws[picked]
