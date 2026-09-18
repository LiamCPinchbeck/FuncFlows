import torch


def coverage_curve(samples, truths, references=None, weights=None, levels=None):
    """TARP: expected coverage from samples alone (Lemos et al. 2023).

    For each test case, count the fraction of posterior draws that land closer to a random
    reference point than the truth does. If the posterior is right, those fractions are uniform,
    so plotting the empirical CDF against the credibility level gives the diagonal.

    Unlike rank statistics it needs no density and no per-dimension marginalisation, and unlike
    simulation-based calibration it is necessary AND sufficient: a model that ignores the data and
    returns the prior is perfectly self-consistent but fails this.

    Read the deviation like this: BELOW the diagonal is overconfident (posteriors too narrow),
    ABOVE is conservative (too wide). Both are failures, and they look nothing alike, which is
    why the curve is worth more than the scalar.

    samples [cases, draws, modes]; truths [cases, modes]; references default to the truths rolled
    by one, which is a draw from the same marginal without needing the prior. weights scale the
    distance, so pass the Cameron-Martin weights to measure in the metric the prior lives in.
    """
    levels = torch.linspace(0.05, 0.95, 19) if levels is None else levels
    levels = levels.to(device=samples.device, dtype=samples.dtype)
    if references is None:
        references = truths.roll(1, dims=0)
    scaled_samples = samples if weights is None else samples * weights
    scaled_truths = truths if weights is None else truths * weights
    scaled_references = references if weights is None else references * weights
    truth_distance = (scaled_truths - scaled_references).norm(dim=-1)
    draw_distance = (scaled_samples - scaled_references.unsqueeze(1)).norm(dim=-1)
    closer = (draw_distance < truth_distance.unsqueeze(1)).to(samples.dtype).mean(1)
    coverage = (closer.unsqueeze(0) <= levels.unsqueeze(1)).to(samples.dtype).mean(1)
    return levels, coverage


def coverage_error(levels, coverage):
    """One number for the curve: the largest gap from the diagonal, signed by direction.

    Positive means conservative on balance, negative overconfident. On a Gaussian check the exact
    posterior scored 0.03 with 800 cases, a posterior 40% too narrow scored -0.18, and one 60% too
    wide +0.19 -- so roughly 0.05 is the noise floor at that many cases.
    """
    deviation = coverage - levels
    largest = deviation.abs().argmax()
    return deviation[largest].item()
