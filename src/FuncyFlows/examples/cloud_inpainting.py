"""Cloud inpainting: synthetic molecular-cloud column density (the generator from the original study,
verbatim). Scale-free turbulent latents rendered as a positive, edge-eroded field: background deck +
diffuse envelope + foreground body whose margins are eaten by the finer bands.

A GP can match the two-point statistics of these fields exactly and still never draw one. The
signature is not the power spectrum, it is the morphology: hard-edged bodies against a near-empty
floor, which is a statement about the phase relationships a Gaussian discards.

Construction: amortised ConditionalFlowMatching from the bank prior to the posterior, conditioned on
the whitened conjugate-posterior mean (128 modes carry ~95% of it at this sensor count). Draws are
one ODE solve each, independent, no MCMC.

The flow targets the FIELD, not the residual from the conjugate posterior -- see the long note in
phase_inpainting.py for the measurement that settled it. The residual construction has the nicer
nesting property and produced draws indistinguishable from the GP's.

The inference basis is larger than it used to be. Generating at 900 modes and inferring at 240 meant
the reconstruction could not represent the truth no matter how good the prior was; most of the
visible gap was bandwidth, not modelling. Inference now runs at 440 modes and the remaining
truncation is absorbed into the noise level, where it belongs.

    python cloud_inpainting.py     (~15 min CPU) -> cloud_examples.png, cloud_prior.png, cloud_posterior.png
"""
import torch

from _common import (DTYPE, uniform_grid, project, diagonal_measure_from_data, make_flow, field_grid,
                     LinearGaussianPosterior, whitened_context, conditional_samples, best_coupling, KernelGP,
                     score_2d, image_row, inpainting_figure, Timer)
from FuncyFlows.base_measures import FourierBasis
from FuncyFlows.objectives import ConditionalFlowMatching
from FuncyFlows.utils.train import train

torch.manual_seed(2026)
# 100 sensors, not 256: with a quarter of the pixels observed at 1% noise there is almost nothing
# left to be uncertain about, and every method returns the same picture. The interesting regime is
# the one where the prior has to supply the morphology.
SIZE, M, CLOUD_M, NUM_OBS, NUM_DRAWS = 48, 440, 900, 100, 300
CONTEXT = 128                             # the conjugate mean keeps ~95% of its energy in these
basis = FourierBasis(M, physical_dim=2)                             # inference basis
cloud_basis = FourierBasis(CLOUD_M, physical_dim=2)                 # generation basis (nested: same first M modes)
grid = uniform_grid(SIZE, dim=2)
design = basis.evaluate(grid)
cloud_design = cloud_basis.evaluate(grid)
GRID = field_grid(basis, M)
print(f"{M} modes (generated at {CLOUD_M}), image {SIZE}x{SIZE}, pointwise-layer grid {GRID}")

# ---- the cloud generator, ported from the original cloud_inpainting.py. Every constant is a property
#      of the FIELDS: scale-free turbulent latents rendered as a positive, edge-eroded column density with
#      a soft background deck, a diffuse envelope and a foreground body eaten away by the finer bands.
LATENT_STD, SLOPE_RANGE, STRETCH_RANGE = 1.60, (2.4, 3.2), (0.55, 1.80)
BAND_EDGES = [0.5, 2.0, 3.5, 5.5, 8.0, 1e9]
BASE_WEIGHTS, DETAIL_WEIGHTS, BACK_WEIGHTS = [1.0, 0.7, 0.5, 0.35, 0.25], [0.0, 0.0, 0.5, 0.35, 0.25], [1.0, 0.55, 0.25, 0.1, 0.05]
DIFFUSE_WEIGHT, DIFFUSE_GAIN, COVERAGE_LEVEL, EROSION_STRENGTH, EDGE_SOFTNESS, OPACITY_GAIN, FLOOR_LEVEL = 0.32, 0.5, 0.25, 0.45, 4.0, 2.5, 0.05
BACK_FRACTION, BACK_COVERAGE, BACK_SOFTNESS, BACK_OPACITY, BACK_WEIGHT = 0.35, 0.10, 1.5, 1.20, 0.78
VMAX = FLOOR_LEVEL + BACK_WEIGHT + DIFFUSE_WEIGHT * 3.0 + 1.0       # fixed colour scale for every cloud panel

k = cloud_basis.wavenumbers                                        # [CLOUD_M, 2]
k_mag = k.norm(dim=1)
bands = [((k_mag >= lo) & (k_mag < hi)).to(DTYPE) for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
shape_weights = sum(w * b for w, b in zip(BASE_WEIGHTS, bands))
back_weights = sum(w * b for w, b in zip(BACK_WEIGHTS, bands))
detail_ladder = [(w, b) for w, b in zip(DETAIL_WEIGHTS, bands) if w]
back_mask = (torch.rand(CLOUD_M, generator=torch.Generator().manual_seed(99)) < BACK_FRACTION).to(DTYPE)
front_mask = 1 - back_mask


def latents(num):
    """Pure power-law spectrum, random slope and anisotropy. No characteristic length."""
    slope = SLOPE_RANGE[0] + (SLOPE_RANGE[1] - SLOPE_RANGE[0]) * torch.rand(num, 1, dtype=DTYPE)
    stretch = STRETCH_RANGE[0] + (STRETCH_RANGE[1] - STRETCH_RANGE[0]) * torch.rand(num, 1, dtype=DTYPE)
    magnitude = ((stretch * k[:, 0]) ** 2 + (k[:, 1] / stretch) ** 2).sqrt()
    var = magnitude.clamp(min=1.0) ** (-slope)
    return LATENT_STD * (var / var.sum(1, keepdim=True)).sqrt() * torch.randn(num, CLOUD_M, dtype=DTYPE)


def normalised(coeffs, design_at):
    """Unit spatial std from the coefficients, so grid and observation points agree."""
    return (coeffs @ design_at.T) / coeffs.pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-12)


def soft_ramp(values, softness):
    """Smooth max(values, 0): zero below, linear above, no pedestal."""
    return (torch.nn.functional.softplus(softness * values) - 0.6931471805599453) / softness


def density(coeffs, design_at):
    back_field = normalised(coeffs * back_mask * back_weights, design_at)
    back_deck = BACK_WEIGHT * (1 - torch.exp(-BACK_OPACITY * soft_ramp(back_field - BACK_COVERAGE, BACK_SOFTNESS)))
    front = coeffs * front_mask
    shape_field = normalised(front * shape_weights, design_at)
    detail_field = sum(w * normalised(front * b, design_at).abs() for w, b in detail_ladder)
    diffuse = DIFFUSE_WEIGHT * torch.exp(DIFFUSE_GAIN * shape_field)
    eroded = shape_field - COVERAGE_LEVEL - EROSION_STRENGTH * detail_field
    body = 1 - torch.exp(-OPACITY_GAIN * soft_ramp(eroded, EDGE_SOFTNESS))
    return FLOOR_LEVEL + back_deck + diffuse + body


def make_fields(num, chunk=200):
    return torch.cat([density(latents(min(chunk, num - done)), cloud_design)
                      for done in range(0, num, chunk)])


# ---- fixed sensor array: amortisation needs one design matrix, not a new one per case
obs_points = torch.rand(NUM_OBS, 2, dtype=DTYPE)
design_obs = basis.evaluate(obs_points)

reference = make_fields(2000)
reference_coeffs = project(reference, design)
prior_mean = reference_coeffs.mean(0)
bank = diagonal_measure_from_data(basis, reference_coeffs - prior_mean)
image_row(reference[:6], SIZE, "cloud_examples.png", cmap="magma", vmin=0, vmax=VMAX)

# the field is not band-limited at M modes; the projection residual at the sensors is real error and
# the likelihood has to admit it, or the posterior is over-confident about structure it cannot hold
_probe = latents(200)
truncation = (density(_probe, cloud_basis.evaluate(obs_points))
              - project(density(_probe, cloud_design), design) @ design_obs.T).pow(2).mean().sqrt().item()
noise_std = max(0.01 * reference.abs().max().item(), truncation)
print(f"sensor noise floor {0.01 * reference.abs().max().item():.4f}, "
      f"projection residual {truncation:.4f} -> using {noise_std:.4f}")

conjugate = LinearGaussianPosterior(bank.variances, design_obs, noise_std)


def observe_coeffs(coeffs):
    """Sensor readings from coefficients, prior-mean-subtracted, with noise."""
    clean = coeffs @ design_obs.T
    return clean + noise_std * torch.randn_like(clean)


# Rendering a cloud costs several [batch, 900] x [900, SIZE^2] products, so generating fresh fields
# every training step spends more time on data than on the model. 12k fields is far more than this
# many parameters can overfit, and the observation noise is redrawn every step anyway.
BANK = torch.cat([reference_coeffs - prior_mean, project(make_fields(10000), design) - prior_mean])
print(f"training bank: {len(BANK)} fields")


def simulate(batch):
    coeffs = BANK[torch.randint(len(BANK), (batch,))]
    gp_mean = conjugate.mean(observe_coeffs(coeffs))
    return coeffs, whitened_context(gp_mean, bank.scale, CONTEXT)


# ---- amortised posterior flow
flow = make_flow(bank, basis, terms=256, grid_size=GRID, field_modes=64, num_time_modes=4,
                 num_steps=12, context_dim=CONTEXT)
with Timer() as t:
    losses = train(ConditionalFlowMatching(flow, simulate, bank.sample, batch_size=128,
                                           weights=1 / bank.scale, coupling=best_coupling()),
                   flow.parameters(), num_steps=15000, learning_rate=2e-3, decay=0.7, decay_every=2000)
print(f"trained in {t.seconds:.0f}s, whitened loss {sum(losses[:100]) / 100:.1f} -> {sum(losses[-100:]) / 100:.1f}")

# ---- one test field
truth = make_fields(1)[0]
truth_coeffs = project(truth[None], design)[0] - prior_mean
values = observe_coeffs(truth_coeffs[None])[0]
gp_mean = conjugate.mean(values)
context = whitened_context(gp_mean, bank.scale, CONTEXT)

with Timer() as t:
    learned = conditional_samples(flow, context, NUM_DRAWS, ode_steps=16) + prior_mean
print(f"{NUM_DRAWS} independent posterior draws in {t.seconds:.1f}s")

gp = conjugate.sample(values, NUM_DRAWS) + prior_mean
raw_values = values + prior_mean @ design_obs.T
kernel_gp = KernelGP("matern32").fit(obs_points, raw_values[None], noise_std)
methods = {kernel_gp.label(): kernel_gp.posterior(obs_points, raw_values, noise_std, grid, NUM_DRAWS),
           "GP (bank variances)": gp @ design.T,
           "flow (amortised)": learned @ design.T}

far = torch.cdist(grid, obs_points).min(1).values
hidden = far > far.median()
for name, v in methods.items():
    print(f"{name:32s} far-from-data pixels: {score_2d(v, truth, hidden)}")

image_row(torch.cat([gp[:3], learned[:3]]) @ design.T, SIZE, "cloud_prior.png",
          titles=["GP posterior"] * 3 + ["flow posterior"] * 3, cmap="magma", vmin=0, vmax=VMAX)
inpainting_figure(truth, truth, methods, SIZE, "cloud_posterior.png", obs_points=obs_points,
                  cmap="magma", vmin=0, vmax=VMAX)
print("saved cloud_examples.png, cloud_prior.png, cloud_posterior.png")
