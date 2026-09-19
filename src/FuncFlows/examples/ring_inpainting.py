"""Ring inpainting: 2D images of soft annuli (5 random parameters each), observed everywhere except a
square hole in the middle. The question is what goes in the hole.

A Gaussian cannot answer it. "The bright pixels lie on a circle" is a statement about a 4-parameter
manifold; the best any GP can do is interpolate smoothly across the gap and hedge. The flow is trained
to complete the arc.

Two things make the comparison fair rather than flattering:

  * The flow learns the RESIDUAL from the exact conjugate posterior, whose per-mode standard
    deviations are its base measure. An untrained flow reproduces that Gaussian posterior. The GP is
    inside the flow, so anything the flow gains is something it added.
  * Conditioning is amortised (ConditionalFlowMatching on simulated observation/field pairs), so
    posterior draws are one ODE solve each and independent. The older latent-pCN version of this
    example reported the GP posterior back to itself: with 1500 observations at 2% noise the chains
    never left their initialisation.

    python ring_inpainting.py     (~10 min CPU) -> ring_examples.png, ring_prior.png, ring_posterior.png
"""
import math

import torch

from _common import (DTYPE, uniform_grid, project, diagonal_measure_from_data, make_flow, field_grid,
                     LinearGaussianPosterior, whitened_context, conditional_samples, KernelGP,
                     best_coupling, score_2d, image_row, inpainting_figure, Timer)
from FuncFlows.base_measures import FourierBasis
from FuncFlows.objectives import ConditionalFlowMatching
from FuncFlows.utils.train import train

torch.manual_seed(0)
SIZE, M, CONTEXT, NUM_DRAWS, NOISE = 48, 320, 96, 300, 0.02
basis = FourierBasis(M, physical_dim=2)
grid = uniform_grid(SIZE, dim=2)                                  # [SIZE^2, 2]
design = basis.evaluate(grid)                                     # [SIZE^2, M]
GRID = field_grid(basis, M)                                       # NOT SIZE: the tanh needs 6*k_max
print(f"{M} modes, image {SIZE}x{SIZE}, pointwise-layer grid {GRID}")


def make_rings(num):
    """Widths are 0.04-0.09, not 0.015: a 0.015-wide annulus needs wavenumbers to ~66 and this basis
    stops near 8, so the old example was training on Gibbs rings and scoring the flow on its failure
    to reproduce something it could not represent."""
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(num, 1, dtype=DTYPE)      # noqa: E731
    cx, cy = u(0.35, 0.65), u(0.35, 0.65)
    radius, width, bright = u(0.12, 0.30), u(0.04, 0.09), u(0.5, 1.0)
    dist = ((grid[:, 0] - cx) ** 2 + (grid[:, 1] - cy) ** 2).sqrt()
    return (bright * torch.exp(-0.5 * ((dist - radius) / width) ** 2)).clamp(0, 1)


# ---- the fixed observation operator: everything except a central square hole
hole = int(0.45 * SIZE)
start = (SIZE - hole) // 2
keep = torch.ones(SIZE, SIZE, dtype=torch.bool)
keep[start:start + hole, start:start + hole] = False
keep = keep.reshape(-1)
design_obs = design[keep]

# ---- bank statistics (prior mean and per-mode variances) from a modest sample; training data are
#      simulated fresh at every step, so this is only used to set the base measure and the GP prior
reference = make_rings(3000)
reference_coeffs = project(reference, design)
prior_mean = reference_coeffs.mean(0)
bank = diagonal_measure_from_data(basis, reference_coeffs - prior_mean)
image_row(reference[:6], SIZE, "ring_examples.png", cmap="gray", vmin=0, vmax=1)

# truncation is part of the noise: the ring is not band-limited, so the residual of the projection
# behaves like extra observation error and the likelihood must admit it
truncation = (reference[:200][:, keep] - (reference_coeffs[:200] @ design_obs.T)).pow(2).mean().sqrt().item()
noise_std = max(NOISE, truncation)
print(f"observation noise {NOISE:.3f}, projection residual {truncation:.3f} -> using {noise_std:.3f}")

conjugate = LinearGaussianPosterior(bank.variances, design_obs, noise_std)
residual = conjugate.measure(basis)          # base measure = the Gaussian posterior's own spread


def observe(images):
    """Pixel values outside the hole, prior-mean-subtracted, with noise."""
    clean = images[:, keep] - prior_mean @ design_obs.T
    return clean + noise_std * torch.randn_like(clean)


def simulate(batch):
    """(residual coefficients, context) for ConditionalFlowMatching. Fresh rings every step -- the
    bank is unlimited here, so there is no train/test gap to argue about."""
    images = make_rings(batch)
    coeffs = project(images, design) - prior_mean
    gp_mean = conjugate.mean(observe(images))
    return coeffs - gp_mean, whitened_context(gp_mean, bank.scale, CONTEXT)


# ---- amortised posterior flow
flow = make_flow(residual, basis, terms=256, grid_size=GRID, field_modes=48, num_time_modes=4,
                 num_steps=12, context_dim=CONTEXT)
with Timer() as t:
    losses = train(ConditionalFlowMatching(flow, simulate, residual.sample, batch_size=96,
                                           weights=1 / residual.scale),
                   flow.parameters(), num_steps=6000, learning_rate=2e-3, decay=0.7, decay_every=1000)
print(f"trained in {t.seconds:.0f}s, whitened loss {sum(losses[:100]) / 100:.1f} -> {sum(losses[-100:]) / 100:.1f}")

# ---- one test image, held out by construction
truth_image = make_rings(1)[0]
truth_coeffs = project(truth_image[None], design)[0] - prior_mean
values = observe(truth_image[None])[0]
gp_mean = conjugate.mean(values)
context = whitened_context(gp_mean, bank.scale, CONTEXT)

with Timer() as t:
    learned = conditional_samples(flow, context, NUM_DRAWS) + gp_mean + prior_mean
print(f"{NUM_DRAWS} independent posterior draws in {t.seconds:.1f}s")

gp = conjugate.sample(values, NUM_DRAWS) + prior_mean
kernel_gp = KernelGP("matern32").fit(grid[keep][::4], (values + prior_mean @ design_obs.T)[None, ::4], noise_std)
methods = {kernel_gp.label(): kernel_gp.posterior(grid[keep], values + prior_mean @ design_obs.T,
                                                  noise_std, grid, NUM_DRAWS),
           "GP (bank variances)": gp @ design.T,
           "flow (amortised)": learned @ design.T}
for name, v in methods.items():
    print(f"{name:32s} hidden pixels: {score_2d(v, truth_image, ~keep)}")

observed = torch.where(keep, truth_image, torch.full_like(truth_image, math.nan))
image_row(torch.cat([(conjugate.sample(values, 3) + prior_mean), learned[:3]]) @ design.T, SIZE,
          "ring_prior.png", titles=["GP posterior"] * 3 + ["flow posterior"] * 3, cmap="gray", vmin=0, vmax=1)
inpainting_figure(truth_image, observed, methods, SIZE, "ring_posterior.png", cmap="gray", vmin=0, vmax=1)
print("saved ring_examples.png, ring_prior.png, ring_posterior.png")
