"""Two-phase inpainting: piecewise-constant fields (a smooth power-law field thresholded at a random
level -- the Darcy-permeability setting). Observed at 256 scattered points.

This is the cleanest example in the set, because the failure of a Gaussian is not a matter of degree.
Away from the data the true posterior puts mass on two values, 0 and 1, and nothing in between. A GP
posterior is Gaussian by construction, so its marginal there is a single smear centred on whatever
the interpolation suggests. The figure that matters is phase_marginals.png: two spikes, or one smear.

The pointwise tanh in PointwiseField is exactly the nonlinearity this target is built from, so if the
flow cannot win here it cannot win anywhere.

Construction: amortised ConditionalFlowMatching from the bank prior to the posterior, conditioned on
the whitened conjugate-posterior mean. Draws are one ODE solve each, independent, no MCMC.

  Why the flow targets the FIELD and not the residual from the conjugate posterior. The residual
  version is prettier on paper -- an untrained flow is then exactly the Gaussian answer, so the GP
  is nested inside the flow rather than competing with it -- and it does not work here. Asking the
  flow to emit r = (binary field - mu), starting from a Gaussian centred at zero, is a far harder
  target than asking it to emit the field: the binary structure is only visible after mu is added
  back, so nothing in the loss rewards producing it. Letting the nonlinear parts *see* mu + r
  (ShiftedField in _common) fixes the field's inputs but not what it is trained to emit, and the
  measured result was 35% of the mass between the phases against the GP's 37-44% -- no real gain.
  Targeting the field directly is the construction that produces binary draws, and it is what an
  unconditional flow on this data has always done.

  Also measured, and worth recording so nobody re-derives it: band-limiting is NOT the obstacle.
  Projecting a truly binary field onto these 440 modes moves the between-phases mass from 1.1% to
  2.2%. Anything above that is the method's own smearing.

    python phase_inpainting.py     (~12 min CPU) -> phase_examples.png, phase_posterior.png, phase_marginals.png
"""
import matplotlib.pyplot as plt
import torch

from _common import (DTYPE, uniform_grid, project, diagonal_measure_from_data, make_flow, field_grid,
                     LinearGaussianPosterior, whitened_context, conditional_samples, best_coupling, KernelGP,
                     score_2d, image_row, inpainting_figure, Timer)
from FuncyFlows.base_measures import FourierBasis
from FuncyFlows.objectives import ConditionalFlowMatching
from FuncyFlows.utils.train import train

torch.manual_seed(2026)
# 80 sensors, not 256. With 256 points at 2% noise the interface is pinned almost everywhere and
# every method's posterior collapses onto it -- the figure then compares three near-identical
# reconstructions. Sparse, noisier data is where a prior earns its keep, and it is the regime the
# posterior-std panel is worth looking at.
SIZE, M, NUM_OBS, NUM_DRAWS, SHARP = 48, 440, 80, 300, 12.0
CONTEXT = 128                             # the conjugate mean keeps ~95% of its energy in these
basis = FourierBasis(M, physical_dim=2)
grid = uniform_grid(SIZE, dim=2)
design = basis.evaluate(grid)
magnitude = basis.wavenumbers.norm(dim=1).clamp(min=1.0)
GRID = field_grid(basis, M)
print(f"{M} modes, image {SIZE}x{SIZE}, pointwise-layer grid {GRID}")


def make_fields(num):
    """Smooth field with slope ~ U(3.0, 3.6), normalised to unit rms, thresholded at level ~ U(-0.6, 0.6)."""
    slope = 3.0 + 0.6 * torch.rand(num, 1, dtype=DTYPE)
    var = magnitude ** (-slope)
    coeffs = (var / var.sum(1, keepdim=True)).sqrt() * torch.randn(num, M, dtype=DTYPE)
    coeffs[:, 0] = 0
    smooth = coeffs @ design.T
    smooth = smooth / smooth.pow(2).mean(1, keepdim=True).sqrt()
    level = -0.6 + 1.2 * torch.rand(num, 1, dtype=DTYPE)
    return 0.5 * (1 + torch.tanh(SHARP * (smooth - level)))            # 0 or 1, soft interface


# ---- fixed sensor array
obs_points = torch.rand(NUM_OBS, 2, dtype=DTYPE)
design_obs = basis.evaluate(obs_points)

reference = make_fields(2000)
reference_coeffs = project(reference, design)
prior_mean = reference_coeffs.mean(0)
bank = diagonal_measure_from_data(basis, reference_coeffs - prior_mean)
image_row(reference[:6], SIZE, "phase_examples.png", vmin=-0.1, vmax=1.1)

noise_std = 0.04
conjugate = LinearGaussianPosterior(bank.variances, design_obs, noise_std)


def observe(coeffs):
    clean = coeffs @ design_obs.T
    return clean + noise_std * torch.randn_like(clean)


def simulate(batch):
    fields = make_fields(batch)
    coeffs = project(fields, design) - prior_mean
    gp_mean = conjugate.mean(observe(coeffs))
    return coeffs, whitened_context(gp_mean, bank.scale, CONTEXT)


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
values = observe(truth_coeffs[None])[0]
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
inpainting_figure(truth, truth, methods, SIZE, "phase_posterior.png", obs_points=obs_points,
                  vmin=-0.1, vmax=1.1)

# ---- the whole result: distribution of values far from the data
#      sharey=False on purpose: the truth histogram spikes to ~30 and squashes every other panel
#      flat if the axes are shared, which is what made the last version of this figure unreadable.
floor = project(truth[None], design)[0] @ design.T          # what 440 modes can do at best
fig, axes = plt.subplots(1, 5, figsize=(17, 3), sharex=True)
panels = [("truth", truth[None]), ("truth, projected\n(the achievable floor)", floor[None])] + list(methods.items())
for ax, (name, v) in zip(axes, panels):
    sample = v[:64, hidden].reshape(-1)
    between = ((sample > 0.25) & (sample < 0.75)).double().mean().item()
    ax.hist(sample, bins=60, density=True, color="0.3" if "truth" in name else "C0")
    ax.set_title(f"{name}\n{between:.0%} between the phases", fontsize=9)
    for phase in (0.0, 1.0):
        ax.axvline(phase, color="C3", lw=0.8, ls="--")
axes[0].set_xlim(-0.3, 1.3)
fig.suptitle("field values away from the observations: two phases, or a smear")
fig.tight_layout()
fig.savefig("phase_marginals.png", dpi=130)

# one number for the same thing: how much mass sits between the phases
for name, v in methods.items():
    between = ((v[:32, hidden] > 0.25) & (v[:32, hidden] < 0.75)).double().mean()
    print(f"{name:32s} mass strictly between the phases: {between:.3f}")
truth_between = ((truth[hidden] > 0.25) & (truth[hidden] < 0.75)).double().mean()
print(f"{'truth':32s} mass strictly between the phases: {truth_between:.3f}")
print("saved phase_examples.png, phase_posterior.png, phase_marginals.png")
