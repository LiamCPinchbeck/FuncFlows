"""Positivity: a log-normal process f = exp(u), u ~ GP with unit pointwise variance. Every sample is
positive and right-skewed, and no Gaussian is either.

This is the easiest win in the set and it is worth being precise about the size of it. A Gaussian
fitted to these data puts a large fraction of its draws below zero somewhere on [0,1]; the flow puts
far fewer there. It does not put none there -- a continuous velocity field cannot produce a hard
boundary in finite time, and the honest claim is the reduction, which the printed table and the
figure both give.

    python positivity.py     (~8 min CPU) -> positivity_samples.png, positivity_minimum.png
"""
import matplotlib.pyplot as plt
import torch

from _common import (DTYPE, uniform_grid, project, diagonal_measure_from_data, make_flow, field_grid,
                     flow_samples, best_coupling, MomentMatchedGaussian, KernelGP, energy_distance,
                     quantiles, report, curves_figure, Timer)
from FuncyFlows.base_measures import FourierBasis, GaussianReferenceMeasure
from FuncyFlows.objectives import FlowMatching
from FuncyFlows.utils.train import train

torch.manual_seed(2)
M, PLOT_GRID, NUM_TRAIN, NUM_TEST, LOG_AMPLITUDE = 48, 256, 8000, 1500, 1.0
basis = FourierBasis(M)
grid = uniform_grid(PLOT_GRID)
design = basis.evaluate(grid)                                   # [PLOT_GRID, M]
GRID = field_grid(basis, M)
print(f"{M} modes, plotting grid {PLOT_GRID}, pointwise-layer grid {GRID}")

# ---- data: f = exp(u) with u normalised to unit pointwise std, projected back onto the same modes
latent = GaussianReferenceMeasure(basis, alpha=0.02, power=2.0)
U_STD = (latent.sample(2000) @ design.T).std().item()


def simulate(num):
    return project(torch.exp(LOG_AMPLITUDE * latent.sample(num) @ design.T / U_STD), design)


train_coeffs, test_coeffs = simulate(NUM_TRAIN), simulate(NUM_TEST)
mean = train_coeffs.mean(0)

# ---- flow: the pointwise layer matters here -- positivity is a pointwise constraint
base = diagonal_measure_from_data(basis, train_coeffs - mean)
flow = make_flow(base, basis, terms=384, grid_size=GRID, field_modes=32, num_time_modes=5, num_steps=20)
with Timer() as t:
    losses = train(FlowMatching(flow, train_coeffs - mean, batch_size=256, weights=1 / base.scale,
                                coupling=best_coupling()),
                   flow.parameters(), num_steps=10000, learning_rate=2e-3, decay=0.7, decay_every=1400)
print(f"trained in {t.seconds:.0f}s, whitened loss {sum(losses[:100]) / 100:.1f} -> {sum(losses[-100:]) / 100:.1f}")

# ---- compare
sets = {"data (test)": test_coeffs,
        "GP (moment-matched)": MomentMatchedGaussian(train_coeffs).sample(NUM_TEST),
        "diag GP (flow base)": base.sample(NUM_TEST) + mean,
        "flow": flow_samples(flow, NUM_TEST, ode_steps=16) + mean}
values = {name: c @ design.T for name, c in sets.items()}
kernel_gp = KernelGP("matern52").fit(grid[::4], (train_coeffs[:200] @ design.T)[:, ::4], noise_std=1e-3)
values[kernel_gp.label()] = kernel_gp.sample_prior(grid, NUM_TEST)

mid = PLOT_GRID // 2
below = {name: (v.min(1).values < 0).double().mean().item() for name, v in values.items()}
report([(name, f"P(min f<0) {below[name]:.3f}   q(f(0.5)) {quantiles(v[:, mid])}   "
               f"E-dist {energy_distance(v, values['data (test)']):.4f}") for name, v in values.items()],
       "positivity | pointwise quantiles at x=0.5 (skew) | distance to test data")
gaussians = [below[name] for name in values if "GP" in name]
print(f"\nflow puts {below['flow']:.3f} of draws below zero; the best Gaussian here puts "
      f"{min(gaussians):.3f}, the worst {max(gaussians):.3f}")

curves_figure(values, grid[:, 0], "positivity_samples.png", zero_line=True)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for name, v in values.items():
    ax.hist(v.min(1).values, bins=60, histtype="step", density=True, label=f"{name}  ({below[name]:.2f})")
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="min_x f(x)",
       title=f"mass below zero: flow {below['flow']:.2f}, best Gaussian {min(gaussians):.2f}, data {below['data (test)']:.2f}")
ax.legend(fontsize=8, title="P(min f < 0)", title_fontsize=8)
fig.tight_layout()
fig.savefig("positivity_minimum.png", dpi=130)
print("saved positivity_samples.png, positivity_minimum.png")
