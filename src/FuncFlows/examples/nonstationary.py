"""Random envelope: f(x) = A(x) u(x), with u a smooth stationary GP of unit pointwise variance and
log A(x) = beta (x - 1/2), beta ~ U(-5, 5) drawn once per function.

Two things follow, and the second is the one that makes this a clean test.

  1. Every draw has its own variance tilt, so the ratio of energy between the halves spans e^{-5}
     to e^{5}. A stationary GP has one tilt for all draws and collapses that spread.

  2. At each x the marginal is a SCALE MIXTURE of Gaussians -- variance exp(2 beta (x-1/2)) with
     beta random -- so it is heavy-tailed, and the more so the further from x = 1/2. Excess
     kurtosis is therefore a U in x: zero at the centre, large at the ends. A Gaussian process has
     excess kurtosis identically zero at every x, by definition, no matter how its covariance is
     fitted. This is not a quantity a GP does badly on; it is one a GP cannot represent.

  Note on what this example used to be. The original drew f(x) = u(x^gamma) -- a random
  LENGTHSCALE. That needs the local spectral content to vary with position, a filter kappa(x, k),
  and no field in the package has one: PointwiseField has a position-dependent amplitude a(x) times
  a global multiplier kappa(k), and OperatorField has kappa(k) with no position at all. A product
  of the two is not a joint function of both. The flow beat the Gaussians there but reproduced only
  about a third of the spread, and that gap was structural, not a training budget. It is the
  motivating case for a layered field alternating a(x) with kappa(k). The roughness ratio is still
  reported below as the honest negative control: nobody matches it, including us.

    python nonstationary.py     (~10 min CPU) -> nonstationary.png
"""
import matplotlib.pyplot as plt
import torch

from _common import (DTYPE, uniform_grid, project, diagonal_measure_from_data, make_flow, field_grid,
                     flow_samples, best_coupling, MomentMatchedGaussian, KernelGP, energy_distance,
                     quantiles, report, Timer)
from FuncFlows.base_measures import CosineBasis, GaussianReferenceMeasure
from FuncFlows.objectives import FlowMatching
from FuncFlows.utils.train import train

torch.manual_seed(4)
M, LATENT_M, PLOT_GRID, NUM_TRAIN, NUM_TEST, TILT = 64, 40, 256, 16000, 2000, 5.0
basis = CosineBasis(M)
grid = uniform_grid(PLOT_GRID)
design = basis.evaluate(grid)
GRID = field_grid(basis, M)                      # 6*k_max for the tanh, not the plotting resolution
print(f"{M} modes, plotting grid {PLOT_GRID}, pointwise-layer grid {GRID}")
latent = GaussianReferenceMeasure(CosineBasis(LATENT_M), alpha=0.02, power=2.0)
latent_design = latent.basis.evaluate(grid)


def simulate(num, chunk=1000):
    out = []
    for done in range(0, num, chunk):
        size = min(chunk, num - done)
        smooth = latent.sample(size) @ latent_design.T                       # [size, PLOT_GRID]
        smooth = smooth / smooth.pow(2).mean(1, keepdim=True).sqrt()
        beta = (2 * torch.rand(size, 1, dtype=DTYPE) - 1) * TILT
        out.append(project(torch.exp(beta * (grid[:, 0] - 0.5)) * smooth, design))
    return torch.cat(out)


def log_energy_ratio(values):
    energy = values ** 2
    half = energy.shape[1] // 2
    return torch.log(energy[:, :half].mean(1) / energy[:, half:].mean(1))


def log_roughness_ratio(values):
    slope2 = torch.diff(values, dim=1) ** 2
    half = slope2.shape[1] // 2
    return torch.log(slope2[:, :half].mean(1) / slope2[:, half:].mean(1))


def excess_kurtosis(values):
    """Per-x excess kurtosis across draws. Identically zero for any Gaussian process."""
    centred = values - values.mean(0, keepdim=True)
    variance = centred.pow(2).mean(0).clamp(min=1e-30)
    return centred.pow(4).mean(0) / variance ** 2 - 3.0


train_coeffs, test_coeffs = simulate(NUM_TRAIN), simulate(NUM_TEST)
mean = train_coeffs.mean(0)

# ---- flow: a per-draw amplitude profile is what the pointwise layer's gates express
base = diagonal_measure_from_data(basis, train_coeffs - mean)
flow = make_flow(base, basis, terms=512, grid_size=GRID, field_modes=48, num_time_modes=6, num_steps=24)
with Timer() as t:
    losses = train(FlowMatching(flow, train_coeffs - mean, batch_size=256, weights=1 / base.scale,
                                coupling=best_coupling()),
                   flow.parameters(), num_steps=16000, learning_rate=2e-3, decay=0.7, decay_every=2000)
print(f"trained in {t.seconds:.0f}s, whitened loss {sum(losses[:100]) / 100:.1f} -> {sum(losses[-100:]) / 100:.1f}")

sets = {"data (test)": test_coeffs,
        "GP (moment-matched)": MomentMatchedGaussian(train_coeffs).sample(NUM_TEST),
        "flow": flow_samples(flow, NUM_TEST, ode_steps=16) + mean}
values = {name: c @ design.T for name, c in sets.items()}
kernel_gp = KernelGP("matern32").fit(grid[::4], (train_coeffs[:200] @ design.T)[:, ::4], noise_std=1e-3)
values[kernel_gp.label()] = kernel_gp.sample_prior(grid, NUM_TEST)

energies = {name: log_energy_ratio(v) for name, v in values.items()}
kurtosis = {name: excess_kurtosis(v) for name, v in values.items()}
roughness = {name: log_roughness_ratio(v) for name, v in values.items()}
edge = torch.cat([torch.arange(PLOT_GRID // 8), torch.arange(7 * PLOT_GRID // 8, PLOT_GRID)])
target_spread = energies["data (test)"].std().item()
target_kurt = kurtosis["data (test)"][edge].mean().item()

report([(name, f"spread {r.std():5.2f} ({r.std() / target_spread:4.0%})   "
               f"edge excess kurtosis {kurtosis[name][edge].mean():6.2f} ({kurtosis[name][edge].mean() / target_kurt:4.0%})   "
               f"E-dist {energy_distance(values[name], values['data (test)']):.4f}")
        for name, r in energies.items()],
       "energy-ratio spread | heavy tails near the ends | distance to test data "
       "(a Gaussian has excess kurtosis 0 everywhere, by definition)")
report([(name, f"spread {r.std():.2f}   q {quantiles(r)}") for name, r in roughness.items()],
       "log ROUGHNESS ratio -- the negative control: needs kappa(x,k), which no field here has")

fig, axes = plt.subplots(1, 6, figsize=(23, 3.5))
for ax, (name, v) in zip(axes[:4], values.items()):
    ax.plot(grid[:, 0], v[:6].T, lw=1)
    ax.set_title(name, fontsize=10)
for name, r in energies.items():
    axes[4].hist(r, bins=50, histtype="step", density=True, label=name)
axes[4].set(xlabel="log energy ratio (left / right)", title="per-draw variance tilt")
axes[4].legend(fontsize=7)
for name, kurt in kurtosis.items():
    axes[5].plot(grid[:, 0], kurt, lw=1.2, label=name)
axes[5].axhline(0, color="k", lw=0.8)
axes[5].set(xlabel="x", title="excess kurtosis (any GP is flat at 0)")
axes[5].legend(fontsize=7)
fig.tight_layout()
fig.savefig("nonstationary.png", dpi=130)
print("saved nonstationary.png")
