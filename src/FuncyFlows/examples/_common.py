"""Shared machinery for the FuncyFlows examples. Nothing here is specific to one example.

Coefficients everywhere: a function on [0,1]^d is its vector of coefficients on a FourierBasis /
CosineBasis. `project` gets you there from values on a uniform grid; `basis.evaluate(points)` gets
you back (values = coeffs @ design.T).

Two conventions the inpainting examples all share, both worth knowing before you read one:

  * The conjugate (linear-Gaussian) posterior is computed in closed form and the flow learns the
    RESIDUAL from it, with the conjugate posterior's own standard deviations as its base measure.
    An untrained flow is therefore already the Gaussian answer; training can only add non-Gaussian
    structure. The GP baseline is nested inside the flow, not competing with it.
  * Conditioning is AMORTISED: one ConditionalFlowMatching model is trained on freshly simulated
    (observation, field) pairs, and posterior draws are one ODE solve each. No MCMC, no burn-in,
    independent samples. `posterior_from_prior_flow` (latent pCN) is still here and still exact,
    but with a sharp likelihood it needs far more steps than an example should take -- see its
    docstring.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from FuncyFlows.base_measures import GaussianReferenceMeasure
from FuncyFlows.transports.continuous import (ContinuousTransformation, SumField, LinearField,
                                             MatrixField, PointwiseField, TimeBasisConditioner,
                                             DataConditioner, VectorField)
from FuncyFlows.samplers import latent_pcn

DTYPE = torch.float64


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds = time.perf_counter() - self.start


# ------------------------------------------------------------------ grids and projection
def uniform_grid(size, dim=1):
    """Cell-centred uniform grid on [0,1]^dim -> [size^dim, dim]. Fourier modes are orthogonal on it."""
    axis = (torch.arange(size, dtype=DTYPE) + 0.5) / size
    return torch.cartesian_prod(*[axis] * dim).reshape(-1, dim)


def project(values, design):
    """Values on a uniform grid [n, points] -> coefficients [n, modes], by the rectangle rule.
    design = basis.evaluate(grid), orthonormal columns."""
    return values @ design / design.shape[0]


def diagonal_measure_from_data(basis, coeffs):
    """The diagonal Gaussian with the data's per-mode variances: the flow's base measure and the
    'GP that knows the bank' baseline in one object."""
    return GaussianReferenceMeasure(basis, variances=coeffs.var(0).clamp(min=1e-10))


def field_grid(basis, num_active, margin=6):
    """Smallest grid that resolves the harmonics a tanh generates from this truncation.

    A tanh's cubic term reaches 3*k_max and Nyquist doubles it, so the pointwise layer wants
    grid_size >= 6*k_max or the projection back aliases and the closed-form trace is wrong. This
    is deliberately NOT tied to the image resolution -- they are independent numbers and tying
    them is how the old examples ended up aliasing."""
    if hasattr(basis, "wavenumbers"):
        k_max = int(basis.wavenumbers[:num_active].abs().max().item())
    else:                                                   # cosine: lambda_k = (k pi)^2
        k_max = int(round(basis.laplacian_eigenvalues[:num_active].max().sqrt().item() / torch.pi))
    size = margin * max(k_max, 1)
    return 1 << (size - 1).bit_length()                      # round up to a power of two for the FFT


# ------------------------------------------------------------------ flows
class ShiftedField(VectorField):
    r"""Let a nonlinear field see mu(c) + r instead of r.

    When the flow transports the RESIDUAL from a conjugate posterior, the state it integrates is
    r = v - mu(c). Everything that makes these targets interesting -- where the phase boundary is,
    where the cloud edge is -- lives in mu, not in r. A pointwise tanh applied to r alone is
    therefore blind to it: it can sharpen, but it cannot know which pixels to sharpen. Adding the
    shift back before the inner field runs fixes that.

    The shift is a function of the context only, never of r, so the Jacobian is unchanged and the
    closed-form trace still holds exactly.

    The context is expected to carry the whitened posterior mean in its first `num_modes` entries,
    so `shift = context[..., :num_modes] * scale` recovers mu. `scale` should be the BANK standard
    deviations, not the base measure's: mu + r is a whole field and has bank scale, while the base
    measure is the much narrower conjugate posterior. Whitening a field by the wrong scale is how
    you saturate every tanh before training starts.
    """

    def __init__(self, inner, num_modes, scale):
        super().__init__()
        self.inner, self.num_modes = inner, num_modes
        self.register_buffer("shift_scale", scale)

    @property
    def supports_batched_time(self):
        return self.inner.supports_batched_time

    def _shifted(self, coeffs, context):
        if context is None:
            return coeffs
        shift = context[..., :self.num_modes] * self.shift_scale
        return coeffs + torch.nn.functional.pad(shift, (0, coeffs.shape[-1] - self.num_modes))

    def velocity(self, coeffs, time_value, context=None):
        return self.inner.velocity(self._shifted(coeffs, context), time_value, context)

    def trace(self, coeffs, time_value, context=None):
        return self.inner.trace(self._shifted(coeffs, context), time_value, context)

    def velocity_and_trace(self, coeffs, time_value, context=None):
        return self.inner.velocity_and_trace(self._shifted(coeffs, context), time_value, context)


def make_flow(measure, basis=None, terms=256, grid_size=None, num_time_modes=4, num_steps=16,
              field_modes=32, context_dim=0, shift_modes=0, shift_scale=None):
    """LinearField + one wide tanh layer; add a PointwiseField (spatial, one FNO layer) when a
    grid_size is given. Pass context_dim to make every part conditional. Every trace is closed-form.

    shift_modes / shift_scale wrap the NONLINEAR parts in ShiftedField, so they act on mu + r while
    the linear part keeps acting on r. Used by the inpainting examples; see ShiftedField."""
    M = measure.num_functions
    shifting = bool(shift_modes)
    inner_scale = shift_scale if shifting else measure.scale
    if context_dim:
        linear = LinearField(M, context_dim, num_time_modes=num_time_modes, mode_scale=measure.scale)
        nonlinear = [MatrixField(DataConditioner(M, terms, context_dim, num_time_modes=num_time_modes),
                                 mode_scale=inner_scale)]
    else:
        linear = LinearField(M, num_time_modes=num_time_modes)
        nonlinear = [MatrixField(TimeBasisConditioner(M, terms, num_time_modes=num_time_modes),
                                 mode_scale=inner_scale)]
    if grid_size:
        nonlinear.append(PointwiseField(basis, M, grid_size, num_time_modes=num_time_modes,
                                        num_field_modes=field_modes, num_spectral_modes=8,
                                        context_dim=context_dim, mode_scale=inner_scale))
    if shifting:
        nonlinear = [ShiftedField(part, shift_modes, shift_scale) for part in nonlinear]
    return ContinuousTransformation(measure, SumField(linear, *nonlinear), num_steps=num_steps)


@torch.no_grad()
def flow_samples(flow, num, batch=512, ode_steps=8):
    saved, flow.num_steps = flow.num_steps, ode_steps
    try:
        return torch.cat([flow.transport(flow.base_measure.sample(min(batch, num - done)))
                          for done in range(0, num, batch)])
    finally:
        flow.num_steps = saved


@torch.no_grad()
def conditional_samples(flow, context, num, batch=256, ode_steps=12):
    """Posterior draws for ONE observation from an amortised flow: one ODE solve per draw, all
    independent. context is the [context_dim] summary that flow was trained to condition on."""
    saved, flow.num_steps = flow.num_steps, ode_steps
    context = context.reshape(1, -1)
    try:
        out = []
        for done in range(0, num, batch):
            size = min(batch, num - done)
            out.append(flow.transport(flow.base_measure.sample(size), context.expand(size, -1)))
        return torch.cat(out)
    finally:
        flow.num_steps = saved


def posterior_from_prior_flow(prior_flow, potential, num_draws, init=None, num_chains=128, num_steps=800,
                              ode_steps=8, target_acceptance=0.25):
    """Exact posterior samples under a LEARNED prior: pCN in the flow's latent space (Cotter et al.
    2013). The flow's Jacobian never enters, so any trained prior flow works, and unlike the
    amortised route it needs no retraining when the observation changes.

    The catch, and the reason the inpainting examples no longer use it: with a few hundred
    observations at 1-2% noise the likelihood is razor-sharp, the adapted beta becomes tiny, and a
    few hundred steps leave the chains sitting on their initialisation. `moved` in the returned
    info is the mean relative distance travelled from `init` -- below ~0.1 the answer you are
    looking at is the initialisation, not the posterior."""
    saved, prior_flow.num_steps = prior_flow.num_steps, ode_steps
    try:
        draws, info = latent_pcn(prior_flow, potential, num_chains=num_chains, num_steps=num_steps, beta=0.1,
                                 init=init, burn=num_steps // 3, thin=5, adapt_to=target_acceptance)
    finally:
        prior_flow.num_steps = saved
    if init is not None:
        reference = init.mean(0)
        info["moved"] = ((draws - reference).norm(dim=-1).mean() / reference.norm().clamp(min=1e-12)).item()
    return draws[torch.randperm(len(draws))[:num_draws]], info


def best_coupling():
    """Minibatch optimal-transport coupling straightens flow-matching paths; needs scipy."""
    try:
        import scipy.optimize  # noqa: F401
        return "optimal"
    except ImportError:
        return "independent"


# ------------------------------------------------------------------ Gaussian baselines
class MomentMatchedGaussian:
    """Empirical mean + FULL covariance of the coefficients: the best any GP can do on this data."""

    def __init__(self, coeffs):
        self.mean = coeffs.mean(0)
        centred = coeffs - self.mean
        cov = centred.T @ centred / (len(coeffs) - 1)
        self.chol = torch.linalg.cholesky(cov + 1e-8 * cov.diagonal().mean() * torch.eye(len(cov), dtype=DTYPE))

    def sample(self, num):
        return self.mean + torch.randn(num, len(self.mean), dtype=DTYPE) @ self.chol.T


class LinearGaussianPosterior:
    """Exact posterior for values = design_obs @ coeffs + noise under a diagonal Gaussian prior.

    The design is FIXED across cases, so the precision, its Cholesky and the posterior covariance
    are built once and every case is one triangular solve. Three roles in the examples:

      * the 'GP that knows the bank' baseline, sampled exactly with the full covariance;
      * the amortised flow's conditioning summary, through `mean`;
      * the amortised flow's base measure, through `measure()` -- its per-mode standard deviations.
    """

    def __init__(self, variances, design_obs, noise_std):
        self.design_obs, self.noise_std = design_obs, noise_std
        precision = torch.diag(1 / variances) + design_obs.T @ design_obs / noise_std ** 2
        self.chol = torch.linalg.cholesky(precision)
        cov = torch.cholesky_inverse(self.chol)
        self.cov = 0.5 * (cov + cov.T)
        self.std = self.cov.diagonal().clamp(min=1e-14).sqrt()
        jitter = 1e-12 * self.cov.diagonal().mean() * torch.eye(len(self.cov), dtype=DTYPE)
        self.cov_chol = torch.linalg.cholesky(self.cov + jitter)

    def mean(self, values):
        """values [..., n] (already prior-mean-subtracted) -> posterior mean coefficients [..., M]."""
        rhs = (values @ self.design_obs / self.noise_std ** 2)
        return torch.cholesky_solve(rhs.reshape(-1, rhs.shape[-1]).T, self.chol).T.reshape(rhs.shape)

    def sample(self, values, num_draws):
        return self.mean(values) + torch.randn(num_draws, len(self.std), dtype=DTYPE) @ self.cov_chol.T

    def measure(self, basis):
        """The flow's base measure: an untrained flow then reproduces this Gaussian posterior
        (up to its off-diagonal correlations, which LinearField has to learn back)."""
        return GaussianReferenceMeasure(basis, variances=self.std ** 2)


def whitened_context(gp_mean, scale, context_modes):
    """The conditioning summary: the low modes of the conjugate posterior mean, whitened.

    Whitening is not cosmetic -- without it the drift weights for the high modes would have to
    reach 1/sigma_k to matter, and they never get there. Truncating to `context_modes` keeps the
    conditional parameter count linear in a small number rather than in M."""
    return gp_mean[..., :context_modes] / scale[:context_modes]


# ------------------------------------------------------------------ kernel GP (no bank): the standard baseline
KERNELS = {
    "matern12": lambda r, ell: torch.exp(-r / ell),
    "matern32": lambda r, ell: (1 + 3 ** 0.5 * r / ell) * torch.exp(-(3 ** 0.5) * r / ell),
    "matern52": lambda r, ell: (1 + 5 ** 0.5 * r / ell + 5 * r ** 2 / (3 * ell ** 2)) * torch.exp(-(5 ** 0.5) * r / ell),
    "rbf": lambda r, ell: torch.exp(-0.5 * (r / ell) ** 2),
    "periodic": lambda r, ell: torch.exp(-2 * torch.sin(torch.pi * r) ** 2 / ell ** 2),      # period 1
}


class KernelGP:
    """A GP with a named stationary kernel and NO knowledge of the bank. Amplitude and lengthscale are
    fitted by marginal likelihood on a grid; then sample the prior or condition on observations exactly.
    Works in value space on arbitrary points (1D or 2D), so it needs no basis."""

    def __init__(self, kernel="matern32", ells=None, amps=(0.25, 0.5, 1.0, 2.0, 4.0)):
        self.kernel, self.name = KERNELS[kernel], kernel
        self.ells = torch.logspace(-2, 0, 13, dtype=DTYPE) if ells is None else torch.as_tensor(ells, dtype=DTYPE)
        self.amps = torch.as_tensor(amps, dtype=DTYPE)
        self.amp, self.ell, self.mean = 1.0, 0.1, 0.0

    def _gram(self, a, b, ell):
        return self.kernel(torch.cdist(a, b), ell)

    @torch.no_grad()
    def fit(self, points, values, noise_std):
        """values [k, n] on points [n, d]: maximise the summed marginal likelihood over the k functions."""
        self.mean = values.mean().item()
        y = (values - self.mean).T                                        # [n, k]
        best = (-float("inf"), None)
        for ell in self.ells:
            base = self._gram(points, points, ell)
            for amp in self.amps:
                K = amp ** 2 * base + (noise_std ** 2 + 1e-8) * torch.eye(len(points), dtype=DTYPE)
                L, info = torch.linalg.cholesky_ex(K)
                if info.item():
                    continue
                alpha = torch.cholesky_solve(y, L)
                logml = -0.5 * (y * alpha).sum() - y.shape[1] * L.diagonal().log().sum()
                if logml > best[0]:
                    best = (logml.item(), (amp.item(), ell.item()))
        self.amp, self.ell = best[1]
        return self

    @torch.no_grad()
    def sample_prior(self, points, num_draws):
        K = self.amp ** 2 * self._gram(points, points, self.ell) + 1e-8 * torch.eye(len(points), dtype=DTYPE)
        return self.mean + torch.randn(num_draws, len(points), dtype=DTYPE) @ torch.linalg.cholesky(K).T

    @torch.no_grad()
    def posterior(self, points_obs, values, noise_std, points_pred, num_draws):
        """Exact conditioning: draws [num_draws, len(points_pred)]."""
        K = self.amp ** 2 * self._gram(points_obs, points_obs, self.ell) + noise_std ** 2 * torch.eye(len(points_obs), dtype=DTYPE)
        L = torch.linalg.cholesky(K)
        Kps = self.amp ** 2 * self._gram(points_pred, points_obs, self.ell)
        mean = self.mean + Kps @ torch.cholesky_solve((values - self.mean)[:, None], L)[:, 0]
        cov = self.amp ** 2 * self._gram(points_pred, points_pred, self.ell) - Kps @ torch.cholesky_solve(Kps.T, L)
        cov = 0.5 * (cov + cov.T) + 1e-8 * torch.eye(len(points_pred), dtype=DTYPE)
        return mean + torch.randn(num_draws, len(points_pred), dtype=DTYPE) @ torch.linalg.cholesky(cov).T

    def label(self):
        return f"GP ({self.name}, ell={self.ell:.3f})"


# ------------------------------------------------------------------ metrics and reporting
@torch.no_grad()
def energy_distance(a, b, max_samples=800):
    """Energy distance between two sets of function values on a common grid (L2 metric). Zero iff
    the two measures agree."""
    a, b = a[:max_samples], b[:max_samples]
    scale = a.shape[-1] ** -0.5
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()).item() * scale


def quantiles(x, qs=(0.05, 0.5, 0.95)):
    return [round(v, 3) for v in torch.quantile(x, torch.tensor(qs, dtype=x.dtype)).tolist()]


def report(rows, header):
    width = max(len(name) for name, _ in rows)
    print("\n" + header)
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")


def score_2d(draws_values, truth_values, hidden):
    """rms of the posterior mean, calibration, and sharpness on the HIDDEN pixels.

    rms alone rewards a blurry mean, so all three are printed: a method that wins on rms while
    |z|<2 sits far from 0.95 is overconfident, and one that wins on rms with a much larger std is
    winning by hedging."""
    mean, std = draws_values.mean(0), draws_values.std(0).clamp(min=1e-6)
    z = ((truth_values - mean) / std)[hidden]
    return (f"rms {((mean - truth_values)[hidden] ** 2).mean().sqrt():.4f}   "
            f"|z|<2 {(z.abs() < 2).double().mean():.2f}   "
            f"std {std[hidden].mean():.4f}")


# ------------------------------------------------------------------ figures
def curves_figure(sets, points, path, sharey=False, zero_line=False, num=25):
    """One panel per named set of function values on 1D points."""
    fig, axes = plt.subplots(1, len(sets), figsize=(4 * len(sets), 3.2), sharey=sharey)
    for ax, (name, values) in zip(axes.flat, sets.items()):
        ax.plot(points, values[:num].T, lw=0.8, alpha=0.7)
        if zero_line:
            ax.axhline(0, color="k", lw=0.8)
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _show(ax, img, size, title, cmap, vmin, vmax):
    ax.imshow(img.reshape(size, size).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def image_row(images, size, path, titles=None, cmap="viridis", vmin=None, vmax=None):
    """A row of 2D fields given as flat vectors [n, size*size]."""
    fig, axes = plt.subplots(1, len(images), figsize=(2.6 * len(images), 2.8))
    for i, (ax, img) in enumerate(zip(axes.flat, images)):
        _show(ax, img, size, titles[i] if titles else "", cmap, vmin, vmax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def inpainting_figure(truth, observed, methods, size, path, obs_points=None, cmap="viridis", vmin=None, vmax=None):
    """Row 0: truth | observed. Row per method: mean | std | two draws.
    truth/observed are flat [size*size] (observed may hold NaN on hidden pixels); methods = {name: draws
    [n, size*size]}; obs_points [k, 2] in [0,1]^2 overplots scattered observations."""
    rows = 1 + len(methods)
    fig, axes = plt.subplots(rows, 4, figsize=(11, 2.7 * rows))
    _show(axes[0, 0], truth, size, "truth", cmap, vmin, vmax)
    _show(axes[0, 1], observed, size, "observed", cmap, vmin, vmax)
    if obs_points is not None:
        axes[0, 1].scatter(obs_points[:, 0] * size - 0.5, obs_points[:, 1] * size - 0.5, s=3, c="r")
    for ax in axes[0, 2:]:
        ax.axis("off")
    for row, (name, draws) in enumerate(methods.items(), start=1):
        _show(axes[row, 0], draws.mean(0), size, f"{name}: mean", cmap, vmin, vmax)
        _show(axes[row, 1], draws.std(0), size, "std", "magma", 0, None)
        _show(axes[row, 2], draws[0], size, "draw", cmap, vmin, vmax)
        _show(axes[row, 3], draws[1], size, "draw", cmap, vmin, vmax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
