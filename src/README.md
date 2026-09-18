# FuncFlows

Normalizing flows and flow matching on function spaces.

A function is represented by its coefficients on a Laplacian eigenbasis (`CosineBasis` or
`FourierBasis`), the reference measure is a Gaussian on those coefficients, and a transport is a
neural ODE in coefficient space. The vector fields (`LinearField`, `MatrixField`) have closed-form
divergences, so the density of the transported measure relative to the
reference measure is exact — no Hutchinson estimators — which makes reverse-KL training,
likelihood training and importance reweighting usable at hundreds of modes.

```bash
pip install funcflows            # torch + tqdm
```

`torch` is large; if you need a specific CUDA/MPS build install it first from pytorch.org.

`examples/quickstart.py` runs both patterns below end to end in a couple of minutes on CPU.

## The pieces

| Piece | Role |
|---|---|
| `CosineBasis(M)`, `FourierBasis(M)` | `M` orthonormal basis functions on [0, 1]; `basis.evaluate(points)` → design matrix |
| `GaussianReferenceMeasure(basis, alpha, power)` or `(basis, variances=...)` | base measure N(0, diag σ²) on the coefficients; `.sample(n)`, `.scale` (= σ) |
| `LinearField` | per-mode gain plus a data-dependent drift; exact for the linear-Gaussian part |
| `MatrixField(conditioner)` | one wide tanh layer over all coefficients; global nonlinear mixing |
| `SumField(*fields)` | add fields; traces add |
| `ContinuousTransformation(measure, field, num_steps)` | the flow: `.transport(v0)`, `.push_forward`, `.pull_back`, `.log_rn_at(v)` |
| `FlowMatching`, `ConditionalFlowMatching`, `ReverseKL`, `NegativeLogL` | objectives; each is a callable returning a loss |
| `train(objective, params, num_steps, learning_rate)` | Adam loop, returns the loss history |
| `ImportanceCorrection`, `coverage_curve`, `latent_pcn` | diagnostics and an exact latent-space MCMC sampler |

Every field takes `mode_scale=measure.scale`: the nonlinearity sees whitened coefficients and the
divergence is unchanged (similarity transform). Always pass it.

## 1. Learn a prior from samples (flow matching)

```python
import torch
from FuncFlows.base_measures import CosineBasis, GaussianReferenceMeasure
from FuncFlows.transports.continuous import (ContinuousTransformation, SumField, LinearField,
                                             MatrixField, TimeBasisConditioner)
from FuncFlows.objectives import FlowMatching
from FuncFlows.utils.train import train

M = 64
basis = CosineBasis(M)
coeffs = ...                                  # [N, M] training functions projected onto the basis
measure = GaussianReferenceMeasure(basis, variances=coeffs.var(0))   # diagonal Gaussian fit = the GP baseline

field = SumField(
    LinearField(M, num_time_modes=4),
    MatrixField(TimeBasisConditioner(M, 256, num_time_modes=4), mode_scale=measure.scale))
flow = ContinuousTransformation(measure, field, num_steps=16)

losses = train(FlowMatching(flow, coeffs, batch_size=256, weights=1 / measure.scale, coupling="optimal"),
               flow.parameters(), num_steps=10000, learning_rate=2e-3)

draws = flow.transport(measure.sample(1000))              # [1000, M] new functions
points = torch.linspace(0, 1, 400).reshape(-1, 1)
values = draws @ basis.evaluate(points).T                 # [1000, 400] on a grid
log_q = flow.log_rn_at(draws)                             # exact log density w.r.t. the reference measure
```

`weights=1/measure.scale` whitens the loss so high modes count; `coupling="optimal"` needs
scipy and only makes sense for an unconditional flow.

## 2. Posterior from a likelihood (no samples needed)

```python
from FuncFlows.objectives import ReverseKL
from FuncFlows.utils.gaussian_misfit import GaussianMisfit
from FuncFlows.diagnostics import ImportanceCorrection

prior = GaussianReferenceMeasure(basis, alpha=0.05, power=2.0)     # Matérn-like spectrum
phi_obs = basis.evaluate(obs_points)                               # [n_obs, M]
forward = lambda v: (v @ phi_obs.T) ** 2                           # any differentiable map
potential = GaussianMisfit(forward, data, noise_std)               # -log likelihood, [batch] -> [batch]

flow = ContinuousTransformation(prior, field, num_steps=16)        # same field recipe as above
train(ReverseKL(flow, potential, num_samples=64, path_gradient=True), flow.parameters(), num_steps=3000)

draws = flow.transport(prior.sample(2000))
check = ImportanceCorrection(flow, draws, potential)               # exact because the traces are exact
print(check.efficiency, check.log_evidence)                        # efficiency near 1 = posterior matched
```

For an exact sampler on top of a learned prior use `latent_pcn(flow, potential)`: pCN in the flow's
latent space, whose acceptance needs no Jacobian.

## 3. Amortised posterior (condition on data)

```python
from FuncFlows.transports.continuous import DataConditioner
from FuncFlows.objectives import ConditionalFlowMatching

C = ...                                                    # context dimension (e.g. whitened projected data)
field = SumField(
    LinearField(M, C, num_time_modes=4, mode_scale=measure.scale),          # data drift: the linear-Gaussian mean
    MatrixField(DataConditioner(M, 256, C, num_time_modes=4), mode_scale=measure.scale))
flow = ContinuousTransformation(measure, field, num_steps=16)

def simulate(batch):                                       # -> (target coeffs [batch, M], context [batch, C])
    ...

train(ConditionalFlowMatching(flow, simulate, measure.sample, batch_size=256, weights=1 / measure.scale),
      flow.parameters(), num_steps=20000, learning_rate=2e-3)

posterior = flow.transport(measure.sample(1000), context.expand(1000, -1))   # one ODE solve per posterior
```

Whiten the context by `measure.scale` when it is itself a projected function; otherwise the drift
weights for high modes have to reach 1/σ and never do. When a conjugate (linear-Gaussian)
posterior is available in closed form, use it as the base measure and let the flow learn the residual.

## Diagnostics

```python
from FuncFlows.diagnostics import coverage_curve, coverage_error
levels, coverage = coverage_curve(samples, truths, weights=1 / measure.scale)   # TARP; samples [cases, draws, M]
coverage_error(levels, coverage)     # signed max gap from the diagonal: negative = overconfident
```

## Layout

```
FuncFlows/
  base_measures/     bases, GaussianReferenceMeasure
  transports/        continuous/: fields, conditioners, ContinuousTransformation
                     (grid_fields.py: PointwiseField / OperatorField, spatial layers — experimental)
  objectives/        flow_matching, reverse_kl, negative_logl
  samplers/          latent_pcn
  diagnostics/       importance, coverage
  utils/             train, gaussian_misfit
```



# Full Example


```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from FuncFlows.base_measures import CosineBasis, GaussianReferenceMeasure
from FuncFlows.transports.continuous import (ContinuousTransformation, SumField, LinearField,
                                             MatrixField, TimeBasisConditioner)
from FuncFlows.objectives import ReverseKL, FlowMatching
from FuncFlows.samplers import latent_pcn
from FuncFlows.utils.train import train
from FuncFlows.utils.gaussian_misfit import GaussianMisfit
from FuncFlows.diagnostics import ImportanceCorrection

torch.manual_seed(1)
M, NUM_OBS, NOISE, N = 32, 15, 0.05, 2000
basis = CosineBasis(M)
prior = GaussianReferenceMeasure(basis, alpha=0.05, power=2.0)
x = torch.linspace(0, 1, 400, dtype=torch.float64)[:, None]
design = basis.evaluate(x)

# data ||  y = f(x)^2 + noise at NUM_OBS random points
truth = prior.sample(1)
obs_x = torch.rand(NUM_OBS, 1, dtype=torch.float64)
phi_obs = basis.evaluate(obs_x)
forward = lambda v: (v @ phi_obs.T) ** 2
data = forward(truth)[0] + NOISE * torch.randn(NUM_OBS, dtype=torch.float64)
misfit = GaussianMisfit(forward, data, NOISE) # -logl


def make_flow():
    field = SumField(LinearField(M, num_time_modes=4),
                     MatrixField(TimeBasisConditioner(M, 256, num_time_modes=4), mode_scale=prior.scale))
    return ContinuousTransformation(prior, field, num_steps=16)


@torch.no_grad()
def pcn(potential, num_chains=64, num_steps=20000, beta=0.15, burn=5000, thin=20):
    state, keep = prior.sample(num_chains), []
    energy = potential(state)
    for step in range(num_steps):
        proposal = (1 - beta ** 2) ** 0.5 * state + beta * prior.sample(num_chains)
        proposal_energy = potential(proposal)
        accept = torch.log(torch.rand_like(energy)) < energy - proposal_energy
        state = torch.where(accept[:, None], proposal, state)
        energy = torch.where(accept, proposal_energy, energy)
        if step >= burn and step % thin == 0:
            keep.append(state.clone())
    return torch.cat(keep)


def laplace_gp():
    white = torch.zeros(M, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([white], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = misfit(white * prior.scale) + 0.5 * (white ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        jac = 2 * (phi_obs @ (white * prior.scale))[:, None] * phi_obs * prior.scale
        cov = torch.linalg.inv(torch.eye(M, dtype=torch.float64) + jac.T @ jac / NOISE ** 2)
        return (white + torch.randn(N, M, dtype=torch.float64) @ torch.linalg.cholesky(cov).T) * prior.scale


# run everything
results = {}
results["pCN reference"] = pcn(misfit)[torch.randperm(64 * 750)[:N]]         # thin to N for the comparisons

results["GP (Laplace)"] = laplace_gp()

rkl = make_flow()
train(ReverseKL(rkl, misfit, num_samples=64, path_gradient=True), rkl.parameters(), num_steps=2500, learning_rate=3e-3)
with torch.no_grad():
    results["reverse KL"] = rkl.transport(prior.sample(N))

fm = make_flow()
train(FlowMatching(fm, results["pCN reference"], batch_size=256), fm.parameters(), num_steps=4000, learning_rate=3e-3)
with torch.no_grad():
    results["flow matching"] = fm.transport(prior.sample(N))

# pCN in the latent space of the posterior flow: target exp(-Phi(Tz)) prior(Tz)/q(Tz) mu0(dz), so add the
# flow's exact log-density to the potential. A good flow makes the target nearly flat -> big steps accepted.
corrected = lambda v: misfit(v) + fm.log_rn_at(v)                         # noqa: E731
draws, info = latent_pcn(fm, corrected, num_chains=128, num_steps=1500, beta=0.5, thin=10)
results["latent pCN"] = draws[torch.randperm(len(draws))[:N]]



# report
ref = results["pCN reference"] @ design.T
print(f"\n{'method':16s} {'+sign':>6s} {'E-dist':>8s}   (ideal 0.50; distance to pCN)")
for name, v in results.items():
    values = v @ design.T
    sign = ((v @ truth[0]) > 0).double().mean().item()
    dist = (2 * torch.cdist(values, ref).mean() - torch.cdist(values, values).mean() - torch.cdist(ref, ref).mean()) / 20
    print(f"{name:16s} {sign:6.2f} {dist.item():8.4f}")
for name, flow in [("reverse KL", rkl), ("flow matching", fm)]:
    with torch.no_grad():
        c = ImportanceCorrection(flow, results[name], misfit)
    print(f"{name}: importance efficiency {c.efficiency:.3f}, log evidence {c.log_evidence:.2f}")
print(f"latent pCN acceptance {info['acceptance']:.2f} at beta 0.5 (plain pCN would be ~0)")


# figures
fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.2), sharey=True)
tv = (truth @ design.T)[0]
for ax, (name, v) in zip(axes, results.items()):
    ax.plot(x[:, 0], (v[:40] @ design.T).T, color="C0", alpha=0.15, lw=1)
    ax.plot(x[:, 0], tv, "k", lw=1.5)
    ax.plot(x[:, 0], -tv, "k--", lw=1.5)
    ax.scatter(obs_x[:, 0], data.clamp(min=0).sqrt(), color="C3", s=12, zorder=3)
    ax.scatter(obs_x[:, 0], -data.clamp(min=0).sqrt(), color="C3", s=12, zorder=3)
    ax.set_title(name, fontsize=10)
fig.suptitle("y = f(x)^2 + noise: black = ±truth, red = ±sqrt(data), blue = posterior draws", fontsize=10)
fig.tight_layout()
fig.savefig("bimodal_posterior.png", dpi=130)
print("saved bimodal_posterior.png")
```