"""Bimodal posterior: observe y = f(x)^2 + noise. The posterior is symmetric under f -> -f, so the
question for every method is: does it find both modes (+sign 0.50) or only one?

  pCN             MCMC reference
  GP (Laplace)    Gaussian at the MAP -- one mode by construction
  reverse KL      flow trained from the potential; mode-seeking
  flow matching   flow trained on the pCN samples
  latent pCN      exact MCMC in the flow's latent space, using the flow's exact log-density

    python bimodal_posterior.py      (~5 min CPU) -> bimodal_posterior.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from FuncyFlows.base_measures import CosineBasis, GaussianReferenceMeasure
from FuncyFlows.transports.continuous import (ContinuousTransformation, SumField, LinearField,
                                             MatrixField, TimeBasisConditioner)
from FuncyFlows.objectives import ReverseKL, FlowMatching
from FuncyFlows.samplers import latent_pcn
from FuncyFlows.utils.train import train
from FuncyFlows.utils.gaussian_misfit import GaussianMisfit
from FuncyFlows.diagnostics import ImportanceCorrection

torch.manual_seed(1)
M, NUM_OBS, NOISE, N = 32, 15, 0.05, 2000
basis = CosineBasis(M)
prior = GaussianReferenceMeasure(basis, alpha=0.05, power=2.0)
x = torch.linspace(0, 1, 400, dtype=torch.float64)[:, None]
design = basis.evaluate(x)

# ---- data: y = f(x)^2 + noise at NUM_OBS random points
truth = prior.sample(1)
obs_x = torch.rand(NUM_OBS, 1, dtype=torch.float64)
phi_obs = basis.evaluate(obs_x)
forward = lambda v: (v @ phi_obs.T) ** 2                                  # noqa: E731
data = forward(truth)[0] + NOISE * torch.randn(NUM_OBS, dtype=torch.float64)
misfit = GaussianMisfit(forward, data, NOISE)                             # -log likelihood


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


# ---- run everything
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

# ---- report
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

# ---- figure
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
