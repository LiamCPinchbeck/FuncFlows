import math, torch
from tqdm.auto import trange


def _as_coefficients(result):
    """push_forward / pull_back may return (coeffs, log_density); keep the coefficients."""
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


@torch.no_grad()
def latent_pcn(flow, potential, num_chains=64, num_steps=2000, beta=0.2, init=None, burn=None,
               thin=1, adapt_to=None, context=None, progress=True, generator=None):
    r"""Sample a posterior by running pCN in the latent space of `flow`. If unfamiliar with pCN,
    it's an MCMC method that makes proposals by scaling the current MCMC sample and adding
    an adjusted sample from the prior. Scales better with dimension than standard MCMC because
    of the implicit geometry added along with the prior samples.

    Parameters
    ----------
    flow : ContinuousTransformation
        Trained transport. Only ``base_measure.sample``, ``transport`` and (for ``init``)
        ``pull_back`` are used. Set ``flow.num_steps`` before calling if you want the ODE
        integrated more coarsely than it was trained -- sampling tolerates far fewer steps than
        training, and every chain step costs one solve.
    potential : callable
        ``potential(v) -> [batch]``, the NEGATIVE log of the target's density relative to the
        flow's pushforward, as a function of COEFFICIENTS. See explanation further down
         in this docstring for which of the two forms you want.
    num_chains : int
        Chains advanced in lockstep. They share one ``transport`` call per step, so more chains are
        nearly free up to the batch the ODE can hold.
    num_steps : int
        Steps per chain, including burn-in.
    beta : float
        Initial step size in (0, 1]. 1 proposes an independent draw from the base measure; small
        values propose small perturbations. Ignored after burn-in if ``adapt_to`` is set.
    init : Tensor [num_chains, M] or [k, M], optional
        Starting states as COEFFICIENTS (not latents); pulled back through the flow. Fewer rows
        than chains are tiled, more are truncated. With a sharp likelihood, starting at a cheap
        approximation -- the conjugate Gaussian posterior, say -- saves thousands of steps. Without
        it the chains start from the base measure.
    burn : int, optional
        Steps discarded, and the window in which ``beta`` adapts. Defaults to ``num_steps // 3``.
    thin : int
        Keep every ``thin``-th state after burn-in.
    adapt_to : float, optional
        Target acceptance rate. During burn-in ``beta`` is nudged towards it geometrically
        (Robbins-Monro on log beta). 0.2-0.3 is the usual choice. Adaptation stops at ``burn``, so
        the sampled part of the chain has a fixed kernel and remains exactly invariant.
    context : Tensor [context_dim] or [num_chains, context_dim], optional
        Conditioning for a conditional flow; broadcast across chains.
    progress : bool
        Show a tqdm bar.
    generator : torch.Generator, optional
        For reproducible chains. Only used for the uniform accept/reject draws; the base measure
        draws its own randomness.

    Returns
    -------
    draws : Tensor [num_kept * num_chains, M]
        Post-burn-in states in COEFFICIENT space, chains concatenated.
    info : dict
        ``acceptance``   mean acceptance after burn-in (the number that matters).
        ``acceptance_burn``  mean acceptance during burn-in, for checking the adaptation worked.
        ``beta``         final step size.
        ``potential``    mean potential over the kept states; a sanity check that it plateaued.
        ``moved``        mean distance travelled from the starting states, relative to their own
                         norm. Below ~0.1 the chain never left its initialisation and the "posterior"
                         you are looking at is whatever you passed as ``init``. Reported only when
                         ``init`` is given.


    Preconditioned Crank-Nicolson MCMC in the latent space of a trained flow.

    What we're doing
    ------------------------
    You have a posterior on coefficients,

        pi(dv)  proportional to  exp(-Phi(v)) rho(dv),

    with Phi the misfit (minus log likelihood) and rho a prior. Sampling it directly with a random-walk
    Metropolis proposal fails as the number of modes M grows: to keep the acceptance rate away from
    zero the step size has to shrink like M^(-1/2), so the chain needs O(M) steps to move anywhere.

    pCN (Cotter, Roberts, Stuart & White, Statistical Science 2013, arXiv:1202.0709) fixes that for a
    GAUSSIAN reference measure mu0. Its proposal is

        z' = sqrt(1 - beta^2) z + beta xi,        xi ~ mu0,

    which leaves mu0 exactly invariant -- if z ~ mu0 then (z, z') is jointly Gaussian and exchangeable,
    so the pair is reversible. Because the proposal already carries the reference measure, the
    Metropolis ratio keeps only the likelihood:

        accept with probability  min(1, exp(Phi(z) - Phi(z'))).

    No prior density appears, nothing scales with M, and beta can stay O(1) at any truncation.

    What the "latent" part adds
    ---------------------------
    pCN needs the reference measure to be Gaussian, because the Gaussian family is closed under this operation 
    and almost nothing else is: with v and xi independent N(0,C), the combination has covariance 
    
        (1-beta^2)C + beta^2 C = C, 
        
    and a mean-zero Gaussian is determined by its covariance, so v' is N(0,C) again.

    If the posterior is curved, multimodal or far from the prior, a Gaussian-geometry sampler does not do very well. 
    
    The constraint is on the REFERENCE measure, not the prior. An arbitrary prior rho enters through -log(drho/dmu0) 
    in the potential (formula above) with no change to the algorithm, or gets absorbed into the transport, which is what a learned prior flow is.
    What must be closed under the proposal is the reference: Gaussian is the only finite-variance choice, 
    though alpha-stable references work with rebalanced coefficients.

    But, we run pCN not on v but on the flow's latent variable z, where v = T(z) and T is the trained
    transport. The reference measure there *IS* Gaussian by construction -- it is the flow's base measure
    -- and the flow has already absorbed the awkward geometry. This is transport-map preconditioning
    (Parno & Marzouk, arXiv:1412.5492; the neural version, with HMC rather than pCN, is NeuTra,arXiv:1903.03704).



    Which potential to pass
    -----------------------
    The chain targets  exp(-Psi(z)) mu0(dz), so v = T(z) comes out distributed as

        Psi(z) = Phi(Tz) + log (dq/dmu0)(Tz) - log (drho/dmu0)(Tz),     q = T_# mu0

    Two cases cover the below, they are not the same call :grimace: :

      * T is a learned PRIOR, so rho = q and the log terms cancel:

            potential = misfit

        The flow's Jacobian never enters, so any trained prior flow works and the trace
        need not even be computable.

      * T approximates the POSTERIOR and the prior is the base measure, rho = mu0:

            potential = lambda v: misfit(v) + flow.log_rn_at(v)

        Here the trace DOES enter, and it unfortunately has to be exact. 
        A Hutchinson estimate doesn't give a noisy version of the 'correct' chain:
            - the randomness lands in the accept test
            - so the chain has a different invariant measure. 
    """
    if not 0 < beta <= 1:
        raise ValueError(f"beta must be in (0, 1]; got {beta}")
    burn = num_steps // 3 if burn is None else burn
    if burn >= num_steps:
        raise ValueError(f"burn ({burn}) must be less than num_steps ({num_steps})")

    measure = flow.base_measure
    if context is not None:
        context = context.reshape(1, -1).expand(num_chains, -1) if context.dim() == 1 else context

    def to_coefficients(latent):
        return _as_coefficients(flow.transport(latent) if context is None
                                else flow.transport(latent, context))

    # ---- starting states, in latent space
    if init is None:
        state = measure.sample(num_chains)
    else:
        if init.shape[0] < num_chains:
            init = init.repeat((num_chains + init.shape[0] - 1) // init.shape[0], 1)
        init = init[:num_chains]
        state = _as_coefficients(flow.pull_back(init) if context is None
                                 else flow.pull_back(init, context))

    current = to_coefficients(state)                      # kept in step with `state`, so a stored
    energy = potential(current)                           # sample never costs a second ODE solve
    reference = current.mean(0)
    start_norm = current.norm(dim=-1).mean().clamp(min=1e-12)

    kept, accepted_burn, accepted_keep, num_burn_steps = [], 0.0, 0.0, 0
    steps = trange(num_steps, desc="latent pCN") if progress else range(num_steps)

    for step in steps:
        proposal = (1 - beta ** 2) ** 0.5 * state + beta * measure.sample(num_chains)
        proposal_coeffs = to_coefficients(proposal)
        proposal_energy = potential(proposal_coeffs)

        threshold = torch.rand(num_chains, dtype=energy.dtype, device=energy.device,
                               generator=generator).log()
        accept = threshold < energy - proposal_energy
        state = torch.where(accept[:, None], proposal, state)
        current = torch.where(accept[:, None], proposal_coeffs, current)
        energy = torch.where(accept, proposal_energy, energy)
        rate = accept.double().mean().item()

        if step < burn:
            accepted_burn += rate
            num_burn_steps += 1
            if adapt_to is not None:
                # Robbins-Monro on log beta: damped by 1/(10 + step) so late burn-in barely moves
                # it, and stopped entirely at `burn` so the sampled kernel is fixed and invariant.
                beta = min(max(beta * math.exp((rate - adapt_to) / (10 + step)), 1e-4), 1.0)
        else:
            accepted_keep += rate
            if (step - burn) % thin == 0:
                kept.append(current.clone())

    if not kept:                                          # thin larger than the post-burn window
        kept.append(current.clone())

    draws = torch.cat(kept)
    info = {"acceptance": accepted_keep / max(num_steps - burn, 1),
            "acceptance_burn": accepted_burn / max(num_burn_steps, 1),
            "beta": beta,
            "potential": potential(draws).mean().item()}
    if init is not None:
        info["moved"] = (draws - reference).norm(dim=-1).mean().item() / start_norm.item()
    return draws, info
