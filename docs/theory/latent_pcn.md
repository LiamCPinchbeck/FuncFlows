# Latent pCN

Posterior sampling inside the latent space of a trained flow. Condensed version; notebook
`05` runs the thing.

## Why not just do a random walk

You've got $\pi(\mathrm{d}v) \propto e^{-\Phi(v)} \mu_0(\mathrm{d}v)$. Obvious first
attempt: propose $v' = v + \beta\xi$ with $\xi \sim \mu_0$. The acceptance ratio picks up

$$ -\tfrac12\big( \|v'\|^2 - \|v\|^2 \big)
   = -\beta \langle v, \xi\rangle - \tfrac12 \beta^2 \|\xi\|^2 . $$

Under $\mu_0$, $\|\xi\|^2$ is a sum of $M$ squared standard normals, so it concentrates
hard at $M$. Which means the exponent carries a systematic $-\tfrac12\beta^2 M$: acceptance
dies like $e^{-\beta^2 M/2}$ unless $\beta \sim M^{-1/2}$, and then you need $O(M)$ steps to
go anywhere.

At $M = 440$ that's painful. At $M = 1023$ forget it. And — this is the bit that should annoy
you — it gets *worse* every time you refine the basis, which is the one thing function-space
methods exist to let you do.

Notice where the damage came from: the **prior** term, not the likelihood. The proposal doesn't
respect $\mu_0$. In $M$ dimensions a Gaussian doesn't sit near its mode, it sits on a thin
shell at radius $\sqrt{M}$, and a step in a random direction almost always walks off it.

## pCN

Shrink first, then add noise:

$$ v' = \sqrt{1-\beta^2}\, v + \beta\, \xi, \qquad \xi \sim \mu_0 . $$

Variances add back to $C_0$, so the proposal leaves $\mu_0$ alone. Better than that: $(v, v')$
is jointly Gaussian with a covariance that's symmetric under swapping the blocks, so the pair
is exchangeable, so it's *reversible* with respect to $\mu_0$. Which kills the prior term
stone dead:

$$ \alpha(v,v') = \min\{1,\ \exp(\Phi(v) - \Phi(v'))\}. $$

No $M$ anywhere in there. This is {ref}`Cotter et al. (2013) <ref-pcn>`.

```{note}
The constraint is on the **reference measure**, not the prior. People conflate these and then
conclude they're stuck with Gaussian priors, which isn't true.

Gaussian is the only finite-variance family closed under that shrink-and-add operation
($\alpha$-stable families work too, with rebalanced coefficients). But an arbitrary prior
$\rho$ costs you nothing algorithmically — it goes into the potential as
$-\log \mathrm{d}\rho/\mathrm{d}\mu_0$, or you absorb it into the transport, which is
precisely what a learned prior flow *is*.
```

## Now put it in latent space

pCN's weakness is that it assumes the target looks like the reference. A trained flow makes
that true, so: run the chain on $z$, report $v = T(z)$.

Demanding that the reported samples actually be $\pi$ pins down the potential completely:

$$ \Psi(z) \;=\; \Phi(Tz)
   \;+\; \log \frac{\mathrm{d}q}{\mathrm{d}\mu_0}(Tz)
   \;-\; \log \frac{\mathrm{d}\rho}{\mathrm{d}\mu_0}(Tz),
   \qquad q = T_{\#}\mu_0 . $$

Two cases, and they are **not** the same call:

:::{list-table}
:header-rows: 1
:widths: 45 55

* - Your situation
  - What you pass as `potential`
* - `T` is a learned prior, so $\rho = q$
  - `misfit`
* - `T` approximates the posterior, $\rho = \mu_0$
  - `lambda v: misfit(v) + flow.log_rn_at(v)`
:::

In the first, the flow's Jacobian cancels out entirely, so any trained prior flow works and the
trace doesn't even need to be computable. In the second the trace shows up in the acceptance
ratio.

## When it falls over

If the likelihood is way sharper than the flow's pushforward, the adapted $\beta$ collapses and
the chains just sit on their initialisation. Quietly — the output still looks like a posterior,
which is how it got past us for a while. `info["moved"]` reports how far the chains actually
travelled relative to where they started; under about 0.1 and you're looking at your own
`init`, not a posterior. When that happens, train an amortised conditional flow instead
(notebook `04`) rather than throwing more steps at it.

## References

- {ref}`Cotter et al. (2013) <ref-pcn>` — pCN itself, and the dimension-robustness.
- {ref}`Parno & Marzouk (2018) <ref-parno>` — running a sampler in a learned transport's latent space.
- {ref}`Hoffman et al. (2019) <ref-neutra>` — NeuTra, same idea with HMC instead of pCN.
- {ref}`arXiv:2310.17817 <ref-saroundtrip>` — the nearest published version.

Full entries on the {doc}`../references` page.

"Latent pCN" is our name for it, not the literature's. It's a composition of the first two.
