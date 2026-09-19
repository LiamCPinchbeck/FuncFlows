# Exact traces

## The thing you have to pay

Pushing a measure along an ODE costs you a divergence:

$$ \log \frac{\mathrm{d}q}{\mathrm{d}\mu_0}(v_T)
   \;=\; \text{(Gaussian exponent difference)}
   \;-\; \int_0^T \operatorname{Tr} \frac{\partial h}{\partial v}\, \mathrm{d}t. $$

For a general neural velocity field, computing $\operatorname{Tr}\,\partial h/\partial v$
exactly costs $M$ backward passes. Nobody does that. The standard move is the Hutchinson
estimator ({ref}`FFJORD; Grathwohl et al., 2019 <ref-ffjord>`): $\operatorname{Tr} J = \mathbb{E}[\epsilon^\top J
\epsilon]$ for any probe with unit covariance. One backward pass per probe, unbiased, noisy.

This package goes the other way and restricts the field until the trace falls out of the
forward pass you were doing anyway.

## How each field pulls it off

**`LinearField`** — $h = g(t) \odot v + \text{drift}(t,c)$. Jacobian is
$\operatorname{diag}(g)$, trace is $\sum_k g_k(t)$. The drift doesn't depend on $v$ so it
contributes nothing. Almost embarrassingly easy.

**`MatrixField`** — $h = B(t)\,\tanh(A(t) v + b(t,c))$, which is a Sylvester flow layer
({ref}`van den Berg et al., 2018 <ref-sylvester>`). Jacobian is $B D A$ with $D = \operatorname{diag}(1 - \tanh^2)$, so

$$ \operatorname{Tr} = \sum_{j} d_j\, (AB)_{jj}, $$

and every ingredient is already sitting there from the forward pass.

**`PointwiseField`** — one FNO-style layer ({ref}`Li et al., 2021 <ref-fno>`). Because no gate depends on $v$,
the grid map is applied point by point, the Jacobian on the grid is *diagonal*, and

$$ \operatorname{Tr} = w \sum_{p} d_p \sum_{j} \Phi_{pj}^2 $$

with $w$ the cell volume. The `squared_sum` buffer is that inner sum, precomputed once.

**`SumField`** — velocities add, so traces add. That's it.

## What exactness actually buys you

| What you want to do | What you need |
|---|---|
| flow matching, sampling, TARP coverage | nothing. no trace at all |
| `ReverseKL`, `NegativeLogL` training | a trace; estimated is survivable |
| `ImportanceCorrection` ({ref}`Dax et al., 2023 <ref-dax>`) | exact |
| `latent_pcn` on a posterior flow | exact, and not negotiable |

```{danger}
An unbiased trace is **not** an unbiased density.

Hutchinson noise lands in $\log \mathrm{d}q/\mathrm{d}\mu_0$. Then importance weights
exponentiate it, and $\mathbb{E}[e^{\varepsilon}] \neq e^{\mathbb{E}[\varepsilon]}$. Your
efficiency and log-evidence come out biased *upward* — not noisy, but actually biased. 
You get a number that looks better than the truth, which is "not good".

So: either you need the density and it has to be exact, or you don't and you should skip the
trace entirely and save the backward passes. Hutchinson only earns its keep when you need a
training signal you can't get any other way.
```

## What it costs

As far as I know, the restriction that one wants an exact trace is that no gate may depend on $v$. 
That rules out just bolting a general neural network on as the velocity. It's also why `OperatorField` — which
stacks layers and mixes between them — has an estimated trace.

## Grid resolution (the one that will bite you)

`PointwiseField` evaluates on a grid of size $G$. A tanh generates harmonics: the cubic term
reaches $3k_{\max}$, and then Nyquist doubles it, so

$$ G \;\geq\; 6\, k_{\max} $$

or the projection back aliases and your "exact" trace is wrong. Note this has nothing to do
with whatever resolution you happen to store your images at — set it from the basis
truncation. Tying `grid_size` to the image size is an easy mistake and it fails silently,
which is the worst kind.
