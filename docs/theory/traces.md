# Traces

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


## Where "exact" stops

Everything above is about the integrand. $\operatorname{Tr}\,\partial h/\partial v$ really is
exact — a closed form, no estimator, no discretisation.

The integral is a different matter. `ContinuousTransformation` accumulates

$$ \int_0^T \operatorname{Tr} \frac{\partial h}{\partial v}\,\mathrm{d}t $$

with the same RK4 weights that advance the state, so it is fourth-order accurate rather than
exact. There are two gaps, both $O(h^4)$: against the continuum log-determinant, and against the
log-determinant of the discrete map actually applied — the second matters because that map is what
`transport` samples from. On a small nonlinear test field the two gaps fall by a factor of
$\approx 16$ for every halving of the step, as they should.

So the honest summary is: **exact integrand, approximated integral**. If you need the number for
your own model, sweep `num_steps` and compare `flow._map(v)[1]` against
`torch.linalg.slogdet` of the Jacobian of `lambda v: flow._map(v)[0]`.

## Grid resolution 

`PointwiseField` evaluates on a grid of size $G$. A tanh generates harmonics: the cubic term
reaches $3k_{\max}$, and then Nyquist doubles it, so

$$ G \;\geq\; 6\, k_{\max} $$

or the projection back aliases. Note this has nothing to do with whatever resolution you happen to store 
your images at — set it from the basis truncation. Tying `grid_size` to the image size is an easy mistake 
and it fails silently, which is not great.
