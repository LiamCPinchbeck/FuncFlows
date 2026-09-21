# Functions, coefficients, and why it's a measure

## Functions become vectors

A function $f$ on $[0,1]^d$ gets held as its coefficients $v \in \mathbb{R}^M$ on a Laplacian
eigenbasis $\{\phi_k\}$:

$$ f(x) \;=\; \sum_{k=1}^{M} v_k\, \phi_k(x), \qquad
   v_k \;=\; \int_{[0,1]^d} f(x)\, \phi_k(x)\, \mathrm{d}x. $$

`CosineBasis` gives you Neumann eigenfunctions, `FourierBasis` periodic ones. Both are
orthonormal in $L^2$, both order their columns by eigenvalue (so truncating keeps the smooth
stuff), and both hand you `laplacian_eigenvalues`, which is what every smoothness knob in the
package is actually built out of.

From there it's all finite-dimensional linear algebra on $v$. So what makes this a *function
space* method rather than just "a flow on a big vector"? Nothing in the construction gets
worse as $M$ grows. That's the whole claim, and it's worth being suspicious of it until you've
read the pCN page, where it's made precise.

## The reference measure

$$ \mu_0 = \mathcal{N}(0, C_0), \qquad
   C_0 = \operatorname{diag}(\sigma_1^2, \dots, \sigma_M^2). $$

`GaussianReferenceMeasure` builds $C_0$ one of two ways. Either from a smoothness prior,

$$ \sigma_k^2 = (1 + \alpha \lambda_k)^{-p}, $$

where bigger $p$ means smoother draws, or you just hand it the variances directly — which is
what you do when you've got a bank of example functions and want the base measure to match
them.

```{warning}
The variances have to be trace-class: $\sum_k \sigma_k^2 < \infty$. White noise (all
$\sigma_k$ equal) is **not** a legal function-space reference, which is why the parametric
form always decays in $k$. {ref}`Kerrigan et al. (2023) <ref-ffm>` make the same point in the
Functional Flow Matching paper, and it's one of the few places where the infinite-dimensional
bookkeeping bites you in practice rather than in theory.
```

## FYI there's no "density"

In infinite dimensions there is no Lebesgue measure to take a density against. In essence:
suppose you had a translation-invariant measure where every ball has finite nonzero size. In
infinite dimensions the unit ball contains infinitely many disjoint balls of radius $1/4$ —
which has no finite-dimensional analogue... They all have the same size by translation-invariance, 
so the unit ball has infinite size. Contradiction, no such measure, no densities.

What you get instead is the **Radon–Nikodym derivative**, which is just the ratio of two
measures:

$$ \nu(\mathrm{d}x) \;=\; \frac{\mathrm{d}\nu}{\mathrm{d}\mu}(x)\, \mu(\mathrm{d}x). $$

Read it as "how much more likely $\nu$ thinks $x$ is than $\mu$ does". Every density in this
package is one of these, always against $\mu_0$, and `flow.log_rn_at(v)` is the log of the one
you want.

You might reasonably ask why we bother, given the code truncates to $M$ modes and everything's
finite again. Because the question that decides whether an algorithm is any good here is *what
happens as $M$ grows*, and the infinite-dimensional bookkeeping answers it before you've run
anything.

## The transport

A flow is the time-$T$ solution of

$$ \frac{\mathrm{d}v}{\mathrm{d}t} = h(v, t, c), $$

integrated with RK4 ({ref}`Grathwohl et al., 2019 <ref-ffjord>` for the finite-dimensional
version of what follows). Its pushforward has the RN derivative:

$$ \log \frac{\mathrm{d}q}{\mathrm{d}\mu_0}(v_T)
   \;=\; \big(\text{Gaussian exponent at } v_0 \text{ minus at } v_T\big)
   \;-\; \int_0^T \operatorname{Tr} \frac{\partial h}{\partial v}\, \mathrm{d}t. $$

That integral is the expensive part, and it's why the next page exists.

## Whitening

Every field takes `mode_scale=measure.scale`. The nonlinearity then sees whitened coefficients
$S^{-1}v$ and the output gets scaled back by $S$. Since that's a similarity transform the
divergence doesn't change, so it costs you nothing.

Skip it and the high modes — whose $\sigma_k$ might be $10^{-3}$ or worse — are effectively
invisible to a tanh, because the drift weights would have to reach $1/\sigma_k$ to matter and
they never get there. Always pass it. This has cost more debugging hours than it has any right
to.
