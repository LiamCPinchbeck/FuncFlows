# FuncyFlows

Normalizing flows and flow matching, but on **function spaces**. 

```{warning}
I've (Liam) written most of the code, but I got Claude to make the documentation. 
I've checked that it all seems reasonable, but if something doesn't make 100% sense/work, then raise a GitHub issue
or just email me.
```

The idea in one breath: a function is its coefficient vector on a Laplacian eigenbasis, the
reference measure is a Gaussian on those coefficients, and a transport is a neural ODE that
moves them around. The vector fields are built so their divergence comes out in closed form,
which means the density of the transported measure is *exact* — no Hutchinson probes. That one
property is what makes reverse-KL training, likelihood training, importance reweighting and
exact latent-space MCMC all still work at hundreds of modes, instead of quietly falling apart.

```{note}
The notebooks here are tiny on purpose. Twenty-odd modes, a couple thousand training steps,
done in a minute or two on a laptop. They're here to show you the shape of the API, not to
produce a good result — if you want results, the full-size studies are in
`FuncyFlows/examples/` and you get them with `python -m FuncyFlows.examples`.
```

## Where to start

If you've never touched this: read `01`, then `02`. That's the whole representation and how you
fit a prior to data. Everything else builds on those two.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
notebooks/01_bases_and_measures
notebooks/02_learn_a_prior
```

```{toctree}
:maxdepth: 1
:caption: Actually doing inference

notebooks/03_posterior_from_a_likelihood
notebooks/04_amortised_posterior
notebooks/05_latent_pcn_and_diagnostics
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/index
```

```{toctree}
:maxdepth: 2
:caption: API

api/index
```

```{toctree}
:maxdepth: 1
:caption: Where this comes from

references
```

## The parts list

| Piece | What it's for |
|---|---|
| `CosineBasis(M)`, `FourierBasis(M)` | `M` orthonormal basis functions on `[0,1]^d` |
| `GaussianReferenceMeasure` | the base measure `N(0, diag(sigma^2))` on coefficients |
| `LinearField` | per-mode gain, plus a drift that can depend on data |
| `MatrixField` | one wide tanh layer across all coefficients |
| `PointwiseField` | one FNO-ish spatial layer, trace still exact |
| `SumField` | adds fields together. Traces add too, which is the whole trick |
| `ContinuousTransformation` | the flow itself: `transport`, `push_forward`, `pull_back`, `log_rn_at` |
| `FlowMatching`, `ConditionalFlowMatching`, `ReverseKL`, `NegativeLogL` | things to minimise |
| `latent_pcn` | exact MCMC, run inside the flow's latent space |
| `ImportanceCorrection`, `coverage_curve` | ways of finding out you were wrong |

## Indices

- {ref}`genindex`
- {ref}`modindex`
