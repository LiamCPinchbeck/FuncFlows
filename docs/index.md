# FuncyFlows

Normalizing flows and flow matching, but on **function spaces**. 

```{image} logo_flow.gif
:alt: a flow turning Gaussian noise into the FuncyFlows logo
:width: 340px
:align: center
```


The idea: a function is its coefficient vector on a Laplacian eigenbasis, the
reference measure is a Gaussian on those coefficients, and a transport is a neural ODE that
moves them around. 

```{warning}
I (Liam) have written most of the code, but I got Claude to make the documentation. 
I've checked that it all seems reasonable and edited bits and pieces, but if something doesn't make 100% sense/work, 
then raise a GitHub issue or just email me.
```

```{note}
The notebooks here should be small such that they could run on anything reasonable (laptop upwards). 
~Twenty modes, a couple thousand training steps, done in a minute or two on a laptop. Just there to
show the skeleton of the API, for 'nice' results you'll typically need more training.
Some full-size studies are in `FuncyFlows/examples/` and you get them with `python -m FuncyFlows.examples`.
```

## Where to start

If you've never touched this kind of thing before: I'd recommend reading [arXiv:2305.17209](https://arxiv.org/abs/2305.17209) for an intro and 
[arXiv:2411.13277](https://arxiv.org/abs/2411.13277) for a bit of a deep-dive. 
Includes the whole representation (both) and how you fit a prior to data (mostly the second). Otherwise the "Getting Started" should hopefully
be good enough for a working understanding.

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
| `PointwiseField` | one FNO-ish spatial layer |
| `SumField` | adds fields together, and traces naturally add. |
| `ContinuousTransformation` | the flow!: `transport`, `push_forward`, `pull_back`, `log_rn_at` |
| `FlowMatching`, `ConditionalFlowMatching`, `ReverseKL`, `NegativeLogL` | things to minimise |
| `latent_pcn` | MCMC, run inside the flow's latent space. Like this [arXiv:1412.5492](https://arxiv.org/abs/1412.5492)|
| `ImportanceCorrection`, `coverage_curve` | some diagnostics |

## Indices

- {ref}`genindex`
- {ref}`modindex`
