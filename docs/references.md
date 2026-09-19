# References

Everything the docs lean on, with the in-text citations pointing here. Grouped by what it's for
rather than alphabetically, because that's how you'll actually look things up.

Every arXiv ID below was checked against arXiv rather than typed from memory — which sounds like a
low bar, except that it isn't, and a wrong identifier in a reference list is worse than no
reference at all.

## Flows and flow matching

(ref-lipman)=
**Lipman, Chen, Ben-Hamu, Nickel & Le (2023)** — *Flow Matching for Generative Modeling.*
ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
: The objective `FlowMatching` implements: regress a velocity field onto the straight-path
  velocity between the reference measure and the data. Simulation-free, which is the whole appeal.

(ref-tong)=
**Tong, Malkin, Huguet, Zhang, Rector-Brooks, Fatras, Wolf & Bengio (2024)** — *Improving and
Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.*
[arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
: Where `coupling="optimal"` comes from. Re-pairing a minibatch by optimal transport straightens
  the paths, which lowers the variance of the target and buys you fewer integration steps at
  sampling time.

(ref-ffjord)=
**Grathwohl, Chen, Bettencourt, Sutskever & Duvenaud (2019)** — *FFJORD: Free-Form Continuous
Dynamics for Scalable Reversible Generative Models.* ICLR 2019.
[arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
: The continuous-flow density formula, and the Hutchinson trace estimator that this package goes
  out of its way to avoid needing. It's also the correct citation for
  `estimated_velocity_and_trace`, which is the one place we do use it.

(ref-sylvester)=
**van den Berg, Hasenclever, Tomczak & Welling (2018)** — *Sylvester Normalizing Flows for
Variational Inference.* UAI 2018. [arXiv:1803.05649](https://arxiv.org/abs/1803.05649)
: `MatrixField` is a Sylvester layer. The reason its trace collapses to a sum over the hidden
  units rather than needing the full Jacobian.

## Function spaces

(ref-ffm)=
**Kerrigan, Migliorini & Smyth (2023)** — *Functional Flow Matching.*
[arXiv:2305.17209](https://arxiv.org/abs/2305.17209)
: Flow matching lifted to infinite dimensions. Source of the trace-class requirement on the
  reference covariance, and of the Feldman–Hájek / Cameron–Martin conditions that our truncation
  quietly satisfies. They use an FNO for the velocity and report no likelihoods — which, given the
  exact-traces page, is not a coincidence.

(ref-fno)=
**Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar (2021)** — *Fourier Neural
Operator for Parametric Partial Differential Equations.* ICLR 2021.
[arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
: `PointwiseField` is one FNO layer; `OperatorField` is a stack of them. Lift, alternate pointwise
  and spectral mixing, project.

## Sampling

(ref-pcn)=
**Cotter, Roberts, Stuart & White (2013)** — *MCMC Methods for Functions: Modifying Old Algorithms
to Make Them Faster.* Statistical Science 28(3), 413–446.
[arXiv:1202.0709](https://arxiv.org/abs/1202.0709)
: pCN itself, and the dimension-robustness argument. If you read one thing about why the sampler
  is built the way it is, read this.

(ref-parno)=
**Parno & Marzouk (2018)** — *Transport Map Accelerated Markov Chain Monte Carlo.*
SIAM/ASA J. Uncertainty Quantification 6(2). [arXiv:1412.5492](https://arxiv.org/abs/1412.5492)
: Running a sampler in the latent space of a learned transport. `latent_pcn` is the pCN instance
  of this idea.

(ref-neutra)=
**Hoffman, Sountsov, Dillon, Langmore, Tran & Vasudevan (2019)** — *NeuTra-lizing Bad Geometry in
Hamiltonian Monte Carlo Using Neural Transport.*
[arXiv:1903.03704](https://arxiv.org/abs/1903.03704)
: The same construction with HMC instead of pCN. Worth reading if you ever want the gradient-aware
  version.

## Diagnostics

(ref-dax)=
**Dax, Green, Gair, Macke, Buonanno & Schölkopf (2023)** — *Neural Importance Sampling for Rapid
and Reliable Gravitational-Wave Inference.* Phys. Rev. Lett. 130, 171403.
[arXiv:2210.05686](https://arxiv.org/abs/2210.05686)
: `ImportanceCorrection`. Reweight an approximate posterior, get an asymptotically exact one, and
  read the sample efficiency as a hard verdict on the proposal.

(ref-tarp)=
**Lemos, Coogan, Hezaveh & Perreault-Levasseur (2023)** — *Sampling-Based Accuracy Testing of
Posterior Estimators for General Inference.* ICML 2023.
[arXiv:2302.03026](https://arxiv.org/abs/2302.03026)
: TARP, which `coverage_curve` implements. Needs no density and no per-dimension marginalisation,
  and unlike simulation-based calibration it's necessary *and* sufficient.

(ref-vaitl)=
**Vaitl, Nicoli, Nakajima & Kessel (2022)** — *Path-Gradient Estimators for Continuous Normalizing
Flows.* ICML 2022. [arXiv:2206.09016](https://arxiv.org/abs/2206.09016)
: The `path_gradient=True` option on `ReverseKL`. Companion paper
  [arXiv:2207.08219](https://arxiv.org/abs/2207.08219) covers the forward-KL case.

## Further reading

(ref-funcnf)=
**Functional normalizing flow for statistical inverse problems of PDEs.**
[arXiv:2411.13277](https://arxiv.org/abs/2411.13277)
: Closest neighbour to this package in the literature. Worth a read before you claim anything is
  novel.

(ref-saroundtrip)=
**Bayesian imaging inverse problem with SA-Roundtrip prior via HMC-pCN sampler.**
[arXiv:2310.17817](https://arxiv.org/abs/2310.17817)
: The nearest published thing to `latent_pcn` — pCN in the latent space of a learned generative
  prior, in imaging rather than function space.

```{warning}
That last one was found by search and the body hasn't been read. Don't cite it in a paper on our
say-so.
```
