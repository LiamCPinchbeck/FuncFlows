# FuncFlows

Normalizing flows and flow matching on **function space**, for Bayesian inverse problems.

The unknown is a function, represented by its coefficients in a fixed orthonormal basis. A transport reshapes a Gaussian reference measure into a prior or a posterior, and every transport carries its **Radon–Nikodym derivative with respect to that reference measure** — not a Lebesgue density, which does not exist in infinite dimensions.

Based on [Functional Normalizing Flows](https://arxiv.org/abs/2411.13277) (P1) and [Learning Informative Priors with infinite-dimensional continuous normalizing flow](https://arxiv.org/abs/2609.03343v1) (P2).

---

## Layout

```
FuncFlows/
  base_measures/      bases, the reference measure
  transports/
    abstract_transformation.py    the Transformation interface
    layers/                       discrete stacks: Householder, Sylvester
    continuous/                   the ODE, the vector fields, conditioners, samplers
  objectives/         what to minimise
  diagnostics/        how to tell whether it worked
  utils/              the training loop, a Gaussian misfit
```

---

## Quick start

```python
import torch
from FuncFlows.base_measures.bases import FourierBasis
from FuncFlows.base_measures.gaussian_reference_measure import GaussianReferenceMeasure
from FuncFlows.transports.layers.base_discrete import DiscreteTransformation
from FuncFlows.transports.layers.layer_classes import SylvesterLayer
from FuncFlows.objectives.negative_logl import NegativeLogL
from FuncFlows.objectives.reverse_kl import ReverseKL
from FuncFlows.utils.gaussian_misfit import GaussianMisfit
from FuncFlows.utils.train import train

NUM_MODES = 400
basis = FourierBasis(NUM_MODES, physical_dim=2)
measure = GaussianReferenceMeasure(basis, alpha=0.02, power=2.5)   # Matern-3/2

# 1. learn a prior from samples of the field
prior_flow = DiscreteTransformation(measure, layer_class=SylvesterLayer, num_layers=3,
                                    num_modes=NUM_MODES, init_scale=0.005)
train(NegativeLogL(prior_flow, dataset_coeffs, batch_size=128), prior_flow.parameters(),
      num_steps=500, learning_rate=1e-3)
for parameter in prior_flow.parameters():
    parameter.requires_grad_(False)

# 2. the likelihood
potential = GaussianMisfit(forward_map, data_values, noise_std)

# 3. posterior UNDER THE LEARNED PRIOR, by reverse KL in the prior's latent space.
#    KL is invariant under the bijection, so fitting Phi(g(v)) against mu0 and pushing
#    through g gives the posterior. No sampler needed.
latent_potential = lambda latent: potential(prior_flow.push_forward(latent)[0])
post_flow = DiscreteTransformation(measure, layer_class=SylvesterLayer, num_layers=3,
                                   num_modes=NUM_MODES, init_scale=0.005)
train(ReverseKL(post_flow, latent_potential, num_samples=64), post_flow.parameters(),
      num_steps=500, learning_rate=5e-3)

# 4. draw
with torch.no_grad():
    draws = prior_flow.push_forward(post_flow.push_forward(measure.sample(2000))[0])[0]
```

Every transport exposes the same three methods:

| method | returns |
|---|---|
| `push_forward(coeffs)` | `(coeffs_out, log_rn_weight)` |
| `pull_back(coeffs_out)` | the inverse (needed by `NegativeLogL` and learned-prior use) |
| `_map(coeffs)` | `(coeffs_out, log_det)` — the raw map, for checks |

---

## Base measures

```python
FourierBasis(num_functions, physical_dim=1, dtype=torch.float64)   # periodic
CosineBasis(num_functions, physical_dim=1, dtype=torch.float64)    # Neumann
GaussianReferenceMeasure(basis, alpha=0.1, power=2.0, variances=None)
```

`C0 = (I - alpha·Laplacian)^(-power)`, diagonal in the basis. `power = nu + d/2`, so
`power=2.5` in 2-D is Matérn-3/2. Columns are ordered by Laplacian eigenvalue and truncated, so
`num_functions` is a **radial wavenumber cutoff**:

| `num_functions` (2-D Fourier) | max radial \|k\| |
|---|---|
| 400 | 11.2 |
| 800 | 16.0 |
| 1257 | 20.0 |
| 2821 | 30.0 |

Roughly `num_functions ≈ π·k²`. Bases at different truncations are **nested**, so a coefficient
vector embeds into a finer basis by zero-padding, exactly.

---

## Transports

### Discrete stacks

`DiscreteTransformation(measure, layer_class=..., num_layers=N, num_modes=M, **layer_kwargs)`

| layer | map | log-det |
|---|---|---|
| `HouseholderLayer(measure, num_modes)` | `u − ½v(v·u + b)`, unit `v` | exactly `log 0.5` |
| `SylvesterLayer(measure, num_modes, init_scale=0.1)` | `u + A tanh(Bu + b)`, `A`,`B` upper-triangular | exact, `O(num_modes)` by Sylvester's identity |

`SylvesterLayer.pull_back` is a **fixed-point iteration**: it converges when the layer is a
contraction, which a small `init_scale` and `|diag(A)| < 1` encourage but do not guarantee. Check
it once — `(layer._map(layer.pull_back(x))[0] - x).abs().max()`.

### Continuous flows

`ContinuousTransformation(measure, vector_field, total_time=1.0, num_steps=20, method="rk4")`

`f` is the time-`T` flow of `dv/dt = h(v,t)`, and `log_det = ∫ Tr Dh dt`. Compose fields with
`SumField(*fields)` — velocities add, traces add.

| field | what it does | trace |
|---|---|---|
| `AffineField(num_functions, ...)` | `gain_k(t)·v_k` — diagonal, full rank | exact, `O(M)` |
| `DriftField(num_functions, context_dim, rank=None, ...)` | `W(t)c` — no dependence on `v` | **exactly zero** |
| `SpectralField(laplacian_eigenvalues, num_spectral_modes=8, ...)` | mode-diagonal tanh, smooth in `log(1+λ)`; parameter count independent of `M` | exact, diagonal |
| `MatrixField(conditioner, activation="tanh", num_active=None)` | `A(t)·act(B(t)v + b(t))`, rank `num_terms` | exact, `O(L·r)` by Sylvester |
| `PointwiseField(basis, num_active, grid_size, num_field_modes=32, num_spectral_modes=0, ...)` | `a(x)·tanh(g(x)v(x) + (K_t v)(x) + c(x))` — one FNO layer | exact, closed form over the grid |
| `OperatorField(basis, num_active, grid_size, num_channels=4, num_layers=3, trace_samples=1, ...)` | a stack of FNO layers with channel mixing | **Hutchinson** — no closed form |

Conditioners for `MatrixField`:

| conditioner | time dependence |
|---|---|
| `MatrixConditioner(num_functions, num_terms, time_width=0)` | free `A`, `B`; `time_width=0` is time-independent |
| `TimeBasisConditioner(num_functions, num_terms, num_time_modes=4)` | `A(t) = Σ_m cos(mπt/T)·A_m` |
| `TimeNetConditioner(num_functions, num_terms, width=32, depth=3, context_dim=0)` | an MLP maps `t` (and context) to the whole parameter set (P2 eq. 13) |
| `DataConditioner(num_functions, num_terms, context_dim, ...)` | as `TimeBasisConditioner`, plus the data shifts the bias — for amortised posteriors |

`mode_scale=measure.scale` puts the field in the **Cameron–Martin metric**. Mixing metrics between
the field and the loss makes training stick at the do-nothing floor.

---

## Objectives

Every objective is a callable returning a scalar; `train(objective, parameters, ...)` minimises it.

| objective | needs | use |
|---|---|---|
| `NegativeLogL(transformation, dataset_coeffs, batch_size=64)` | `pull_back` + log-det | fit a prior by exact maximum likelihood (P2 Alg. 1). **The only one a discrete stack can use.** |
| `FlowMatching(transformation, dataset_coeffs, batch_size, weights=None, coupling="independent")` | a continuous field | simulation-free prior fitting. `coupling="optimal"` is minibatch OT |
| `PairedFlowMatching(...)` + `reflow_pairs(flow, count)` | a trained flow | reflow: refit on the model's own `(base, endpoint)` pairs, straighter paths |
| `ConditionalFlowMatching(transformation, simulate, draw_base, batch_size, context_weight=0.0)` | a simulator | amortised posterior (FMPE). `context_weight > 0` gives conditional OT |
| `MeanFlowMatching(transformation, dataset_coeffs, batch_size, instant_fraction=0.25)` | a field with `supports_time_pair` | learn the **average** velocity, so sampling is one field evaluation |
| `ReverseKL(transformation, potential, num_samples=30, context=None, path_gradient=False)` | a potential | variational posterior. `path_gradient=True` is the sticking-the-landing estimator |
| `AlphaDivergence(transformation, potential, alpha=2.0)` | a potential | mass-covering alternative to reverse KL |
| `DynamicsPenalty(field, states, kinetic_weight=0.0, jacobian_weight=0.0)` | a field | RNODE regularisers: simple dynamics, so the solver can be cheap |

`weights=measure.variances.rsqrt()` is the Cameron–Martin weighting for the flow-matching losses.


---

## References

### Functional flows

- [Functional Normalizing Flows](https://arxiv.org/abs/2411.13277) - Yang Zhao, Haoyu Lu, Junxiong Jia, Tao Zhou
- [Learning Informative Priors with infinite-dimensional continuous normalizing flow for Bayesian inverse problems](https://arxiv.org/abs/2609.03343v1) - Yang Zhao, Junxiong Jia, Tao Zhou
- [Universal Functional Regression with Neural Operator Flows](https://arxiv.org/abs/2404.02986) - Yaozhong Shi, Angela F. Gao, Zachary E. Ross, Kamyar Azizzadenesheli



### Functional flow matching and diffusion

- [Functional Flow Matching](https://arxiv.org/abs/2305.17209) - Gavin Kerrigan, Giosue Migliorini, Padhraic Smyth
- [Discretization and Statistical Consistency of Functional Flow Matching](https://arxiv.org/abs/2608.04531) - Lennon J. Shikhman
- [Optimal-Transport-Guided Functional Flow Matching for Turbulent Field Generation in Hilbert Space](https://arxiv.org/abs/2604.05700) - Kunpeng Li, Chenguang Wan, Zhisong Qu, Kyungtak Lim, Virginie Grandgirard, Xavier Garbet, Hua Yu, Ong Yew Soon
- [Stochastic Process Learning via Operator Flow Matching](https://arxiv.org/abs/2501.04126) - Yaozhong Shi, Zachary E. Ross, Domniki Asimaki, Kamyar Azizzadenesheli
- [Probability-Flow ODE in Infinite-Dimensional Function Spaces](https://arxiv.org/abs/2503.10219) - Kunwoo Na, Junghyun Lee, Se-Young Yun, Sungbin Lim
- [Flow Straight and Fast in Hilbert Space: Functional Rectified Flow](https://arxiv.org/abs/2509.10384) - Jianxin Zhang, Clayton Scott
- [Diffusion Generative Models in Infinite Dimensions](https://proceedings.mlr.press/v206/kerrigan23a.html) - Gavin Kerrigan, Justin Ley, Padhraic Smyth
- [Score-based Diffusion Models in Function Space](https://arxiv.org/abs/2302.07400) - Jae Hyun Lim, Nikola B. Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, Anima Anandkumar
- [Infinite-Dimensional Diffusion Models](https://arxiv.org/abs/2302.10130) - Jakiw Pidstrigach, Youssef Marzouk, Sebastian Reich, Sven Wang
- [Conditional score-based diffusion models for Bayesian inference in infinite dimensions](https://arxiv.org/abs/2305.19147) - Lorenzo Baldassari, Ali Siahkoohi, Josselin Garnier, Knut Solna, Maarten V. de Hoop
- [Infinite Resolution Diffusion with Subsampled Mollified States](https://arxiv.org/abs/2303.18242) - Sam Bond-Taylor, Chris G. Willcocks





### Flow matching

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797) - Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) - Xingchao Liu, Chengyue Gong, Qiang Liu
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482) - Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, Yoshua Bengio
- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) - Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He
- [Flow Matching for Scalable Simulation-Based Inference](https://arxiv.org/abs/2305.17161) - Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen R. Green, Jakob H. Macke, Bernhard Schölkopf
- [Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching](https://arxiv.org/abs/2403.18705) - Jannis Chemseddine, Paul Hagemann, Christian Wald, Gabriele Steidl
- [Dynamic Conditional Optimal Transport through Simulation-Free Flows](https://arxiv.org/abs/2404.04240) - Gavin Kerrigan, Giosue Migliorini, Padhraic Smyth




### Normalizing flows

- [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) - Danilo Jimenez Rezende, Shakir Mohamed
- [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) - Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
- [Improving Variational Auto-Encoders using Householder Flow](https://arxiv.org/abs/1611.09630) - Jakub M. Tomczak, Max Welling
- [Sylvester Normalizing Flows for Variational Inference](https://arxiv.org/abs/1803.05649) - Rianne van den Berg, Leonard Hasenclever, Jakub M. Tomczak, Max Welling
- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) - George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) - Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367) - Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, David Duvenaud
- [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681) - Emilien Dupont, Arnaud Doucet, Yee Whye Teh
- [Invertible Monotone Operators for Normalizing Flows](https://openreview.net/forum?id=USoYIT4IQz) - Byeongkeun Ahn, Chiyoon Kim, Youngjoon Hong, Hyunwoo J. Kim


### Neural operators

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895) - Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- [Neural Operator: Learning Maps Between Function Spaces](https://arxiv.org/abs/2108.08481) - Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://arxiv.org/abs/1910.03193) - Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, George Em Karniadakis
- [Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning](https://arxiv.org/abs/2305.19913) - Francesca Bartolucci, Emmanuel de Bézenac, Bogdan Raonić, Roberto Molinaro, Siddhartha Mishra, Rima Alaifari





### Bayesian inverse problems in function space

- [Inverse problems: a Bayesian perspective](https://arxiv.org/abs/1302.6989) - Andrew M. Stuart
- [MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster](https://arxiv.org/abs/1202.0709) - Simon L. Cotter, Gareth O. Roberts, Andrew M. Stuart, David White
- [Geometric MCMC for infinite-dimensional inverse problems](https://arxiv.org/abs/1606.06351) - Alexandros Beskos, Mark Girolami, Shiwei Lan, Patrick E. Farrell, Andrew M. Stuart
- [Optimal low-rank approximations of Bayesian linear inverse problems](https://arxiv.org/abs/1407.3463) - Alessio Spantini, Antti Solonen, Tiangang Cui, James Martin, Luis Tenorio, Youssef Marzouk
- [Optimal low-rank posterior mean and distribution approximation in linear Gaussian inverse problems on Hilbert spaces](https://arxiv.org/abs/2503.24209) - Giuseppe Carere, Han Cheng Lie
- [Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](https://arxiv.org/abs/2505.18276) - Lorenzo Baldassari, Ali Siahkoohi, Josselin Garnier, Knut Solna, Maarten V. de Hoop


### Reduced bases and latent representations

- [Model Reduction and Neural Networks for Parametric PDEs](https://arxiv.org/abs/2005.03180) - Kaushik Bhattacharya, Bamdad Hosseini, Nikola B. Kovachki, Andrew M. Stuart
- [Normalizing field flows: Solving forward and inverse stochastic differential equations using physics-informed flow models](https://arxiv.org/abs/2108.12956) - Ling Guo, Hao Wu, Tao Zhou
- [From data to functa: Your data point is a function and you can treat it like one](https://proceedings.mlr.press/v162/dupont22a.html) - Emilien Dupont, Hyunjik Kim, S. M. Ali Eslami, Danilo Jimenez Rezende, Dan Rosenbaum
- [Reduced Basis Methods: Success, Limitations and Future Challenges](https://arxiv.org/abs/1511.02021) - Mario Ohlberger, Stephan Rave
- [Efficiently Sampling Functions from Gaussian Process Posteriors](https://arxiv.org/abs/2002.09309) - James T. Wilson, Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth



### Evaluation

- [A note on the evaluation of generative models](https://arxiv.org/abs/1511.01844) - Lucas Theis, Aäron van den Oord, Matthias Bethge
- [Do Deep Generative Models Know What They Don't Know?](https://arxiv.org/abs/1810.09136) - Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, Balaji Lakshminarayanan
- [Validating Bayesian Inference Algorithms with Simulation-Based Calibration](https://arxiv.org/abs/1804.06788) - Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, Andrew Gelman
- [Sampling-Based Accuracy Testing of Posterior Estimators for General Inference](https://arxiv.org/abs/2302.03026) - Pablo Lemos, Adam Coogan, Yashar Hezaveh, Laurence Perreault-Levasseur
- [Compressed Sensing using Generative Models](https://arxiv.org/abs/1703.03208) - Ashish Bora, Ajil Jalal, Eric Price, Alexandros G. Dimakis
- [Deep Image Prior](https://arxiv.org/abs/1711.10925) - Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
- [Regularized estimation of large covariance matrices](https://arxiv.org/abs/0803.1909) - Peter J. Bickel, Elizaveta Levina



### Information field theory

- [Information field theory for cosmological perturbation reconstruction and nonlinear signal analysis](https://arxiv.org/abs/0806.3474) - Torsten A. Enßlin, Mona Frommert, Francisco S. Kitaura
- [Metric Gaussian Variational Inference](https://arxiv.org/abs/1901.11033) - Jakob Knollmüller, Torsten A. Enßlin
- [Geometric Variational Inference](https://arxiv.org/abs/2105.10470) - Philipp Frank, Reimar Leike, Torsten A. Enßlin
- [NIFTy.re: A Versatile Python Library for High-Performance Bayesian Imaging](https://arxiv.org/abs/2402.16683) - Gordian Edenhofer, Philipp Frank, Jakob Roth, Reimar H. Leike, Massin Guerdi, Lukas I. Scheel-Platz, Matteo Guardiani, Vincent Eberle, Margret Westerkamp, Torsten A. Enßlin

