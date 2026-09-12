# FuncFlows
A python package for functional flows, focusing on minimal design and easy usability. Based around a few works:

### Functional flows

- [Functional Normalizing Flows](https://arxiv.org/abs/2411.13277) - Yang Zhao, Haoyu Lu, Junxiong Jia, Tao Zhou
- [Learning Informative Priors with infinite-dimensional continuous normalizing flow for Bayesian inverse problems](https://arxiv.org/abs/2609.03343v1) - Yang Zhao, Junxiong Jia, Tao Zhou
- [Universal Functional Regression with Neural Operator Flows](https://arxiv.org/abs/2404.02986) - Yaozhong Shi, Angela F. Gao, Zachary E. Ross, Kamyar Azizzadenesheli

### Functional flow matching and diffusion

- [Functional Flow Matching](https://arxiv.org/abs/2305.17209) - Gavin Kerrigan, Giosue Migliorini, Padhraic Smyth
- [Discretization and Statistical Consistency of Functional Flow Matching](https://arxiv.org/abs/2608.04531) - Lennon J. Shikhman
- [Probability-Flow ODE in Infinite-Dimensional Function Spaces](https://arxiv.org/abs/2503.10219) - Kunwoo Na, Junghyun Lee, Se-Young Yun, Sungbin Lim
- [Flow Straight and Fast in Hilbert Space: Functional Rectified Flow](https://arxiv.org/abs/2509.10384) - Jianxin Zhang, Clayton Scott
- [Diffusion Generative Models in Infinite Dimensions](https://proceedings.mlr.press/v206/kerrigan23a.html) - Gavin Kerrigan, Justin Ley, Padhraic Smyth
- [Score-based Diffusion Models in Function Space](https://arxiv.org/abs/2302.07400) - Jae Hyun Lim, Nikola B. Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, Anima Anandkumar
- [Infinite-Dimensional Diffusion Models](https://arxiv.org/abs/2302.10130) - Jakiw Pidstrigach, Youssef Marzouk, Sebastian Reich, Sven Wang
- [Infinite Resolution Diffusion with Subsampled Mollified States](https://arxiv.org/abs/2303.18242) - Sam Bond-Taylor, Chris G. Willcocks

### Flow matching

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797) - Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) - Xingchao Liu, Chengyue Gong, Qiang Liu
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482) - Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, Yoshua Bengio
- [Flow Matching for Scalable Simulation-Based Inference](https://arxiv.org/abs/2305.17161) - Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen R. Green, Jakob H. Macke, Bernhard Schölkopf

### Normalizing flows

- [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) - Danilo Jimenez Rezende, Shakir Mohamed
- [Improving Variational Auto-Encoders using Householder Flow](https://arxiv.org/abs/1611.09630) - Jakub M. Tomczak, Max Welling
- [Sylvester Normalizing Flows for Variational Inference](https://arxiv.org/abs/1803.05649) - Rianne van den Berg, Leonard Hasenclever, Jakub M. Tomczak, Max Welling
- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) - George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) - Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367) - Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, David Duvenaud
- [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681) - Emilien Dupont, Arnaud Doucet, Yee Whye Teh
- [Understanding and Mitigating Exploding Inverses in Invertible Neural Networks](https://arxiv.org/abs/2006.09347) - Jens Behrmann, Paul Vicol, Kuan-Chieh Wang, Roger Grosse, Jörn-Henrik Jacobsen

### Neural operators

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895) - Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- [Neural Operator: Learning Maps Between Function Spaces](https://arxiv.org/abs/2108.08481) - Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar

### Bayesian inverse problems in function space

- [Inverse problems: a Bayesian perspective](https://arxiv.org/abs/1302.6989) - Andrew M. Stuart
- [MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster](https://arxiv.org/abs/1202.0709) - Simon L. Cotter, Gareth O. Roberts, Andrew M. Stuart, David White
- [Geometric MCMC for infinite-dimensional inverse problems](https://arxiv.org/abs/1606.06351) - Alexandros Beskos, Mark Girolami, Shiwei Lan, Patrick E. Farrell, Andrew M. Stuart
- [Optimal low-rank approximations of Bayesian linear inverse problems](https://arxiv.org/abs/1407.3463) - Alessio Spantini, Antti Solonen, Tiangang Cui, James Martin, Luis Tenorio, Youssef Marzouk
- [Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](https://arxiv.org/abs/2505.18276) - Lorenzo Baldassari, Ali Siahkoohi, Josselin Garnier, Knut Solna, Maarten V. de Hoop

### Information field theory

- [Information field theory for cosmological perturbation reconstruction and nonlinear signal analysis](https://arxiv.org/abs/0806.3474) - Torsten A. Enßlin, Mona Frommert, Francisco S. Kitaura
- [Metric Gaussian Variational Inference](https://arxiv.org/abs/1901.11033) - Jakob Knollmüller, Torsten A. Enßlin
- [Geometric Variational Inference](https://arxiv.org/abs/2105.10470) - Philipp Frank, Reimar Leike, Torsten A. Enßlin
- [NIFTy.re: A Versatile Python Library for High-Performance Bayesian Imaging](https://arxiv.org/abs/2402.16683) - Gordian Edenhofer, Philipp Frank, Jakob Roth, Reimar H. Leike, Massin Guerdi, Lukas I. Scheel-Platz, Matteo Guardiani, Vincent Eberle, Margret Westerkamp, Torsten A. Enßlin

### Evaluation

- [A note on the evaluation of generative models](https://arxiv.org/abs/1511.01844) - Lucas Theis, Aäron van den Oord, Matthias Bethge
- [Do Deep Generative Models Know What They Don't Know?](https://arxiv.org/abs/1810.09136) - Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, Balaji Lakshminarayanan
- [Validating Bayesian Inference Algorithms with Simulation-Based Calibration](https://arxiv.org/abs/1804.06788) - Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, Andrew Gelman
- [Compressed Sensing using Generative Models](https://arxiv.org/abs/1703.03208) - Ashish Bora, Ajil Jalal, Eric Price, Alexandros G. Dimakis
- [Deep Image Prior](https://arxiv.org/abs/1711.10925) - Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
- [Regularized estimation of large covariance matrices](https://arxiv.org/abs/0803.1909) - Peter J. Bickel, Elizaveta Levina




# Dev Note

Short variable names are for losers. If your variable/function/class name is three letters or less, reconsider.