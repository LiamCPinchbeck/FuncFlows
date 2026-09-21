# Samplers

MCMC bolted onto a trained flow. Exact under a learned PRIOR, where the acceptance ratio needs
no Jacobian; under a posterior flow the potential carries `log_rn_at` and inherits the ODE
solver's accuracy along with it.

```{eval-rst}
.. automodule:: FuncyFlows.samplers.latent_pcn
   :members:
   :show-inheritance:
```
