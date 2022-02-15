# mini-hmc-jax 

This is a simple implementation of Hamiltonian Monte Carlo in JAX that is vectorized and supports pytree parameters (i.e. tree-like structures).

Here's a minimal example to sample from a distribution:

```python
import jax
import jax.numpy as jnp
from hmc import hmc_sampler

# define target distribution
def target_log_pdf(params):
    return jax.scipy.stats.t.logpdf(params, df=1).sum()

# run HMC
params_init = jnp.zeros(10)
key = jax.random.PRNGKey(0)
chain = hmc_sampler(params_init, target_log_pdf, n_steps=10, n_leapfrog_steps=100, step_size=0.1, key=key)
```

Inspired by the paper [What Are Bayesian Neural Network Posteriors Really Like?](https://github.com/google-research/google-research/tree/master/bnn_hmc)

```bibtex
@article{izmailov2021bayesian,
  title={What Are Bayesian Neural Network Posteriors Really Like?},
  author={Izmailov, Pavel and Vikram, Sharad and Hoffman, Matthew D and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2104.14421},
  year={2021}
}
```
