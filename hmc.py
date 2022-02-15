import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def normal_like_tree(a, key):
    treedef = jax.tree_structure(a)
    num_vars = len(jax.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_multimap(lambda p, k: jax.random.normal(k, shape=p.shape), a, jax.tree_unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]


def leapfrog(params, momentum, log_prob_fn, step_size, n_steps):
    
    # define a single step
    def step(i, args):
        params, momentum = args
        
        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g, momentum, grad)

        # update params
        params = jax.tree_multimap(lambda s, m: s + m * step_size, params, momentum)

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g, momentum, grad)
        
        return params, momentum

    # do 'n_steps'
    new_params, new_momentum = jax.lax.fori_loop(0, n_steps, step, (params, momentum))

    return new_params, new_momentum


def hmc_sampler(params, log_prob_fn, n_steps, n_leapfrog_steps, step_size, key):

    # define a single step
    def step(i, args):
        params, params_history, total_accept_prob, key = args
        key, normal_key, uniform_key = jax.random.split(key, 3)

        # generate random momentum
        momentum, _ = normal_like_tree(params, normal_key)

        # leapfrog
        new_params, new_momentum = leapfrog(params, momentum, log_prob_fn, step_size, n_leapfrog_steps)

        # MH correction
        potentaial_energy_diff = log_prob_fn(new_params) - log_prob_fn(params)
        kinetic_energy_diff = 0.5*sum([jnp.sum(m1**2-m2**2) for m1, m2 in zip(jax.tree_leaves(momentum), jax.tree_leaves(new_momentum))])
        log_accept_prob = potentaial_energy_diff + kinetic_energy_diff
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        total_accept_prob += accept_prob
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, new_params, params)
        
        # store history
        params_raveled = ravel_pytree(params)[0]
        params_history = params_history.at[i].set(params_raveled)
         
        return params, params_history, total_accept_prob, key
    
    # ravel params
    params_raveled, unravel_fn = ravel_pytree(params)
    
    # do 'n_steps'
    params_history_raveled = jnp.zeros([n_steps]+list(params_raveled.shape))
    _, params_history_raveled, total_accept_prob, key = jax.lax.fori_loop(0, n_steps, step, (params, params_history_raveled, 0, key))
    
    # unravel params
    params_history_unraveled = [unravel_fn(params_raveled) for params_raveled in params_history_raveled]
    
    print(f'Avg. accept. prob.: {(total_accept_prob/n_steps):.2%}')
    return params_history_unraveled

