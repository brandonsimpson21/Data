# Evolution Strategies
# requires jax, numpy and evosax

import jax.numpy as jnp
import jax
import numpy as np
from evosax import Strategies
from typing import Tuple


def key_gen(seed=0):
    """
    Infinite PRNG key generator

    Args:
        seed int, optional: PRNG starting seed. Defaults to 0.

    Yields:
        jnp.array: PRNG key
    """
    _KEY = jax.random.PRNGKey(seed)
    _KEY, subkey = jax.random.split(_KEY)
    while True:
        yield subkey
        _KEY, subkey = jax.random.split(_KEY)


_key = key_gen()


def new_key():
    """
    get a new PRNG key by calling next() on an instance of inf
    generator produced by key_gen()

    Returns:
        jnp.array: PRNG Key
    """
    return next(_key)


def row_vec(x):
    return x.reshape((1, -1))


def col_vec(x):
    return x.reshape((-1, 1))


def run_es_loop(rng, num_steps, fit_fn, model, **kwargs):
    """
    scan through evolution rollouts

    kwargs are passed to fitness function which should have signature
    fn(x, kwargs) -> float
    Args:
        rng (PRNG): a prng key
        n_generations (int): number of generations

    Returns:
        tuple(jnp.array, Any): (mean, state)
    """
    es_params = model.default_params
    state = model.initialize(rng, es_params)

    def es_step(state_input, tmp):
        rng, state = state_input
        rng, rng_iter = jax.random.split(rng)
        x, state = model.ask(rng_iter, state, es_params)
        fitness = fit_fn(x, kwargs).flatten()
        state = model.tell(x, fitness, state, es_params)
        return [rng, state], fitness[jnp.argmin(fitness)]

    state, scan_out = jax.lax.scan(es_step, [rng, state], [jnp.zeros(num_steps)])
    return jnp.mean(scan_out), state


@jax.jit
def sort_on(arr, on):
    # sortby <on> reshuffle arr to match
    sorted_on, sort_idxs = jax.lax.sort_key_val(on, jnp.arange(0, on.shape[0]))
    sorted_arr = arr[sort_idxs]
    return sorted_arr, sorted_on
