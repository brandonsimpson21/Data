from jax import jit
from jax.lax import scan
import jax.numpy as jnp
import jax.scipy as jsc

import math
from typing import Tuple
from collections import namedtuple

StateSpaceModel = namedtuple(
    "StateSpaceModel", ["F", "H", "Q", "R", "m0", "P0", "xdim", "ydim"]
)


@jit
def mvn_logpdf(x, mean, cov):
    n = mean.shape[0]
    upper = jsc.linalg.cholesky(cov, lower=False)
    log_det = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(upper))))
    diff = x - mean
    scaled_diff = jsc.linalg.solve_triangular(upper, diff.T, lower=False)
    distance = jnp.sum(scaled_diff * scaled_diff, 0)
    return -0.5 * (distance + n * jnp.log(2 * math.pi) + log_det)


@jit
def kallman_filter(model: StateSpaceModel, observations) -> Tuple[jnp.array, jnp.array]:
    def body(carry, y):
        m, P = carry
        m = model.F @ m
        P = model.F @ P @ model.F.T + model.Q

        S = model.H @ P @ model.H.T + model.R

        K = jsc.linalg.solve(S, model.H @ P, assume_a="pos").T
        m = m + K @ (y - model.H @ m)
        P = P - K @ S @ K.T
        return (m, P), (m, P)

    _, (fms, fPs) = scan(body, (model.m0, model.P0), observations)
    return fms, fPs


@jit
def kallman_smoother(
    model: StateSpaceModel, ms: jnp.array, Ps: jnp.array
) -> Tuple[jnp.array, jnp.array]:
    def body(carry, inp):
        m, P = inp
        sm, sP = carry

        pm = model.F @ m
        pP = model.F @ P @ model.F.T + model.Q

        C = jsc.linalg.solve(pP, model.F @ P, assume_a="pos").T  # notice the jsc here

        sm = m + C @ (sm - pm)
        sP = P + C @ (sP - pP) @ C.T
        return (sm, sP), (sm, sP)

    _, (sms, sPs) = scan(body, (ms[-1], Ps[-1]), (ms[:-1], Ps[:-1]), reverse=True)
    sms = jnp.append(sms, jnp.expand_dims(ms[-1], 0), 0)
    sPs = jnp.append(sPs, jnp.expand_dims(Ps[-1], 0), 0)
    return sms, sPs


@jit
def smoothed_kallman_filter(model, observations):
    return kallman_smoother(model, *kallman_filter(model, observations))
