"""Minimal valid submission: spends the burn-in without using it, returns defaults."""
import jax.numpy as jnp

from evaluator.interface import Result
from evaluator.mclmc_stepper import Beta


def phase_fractions(config):
    return (1.0,)


def init(x0, key, config):
    return {"query_position": x0, "production_position": x0, "beta": default_beta(config.dim, x0.dtype)}


def default_beta(dim, dtype):
    root = jnp.sqrt(jnp.asarray(dim, dtype))
    return Beta(root, 0.25 * root, jnp.ones(dim, dtype))


def step_phase(phase, carry, obs):
    return carry, {}


def transition(carry, phase_history, phase_info):
    return carry


def finalize(carry, history, config):
    return Result(carry["production_position"], carry["beta"])
