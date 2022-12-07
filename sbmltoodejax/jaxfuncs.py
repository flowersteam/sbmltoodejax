import jax.numpy as jnp
from jax import jit

@jit
def factorial(x):
    raise NotImplementedError

@jit
def sec(x):
    return jnp.reciprocal(jnp.cos(x))

@jit
def csc(x):
    return jnp.reciprocal(jnp.sin(x))

@jit
def cot(x):
    return jnp.reciprocal(jnp.tan(x))

@jit
def sech(x):
    return jnp.reciprocal(jnp.cosh(x))

@jit
def csch(x):
    return jnp.reciprocal(jnp.sinh(x))

@jit
def coth(x):
    return jnp.reciprocal(jnp.tanh(x))

@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

@jit
def piecewise(*args):
    cond_list = jnp.array(args[1::2])
    func_list = []
    for arg_idx in range(len(args[::2])):
        func = lambda x, arg_idx=arg_idx: jnp.array(args[2*arg_idx])  # python closure pb: https://stackoverflow.com/questions/20536362/python-append-lambda-functions-to-list
        func_list.append(jit(func))
    if len(cond_list) == len(func_list) - 1:
        cond_list = jnp.concatenate([cond_list, ~cond_list.sum().astype("bool")[jnp.newaxis]])

    return jnp.piecewise(jnp.empty(()), cond_list, func_list)
