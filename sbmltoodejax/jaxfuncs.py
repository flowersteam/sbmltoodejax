import jax.numpy as jnp
from jax import jit

@jit
def factorial(n):
    raise NotImplementedError

@jit
def sec(x):
    """
    Secant function :math:`sec(x) = \\frac{1}{\\cos(x)}`
    """
    return jnp.reciprocal(jnp.cos(x))

@jit
def csc(x):
    """
    Cosecant function :math:`csc(x) = \\frac{1}{\\sin(x)}`
    """
    return jnp.reciprocal(jnp.sin(x))

@jit
def cot(x):
    """
    Cotangent function :math:`cot(x) = \\frac{1}{\\tan(x)}`
    """
    return jnp.reciprocal(jnp.tan(x))

@jit
def sech(x):
    """
    Hyperbolic secant function :math:`sech(x) = \\frac{1}{\\cosh(x)}`
    """
    return jnp.reciprocal(jnp.cosh(x))

@jit
def csch(x):
    """
    Hyperbolic cosecant function :math:`csch(x) = \\frac{1}{\\sinh(x)}`
    """
    return jnp.reciprocal(jnp.sinh(x))

@jit
def coth(x):
    """
    Hyperbolic cotangent function :math:`csch(x) = \\frac{1}{\\tanh(x)}`
    """
    return jnp.reciprocal(jnp.tanh(x))

@jit
def sigmoid(x):
    """
    Sigmoid function :math:`sigmoid(x) = \\frac{1}{1+\\exp(-x)}`
    """
    return 1 / (1 + jnp.exp(-x))

@jit
def piecewise(*args):
    """
    This function implements the Piecewise function used in SBML models:
    :code:`Piecewise(expression1, condition1 [, expression2, condition2 [,...]])`

    Args:
        expressionN (float): a numerical value

        conditionN (bool): a boolean value

    Returns:
        float: The first expression passed as argument with a ``True`` condition, read left to right. If all conditions are false, will return 0.

    Note:
        This function is not intended to be used by a user, but is defined in a way that matches how libSBML formats piecewise
        functions are used in SBML models. This is similar to ``jax.numpy.piecewise`` function but instead of evaluating inputs
        inside the function, they are evaluated before being passed to the function.

    Examples:
        For example, if called like so ``piecewise(x + 2, x < 3, x + 4, x > 3)``
        and if ``x = 2``, then the arguments will be evaluated to ``piecewise(4, True, 6, False)``
        and returns 4.
    """
    cond_list = jnp.array(args[1::2])
    func_list = []
    for arg_idx in range(len(args[::2])):
        func = lambda x, arg_idx=arg_idx: jnp.array(args[2*arg_idx])  # python closure pb: https://stackoverflow.com/questions/20536362/python-append-lambda-functions-to-list
        func_list.append(jit(func))
    if len(cond_list) == len(func_list) - 1:
        cond_list = jnp.concatenate([cond_list, ~cond_list.sum().astype("bool")[jnp.newaxis]])

    return jnp.piecewise(jnp.empty(()), cond_list, func_list)
