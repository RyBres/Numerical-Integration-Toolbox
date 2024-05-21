from typing import Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

from source.midpoint import midpoint


def _adapt_midpoint_inner(f: sympy.Expr | Callable[[float], float], a: int | float, b: int | float,
                          tol: float, int_input: float) -> float:
    """Inner function that is used for the recursion in adaptive_midpoint.
    """
    # Evaluate left and right integrals at midpoint c
    c = (a + b) / 2

    left_int = midpoint(f, a, c)
    right_int = midpoint(f, c, b)
    whole_int = left_int + right_int
    delta = whole_int - int_input

    # Check if error is within desired tolerance (with a factor of 3 applied), else recursion process
    if abs(delta) <= 3 * tol:
        return whole_int + delta / 3
    else:
        tol /= 2
        left_res = _adapt_midpoint_inner(f, a, c, tol, left_int)
        right_res = _adapt_midpoint_inner(f, c, b, tol, right_int)
        return left_res + right_res


def adaptive_midpoint(f: sympy.Expr | Callable[[float], float], a: float, b: float, tol: float) -> float:
    """Adaptive Midpoint integral approximation. Requires Midpoint function.

        Parameters:
            f (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            tol (int | float): The desired tolerance of the approximation

        Returns:
            I (float): Floating point approximation of the integral

        Author:
            Ryan Bresnahan
    """

    class InvalidIntervalException(Exception):
        """Raised when the upper limit is less than the lower limit."""
        pass

    if a > b:
        raise InvalidIntervalException("The upper limit 'b' must be greater than the lower limit 'a'.")

    # Ensure the expression is evaluated as lambda
    if isinstance(f, sympy.Expr):
        f = lambdify(sympy.symbols('x'), f, modules=['numpy'])

    int_input = midpoint(f, a, b)

    return _adapt_midpoint_inner(f, a, b, tol, int_input)
