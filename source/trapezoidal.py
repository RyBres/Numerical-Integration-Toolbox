from typing import Union, Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy


def trapezoidal(f: Union[sympy.Expr, Callable[[float], float]], a: Union[int, float], b: Union[int, float]) -> float:
    """Trapezoidal integral approximation.

        Parameters:
            f (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration

        Returns:
            T (float): Floating point approximation of the integral

    """

    class InvalidIntervalException(Exception):
        """Raised when the upper limit is less than the lower limit."""
        pass

    if a > b:
        raise InvalidIntervalException("The upper limit 'a' must be greater than the lower limit 'b'.")

    # Ensure the expression is evaluated as lambda
    if isinstance(f, sympy.Expr):
        f = lambdify(sympy.symbols('x'), f, modules=['numpy'])

    # Get step size
    h = (b - a)

    # Define array of x values and compute yi at xi
    xarr = [a, b]
    yarr = f(xarr)

    # Compute approximation I with Trapezoidal approximation formula
    T = (h / 2) * (yarr[0] + yarr[1])

    return T
