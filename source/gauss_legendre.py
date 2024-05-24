from typing import Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy
from numpy.polynomial import Polynomial, Legendre


def gauss_legendre(f: sympy.Expr | Callable[[float], float], a: int | float, b: int | float, n: int) -> float:
    """Gauss-Legendre integral approximation.

        Parameters:
            f (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            n (int): The number of iterations

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

    # Find roots/weights w/ np. Could add in an approx routine with Newton's method on Chebyshev nodes
    nodes, weights = np.polynomial.legendre.leggauss(n)

    px_arr = []

    for i in range(n):
        x = 0.5 * ((b - a) * nodes[i] + (b + a))  # is ~x centered on -1 to 1
        px = weights[i] * f(x)**2
        px_arr.append(px)

    # Sum the elements of array times the jacobian factor, producing answer
    I = sum(px_arr) * (a - b) * 0.5

    return I
