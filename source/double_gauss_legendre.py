from typing import Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy
from numpy.polynomial import Polynomial, Legendre
from source.gauss_legendre import gauss_legendre

def checkfun(fun, num_vars):
    """Helper function. Ensures the function is evaluated as a lambda expression."""
    if isinstance(fun, sympy.Expr):
        variables = sympy.symbols('x y')[:num_vars] # Lets checkfun be applied to fun's w/ up to 2 variables (x and y)
        fun = lambdify(variables, fun, modules=['numpy'])
    return fun

def double_gauss_legendre(f: sympy.Expr | Callable[[float], float],
                          a: int | float, b: int | float,
                          c: Callable[[float], float] | int | float,
                          d: Callable[[float], float] | int | float,
                          n: int, m: int) -> float:
    """Gauss-Legendre double integral approximation. Note that the bounds of the inner integral may be real numbers or functions.

        Parameters:
            f (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration for the outer integral
            b (int | float): The upper limit of integration for the outer integral
            c (Callable[[float], float] | int | float): The lower limit of integration for the inner integral
            d (Callable[[float], float] | int | float): The upper limit of integration for the inner integral
            n (int): The number of iterations for the outer integral

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

    if isinstance(d, (int, float)) and isinstance(c, (int, float)):
        if c > d:
            raise InvalidIntervalException("The upper limit 'd' must be greater than the lower limit 'c'.")

    # Ensure the functions are evaluated as lambda expressions
    f = checkfun(f, 2)
    c = checkfun(c, 1) if callable(c) else c
    d = checkfun(d, 1) if callable(d) else d

    # Define inner integral
    def _inner_integral(y):
        """Inner function for double_gauss_legendre. Computes the inner integral of with respect to x."""
        cx = c(y) if callable(c) else c
        dx = d(y) if callable(d) else d
        return gauss_legendre(lambda x: f(x, y), cx, dx, m)

    # Apply nested integral w/ inner function.
    I = gauss_legendre(_inner_integral, a, b, n)
    return I