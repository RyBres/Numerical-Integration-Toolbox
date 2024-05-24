from typing import Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy
from numpy.polynomial import Polynomial, Legendre



def legendre(f: sympy.Expr | Callable[[float], float], a: int | float, b: int | float, n: int):

    # Find roots/weights w/ np. Could add in an approx routine with Newton's method on Chebyshev nodes
    

    h = (a - b) / 2 # outer constant of integral

    for i in range(n):
        # t is the root for that index
        x = (0.5 * ((b - a)*t + (b + a))**2) * h # is ~x
        # substitute ~x into f(x)
        prod = weight * f(x)
        # add to array showing elements of summation

    # sum array times outer product (which is h)


