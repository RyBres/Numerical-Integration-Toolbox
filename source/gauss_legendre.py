from typing import Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy
from numpy.polynomial import Polynomial, Legendre



def legendre(f: sympy.Expr | Callable[[float], float], a: int | float, b: int | float, n: int):

    # Find roots/weights w/ np

    h = (a - b) / 2

    for i in range(n):
        x = 0.5 * ((b - a)*t + (b + a))**2
        prod = weight * f(x)
        # add to array

    # sum array times outer product


