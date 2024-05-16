from typing import Union
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

from composite_trapezoid import composite_trapezoid


def romberg(expr: sympy.Expr, a: Union[int, float], b: Union[int, float], n: int):
    '''Romberg's integral approximation. Requires composite_trapezoid function.
        
        Parameters:
            expr (sympy.Expr): A SymPy expression, f(x)
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            n (int): The number of iterations
            
        Returns:
            R (float): Floating point approximation of the integral, R_n,n, at O(h^2n)
        
        Author:
            Ryan Bresnahan
    '''
    class InvalidnException(Exception):
        "Raised when n < 1 or is a non-integer."
        pass
    
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
    
    if n < 1 or not isinstance(n, int):
        raise InvalidnException("n must be a positive integer.")
        
    if a > b:
        raise InvalidIntervalException("The upper limit 'a' must be greater than the lower limit 'b'.")
    
    rarr = np.zeros((n, n)) # Initialize array to store Romberg approx

    for i in range(n):
        m = 2**i + 1
        rarr[i, 0] = composite_trapezoid(expr, a, b, m)
        
    for j in range(1, n):
        for k in range(j, n):
            rarr[k, j] = rarr[k, j-1] + (rarr[k, j-1] - rarr[k-1, j-1]) / (4**j - 1)
    
    return rarr