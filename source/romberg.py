from typing import Union, Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

from composite_trapezoid import composite_trapezoid


def romberg(expr: sympy.Expr, a: int | float, b: int | float, n: int) -> float:
    '''Romberg's integral approximation. Requires composite_trapezoid function.
        
        Parameters:
            expr (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            n (int): The number of iterations
            
        Returns:
            R (float): Floating point approximation of the integral, R_n,n, at O(h^2n)
        
        Author:
            Ryan Bresnahan
    '''
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
    
    if n < 1 or not isinstance(n, int):
        raise ValueError("n must be a positive integer.")
        
    if a > b:
        raise InvalidIntervalException("The upper limit 'a' must be greater than the lower limit 'b'.")
    
    rarr = np.zeros((n, n)) # Initialize array to store Romberg approx

    for i in range(n): # Obtain first col estimates
        m = 2**i + 1
        rarr[i, 0] = composite_trapezoid(expr, a, b, m)
        
    for j in range(1, n): # Extrapolation across matrix
        for k in range(j, n):
            rarr[k, j] = rarr[k, j-1] + (rarr[k, j-1] - rarr[k-1, j-1]) / (4**j - 1)
    
    return rarr