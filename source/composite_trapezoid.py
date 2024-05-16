from typing import Union
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy


def composite_trapezoid(expr: sympy.Expr, a: Union[int, float], b: Union[int, float], n: int) -> float:
    '''Composite Trapezoid integral approximation.
        
        Parameters:
            expr (sympy.Expr): A SymPy expression, f(x)
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            n (int): The number of iterations
            
        Returns:
            I (float): Floating point approximation of the integral
            
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
    
    # Get step size
    h = (b - a) / (n - 1)
    
    # Define array of x values and compute yi at xi
    xarr = np.linspace(a, b, n)
    f = lambdify(sympy.symbols('x'), expr, modules=['numpy'])
    yarr = f(xarr)
    
    # Compute approximation I with Composite Trapezoid formula
    I = (h/2) * (yarr[0] + 2 * sum(yarr[1:-1]) + yarr[-1])

    return I
