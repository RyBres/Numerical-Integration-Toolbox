from typing import Union
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy


def composite_simpson(expr: sympy.Expr, a: Union[int, float], b: Union[int, float], n: int) -> float:
    '''Composite Simpson's integral approximation.
        
        Parameters:
            expr (sympy.Expr): A SymPy expression, f(x)
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            n (int): The number of iterations
            
        Returns:
            I (float): Floating point approximation of the integral
            
    '''
    class InvalidnException(Exception):
        "Raised when n is an odd number or non-integer."
        pass
    
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
    
    if n % 2 != 0 or not isinstance(n, int):
        raise InvalidnException("n must be an even integer.")
        
    if a > b:
        raise InvalidIntervalException("The upper limit 'a' must be greater than the lower limit 'b'.")
    
    # Get step size
    h = (b - a) / n
    
    # Define array of x values and compute yi at xi
    xarr = np.linspace(a, b, n+1)
    f = lambdify(sympy.symbols('x'), expr, modules=['numpy'])
    yarr = f(xarr)
    
    # Get sum of odd and even indexed yi's
    odd_sum = np.sum(yarr[1:-1:2])
    even_sum = np.sum(yarr[2:-2:2])
    
    # Compute approximation I with Composite Simpson's formula
    I = (h/3) * (yarr[0] + 4 * odd_sum + 2 * even_sum + yarr[-1])

    return I
