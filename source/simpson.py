from typing import Union
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy


def simpson(expr: sympy.Expr, a: Union[int, float], b: Union[int, float]) -> float:
    '''Simpson's integral approximation.
        
        Parameters:
            expr (sympy.Expr): A SymPy expression, f(x)
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            
        Returns:
            I (float): Floating point approximation of the integral
            
    '''
    
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
        
    if a > b:
        raise InvalidIntervalException("The upper limit 'a' must be greater than the lower limit 'b'.")
    
    # Get step size
    h = (b - a) / 2
    c = (a + b) / 2 # this midpoint value is arbitrarily named c - literature typically doesn't name it 'c'
    
    # Define array of x values and compute yi at xi
    xarr = [a, c, b]
    f = lambdify(sympy.symbols('x'), expr, modules=['numpy'])
    yarr = f(xarr)
    
    # Compute approximation I with Simpson's formula
    I = (h/3) * (yarr[0] + 4 * yarr[1] + yarr[2])

    return I