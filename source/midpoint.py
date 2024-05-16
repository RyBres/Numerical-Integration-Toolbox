from typing import Union
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

def trapezoidal(expr: sympy.Expr, a: Union[int, float], b: Union[int, float]) -> float:
    '''Trapezoidal integral approximation.
        
        Parameters:
            expr (sympy.Expr): A SymPy expression, f(x)
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            
        Returns:
            M (float): Floating point approximation of the integral
            
    '''
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
    
    if a > b:
        raise InvalidIntervalException("The upper limit 'a' must be greater than the lower limit 'b'.")
    
    # Get step size and midpoint
    h = (b - a)
    c = (a + b) / 2 # this midpoint value is arbitrarily named c - literature typically doesn't name it 'c'
    
    f = lambdify(sympy.symbols('x'), expr, modules=['numpy'])
    
    # Compute approximation I with Trapezoidal approximation formula
    M = h * f(c)

    return M