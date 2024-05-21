from typing import Union, Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

def midpoint(f: sympy.Expr | Callable[[float], float], a: Union[int, float], b: Union[int, float]) -> float:
    '''Midpoint integral approximation.
        
        Parameters:
            expr (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            
        Returns:
            M (float): Floating point approximation of the integral
                        
        Author:
            Ryan Bresnahan
    '''
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
    
    if a > b:
        raise InvalidIntervalException("The upper limit 'a' must be greater than the lower limit 'b'.")
    
    # Ensure the expression is evaluated as lambda
    if isinstance(f, sympy.Expr):
        f = lambdify(sympy.symbols('x'), f, modules=['numpy'])
    
    # Get step size and midpoint
    h = (b - a)
    c = (a + b) / 2 # this midpoint value is arbitrarily named c - literature typically doesn't name it 'c'
    
    # Compute approximation I with Midpoint approximation formula
    M = h * f(c)

    return M