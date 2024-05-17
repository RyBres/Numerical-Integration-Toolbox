from typing import Union, Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

def composite_simpson(expr: Union[sympy.Expr, Callable[[float], float]], a: Union[int, float], b: Union[int, float], n: int) -> float:
    '''Composite Simpson's integral approximation.
        
        Parameters:
            expr (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            n (int): The number of iterations
            
        Returns:
            I (float): Floating point approximation of the integral
            
    '''
    
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
    
    if n % 2 != 0 or not isinstance(n, int):
        raise ValueError("'n' must be an even integer.")
        
    if a > b:
        raise InvalidIntervalException("The upper limit 'b' must be greater than the lower limit 'a'.")
    
    # Ensure the expression is evaluated as lambda
    if isinstance(expr, sympy.Expr):
        f = lambdify(sympy.symbols('x'), expr, modules=['numpy'])
    elif callable(expr):
        f = expr
    else:
        raise TypeError("'expr' must be either a sympy or lambda expression.")
        
    # Get step size
    h = (b - a) / n
    
    # Define array of x values and compute yi at xi
    xarr = np.linspace(a, b, n+1)
    yarr = f(xarr)
    
    # Get sum of odd and even indexed yi's
    odd_sum = np.sum(yarr[1:-1:2])
    even_sum = np.sum(yarr[2:-1:2])
    
    # Compute approximation I with Composite Simpson's formula
    I = (h/3) * (yarr[0] + 4 * odd_sum + 2 * even_sum + yarr[-1])

    return I
