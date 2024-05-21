from typing import Callable
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

from composite_simpson import composite_simpson
    
def adaptive_composite_simpson(f: sympy.Expr | Callable[[float], float], a: float, b: float, tol: float, n: int = 10) -> float:
    '''Composite Simpson's adaptive integral approximation. Requires composite_simpson function.
        
        Parameters:
            expr (sympy.Expr | Callable[[float], float]): A SymPy expression or lambda expression
            a (int | float): The lower limit of integration
            b (int | float): The upper limit of integration
            tol (int | float): The desired tolerance of the approximation
            n_initial (int = 10): Number of iterations for initial approximation. Default is 10.
            
        Returns:
            I (float): Floating point approximation of the integral
        
        Author:
            Ryan Bresnahan
    '''
    
    def _adapt_composite_simpson_inner(f: sympy.Expr | Callable[[float], float], a: int | float, b: int | float, tol: float, int_input: float, n: int) -> float:
        '''Inner function that is used for the recursion in adaptive_composite_simpson.
        '''
        # Evaluate left and right integrals at midpoint c
        c = (a + b) / 2
        
        left_int = composite_simpson(f, a, c, n)
        right_int= composite_simpson(f, c, b, n)
        whole_int = left_int + right_int
        delta = whole_int - int_input
        
        # Check if error is within desired tolerance, else recursion process
        if abs(delta) <= 15 * tol:
            return whole_int + delta / 15
        else:
            tol /= 2
            left_res = _adapt_composite_simpson_inner(f, a, c, tol, left_int)
            right_res = _adapt_composite_simpson_inner(f, c, b, tol, right_int)
            return left_res + right_res
    
    class InvalidIntervalException(Exception):
        "Raised when the upper limit is less than the lower limit."
        pass
    
    if n % 2 != 0 or not isinstance(n, int):
        raise ValueError("'n_initial' must be an even integer.")
        
    if a > b:
        raise InvalidIntervalException("The upper limit 'b' must be greater than the lower limit 'a'.")
    
    # Ensure the expression is evaluated as lambda
    if isinstance(f, sympy.Expr):
        f = lambdify(sympy.symbols('x'), f, modules=['numpy'])
    
    int_input = composite_simpson(f, a, b, n) # Note that n_initial is conceptually different from the n that is typically used in the integration functions - the n will exceed n_initial if there is recursion (and hence more intervals)
    
    return _adapt_composite_simpson_inner(f, a, b, tol, int_input)
        