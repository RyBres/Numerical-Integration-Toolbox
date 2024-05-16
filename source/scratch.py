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
    
    # Get step size and define function
    
    f = lambdify(sympy.symbols('x'), expr, modules=['numpy'])
    
    rarr = np.zeros((n, n)) # Initialize array to store Romberg approx
    
    h = b - a
    rarr[0,0] = 0.5 * h * (f(a) + f(b))
    

    
    for i in range(1, n):
        romsum = sum(f(a + (2 * k - 1) * h) for k in range(1, 2**i))
        h *= 0.5
        rarr[i, 0] = 0.5 * (rarr[i-1, 0] + h * romsum)

    for j in range(1, n):
        for k in range(j, n):
            rarr[k, j] = rarr[k, j-1] + (rarr[k, j-1] - rarr[k-1, j-1]) / (4**j - 1)
    
    return rarr


from sympy import symbols, sin
from typing import Union
from sympy import Expr
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

from composite_trapezoid import composite_trapezoid
# Define test parameters
x = symbols('x')
expr = sin(x)
a = 0
b = 1
n = 6

# Call the function under test
result = romberg(sympy.sin(x), 0, np.pi, 11)