from sympy import symbols, sin
import romberg
from romberg import romberg
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

# Print the result
print("Result:")
print(result)

a = 0
b = np.pi
n = 5
h = (b - a) / (n - 1)
x = np.linspace(a, b, n)
f = np.sin(x)
I_trap = (h/2)*(f[0] + \
          2 * sum(f[1:n-1]) + f[n-1])