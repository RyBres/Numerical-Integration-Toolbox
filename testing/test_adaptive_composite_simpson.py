import unittest

import numpy as np
import sympy

from source.adaptive_composite_simpson import adaptive_composite_simpson


class TestAdaptiveCompositeSimpson(unittest.TestCase):
    
    def test_polynomial(self):
        x = sympy.symbols('x')
        f = x**2
        a = 0
        b = 1
        tol = 1e-6
        n = 2 # may be changed to whatever - is accurate at low n
        result = adaptive_composite_simpson(f, a, b, tol, n)
        expected = 1/3  # Integral of x^2 from 0 to 1 is 1/3
        self.assertAlmostEqual(result, expected, delta=tol)
    
    def test_trigonometric(self):
        x = sympy.symbols('x')
        f = sympy.sin(x)
        a = 0
        b = np.pi
        tol = 1e-6
        n = 4
        result = adaptive_composite_simpson(f, a, b, tol, n)
        expected = 2  # Integral of sin(x) from 0 to pi is 2
        self.assertAlmostEqual(result, expected, delta=tol)
    
    def test_exponential(self):
        x = sympy.symbols('x')
        f = sympy.exp(x)
        a = 0
        b = 1
        tol = 1e-6
        n = 6
        result = adaptive_composite_simpson(f, a, b, tol, n)
        expected = np.e - 1  # Integral of e^x from 0 to 1 is e - 1
        self.assertAlmostEqual(result, expected, delta=tol)

if __name__ == '__main__':
    unittest.main()