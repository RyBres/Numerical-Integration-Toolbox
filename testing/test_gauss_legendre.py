import unittest
from math import sin, pi, exp
from sympy import symbols, sin as sym_sin, exp as sym_exp
from sympy.utilities.lambdify import lambdify
from source.gauss_legendre import gauss_legendre

class TestGaussLegendre(unittest.TestCase):

    def test_lambda_function_sin(self):
        # Integral of sin(x) from 0 to pi should be 2
        result = gauss_legendre(lambda x: sin(x), 0, pi, 5)
        self.assertAlmostEqual(result, 2, places=5)

    def test_lambda_function_exp(self):
        # Integral of exp(x) from 0 to 1 should be e - 1
        result = gauss_legendre(lambda x: exp(x), 0, 1, 5)
        self.assertAlmostEqual(result, exp(1) - 1, places=5)

    def test_sympy_expression_sin(self):
        x = symbols('x')
        expr = sym_sin(x)
        # Integral of sin(x) from 0 to pi should be 2
        result = gauss_legendre(expr, 0, pi, 5)
        self.assertAlmostEqual(result, 2, places=5)

    def test_sympy_expression_exp(self):
        x = symbols('x')
        expr = sym_exp(x)
        # Integral of exp(x) from 0 to 1 should be e - 1
        result = gauss_legendre(expr, 0, 1, 5)
        self.assertAlmostEqual(result, exp(1) - 1, places=5)

if __name__ == '__main__':
    unittest.main()