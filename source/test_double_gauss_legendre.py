import unittest
import numpy as np
import sympy as sp
from sympy.abc import x, y
from source.double_gauss_legendre import double_gauss_legendre


class TestDoubleGaussLegendre(unittest.TestCase):

    def test_quadratic_function(self):
        # Integral of f(x, y) = x^2 + y^2 over the square [0, 1] x [0, 1] should be 2/3
        f = lambda x, y: x ** 2 + y ** 2
        a, b = 0, 1
        c, d = 0, 1
        n, m = 5, 5  # Number of points for Gauss-Legendre quadrature
        result = double_gauss_legendre(f, a, b, c, d, n, m)
        self.assertAlmostEqual(result, 2 / 3, places=6)

    def test_exponential_function(self):
        # Integral of f(x, y) = e^(x+y) over the square [0, 1] x [0, 1]
        f = lambda x, y: np.exp(x + y)
        a, b = 0, 1
        c, d = 0, 1
        n, m = 5, 5
        expected_result = (np.exp(2) - 1 - np.exp(1) + 1)
        result = double_gauss_legendre(f, a, b, c, d, n, m)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_trigonometric_function(self):
        # Integral of f(x, y) = sin(x) * cos(y) over the square [0, pi/2] x [0, pi/2] should be 1
        f = lambda x, y: np.sin(x) * np.cos(y)
        a, b = 0, np.pi / 2
        c, d = 0, np.pi / 2
        n, m = 5, 5
        result = double_gauss_legendre(f, a, b, c, d, n, m)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_variable_limits(self):
        # Integral of f(x, y) = x^2 * y over the region 0 <= y <= 1 and 0 <= x <= 1 - y^2
        f = lambda x, y: x ** 2 * y
        a, b = 0, 1
        c = 0
        d = lambda y: 1 - y ** 2
        n, m = 5, 5
        expected_result = 1 / 12
        result = double_gauss_legendre(f, a, b, c, d, n, m)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_sympy_expr(self):
        # Integral of f(x, y) = sin(x) * exp(y) over the square [0, pi] x [0, 1]
        f = sp.sin(x) * sp.exp(y)
        a, b = 0, sp.pi
        c, d = 0, 1
        n, m = 5, 5
        expected_result = sp.integrate(sp.integrate(f, (x, a, b)), (y, c, d)).evalf()
        result = double_gauss_legendre(f, float(a), float(b), float(c), float(d), n, m)
        self.assertAlmostEqual(result, expected_result, places=6)


if __name__ == "__main__":
    unittest.main()
