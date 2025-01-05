# **Numerical-Integration-Toolbox**
 A Python package for numerical integration, providing a collection of easy-to-use functions for accurately approximating definite integrals.

## **Methods**
The following is a list of the numerical integrations that can be found in the source folder:
- `adaptive_composite_simpson.py`: Implements the adaptive composite Simpson's rule, which adjusts interval sizes to improve accuracy for functions with varying smoothness.
- `adaptive_midpoint.py`: Implements the adaptive midpoint rule, which dynamically adjusts the partition of the integration interval to enhance accuracy where the function changes more rapidly.
- `adaptive_simpson.py`: Implements the adaptive Simpson's rule, combining Simpson's rule with adaptive interval adjustments for better integration of non-uniform functions.
- `adaptive_trapezoid.py`: Implements the adaptive trapezoidal rule, which modifies interval sizes based on the function's behaviour to increase accuracy for complex integrands.
- `composite_midpoint.py`: Implements the composite midpoint rule, which divides the integration interval into subintervals and applies the midpoint rule to each for improved accuracy.
- `composite_simpson.py`: Implements the composite Simpson's rule, using multiple applications of Simpson's rule over subintervals to enhance the approximation of definite integrals.
- `composite_trapezoid.py`: Implements the composite trapezoidal rule, applying the trapezoidal rule over subdivided intervals to refine the precision of numerical integration.
- `double_gauss_legendre.py`: Implements the double Gauss-Legendre quadrature method, using orthogonal polynomials to compute integrals over complex functions and intervals accurately.
- `gauss_legendre.py`: Implements the Gauss-Legendre quadrature method, approximating definite integrals using points and weights derived from Legendre polynomials.
- `midpoint.py`: Implements the midpoint rule for numerical integration, approximating the area under a curve using each interval's midpoint.
- `romberg.py`: Implements Romberg's method, which refines the trapezoidal rule using Richardson extrapolation to achieve higher precision in numerical integration.
- `simpson.py`: Implements Simpson's rule, a numerical integration technique that approximates the integral using quadratic polynomials.
- `trapezoidal.py`: Implements the trapezoidal rule, estimating the integral by approximating the region under the curve as a series of trapezoids.
