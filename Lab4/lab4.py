import numpy as np
import matplotlib.pyplot as plt

# Function to find the roots of the Chebyshev polynomial
def chebyshev_roots(n):
    return np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))

# Function to evaluate the given non-linear equation
def equation(x):
    return x**3 - 4*x**2 - 4*x + 13 - np.sin(x)

# Newton's interpolation polynomial coefficients
def newton_coefficients(x_nodes, y_nodes):
    n = len(x_nodes)
    coefficients = y_nodes.copy()

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coefficients[i] = (coefficients[i] - coefficients[i - 1]) / (x_nodes[i] - x_nodes[i - j])

    return coefficients

# Newton's interpolation polynomial
def newton_interpolation(x, x_nodes, coefficients):
    n = len(x_nodes)
    result = coefficients[-1]

    for i in range(n - 2, -1, -1):
        result = result * (x - x_nodes[i]) + coefficients[i]

    return result

# Number of interpolation points
n_points = 10

# Chebyshev roots as interpolation points
interpolation_points = chebyshev_roots(n_points)

# Evaluate the function at interpolation points
function_values = equation(interpolation_points)

# Compute Newton's interpolation coefficients
coefficients = newton_coefficients(interpolation_points, function_values)
print(coefficients)

# Generate a fine grid of x values for plotting
x_values = np.linspace(-1, 5, 1000)

equation_values = equation(x_values)

# Evaluate the interpolation polynomial at the fine grid
interpolation_result = newton_interpolation(x_values, interpolation_points, coefficients)

# Find roots of the interpolation polynomial (greatest root corresponds to the root of the original equation)
roots = np.roots(np.flip(coefficients))

print(roots)

# Find the greatest root (real part) among the roots
greatest_root = max([root.real for root in roots])

# Plotting
plt.plot(x_values, interpolation_result, label="Interpolation Polynomial")
plt.plot(x_values, equation_values, label = "Function to approximate", linestyle = "--")
plt.scatter(interpolation_points, function_values, color='red', label='Interpolation Points')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton\'s Interpolation for the Non-linear Equation')
plt.legend()
plt.grid(True)
plt.show()

print(f"The greatest root of the equation is: {greatest_root:.6f}")