import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.optimize import newton

# Define the equation for which you are looking for the root
def equation(x):
    return x**3 - 4*x**2 - 4*x + 13 - np.sin(x)

# Function to find Chebyshev nodes
def chebyshev_nodes(n):
    nodes = [np.cos((2*k - 1) * np.pi / (2*n)) for k in range(1, n+1)]
    return nodes


# Function to plot the function and the interpolated polynomial
def plot_function_and_interpolation(equation, interp_poly, nodes, x_range, root_direct, root_inverse):
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    y_function = equation(x_values)
    y_interpolation = interp_poly(x_values)

    plt.plot(x_values, y_function, label="Original Function", linestyle="--")
    plt.plot(nodes, equation(np.array(nodes)), 'o', label="Chebyshev Nodes")
    plt.plot(x_values, y_interpolation, label="Interpolated Polynomial")
    plt.axhline(0, x_range[0], x_range[1], linestyle="--", label="Y axis")
    plt.plot(root_direct, equation(root_direct), 'o', label="Direct Root")
    plt.plot(root_inverse, equation(root_inverse), 'o', label="Root Inverse")
    plt.title("Function and Interpolated Polynomial")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Find the greatest root using inverse interpolation
def find_greatest_root_inverse_interpolation():
    n = 10
    cheb_nodes = chebyshev_nodes(n)
    f_values = equation(np.array(cheb_nodes))
    interp_poly = lagrange(cheb_nodes, f_values)

    # Find the root using inverse interpolation
    initial_guess = 4.0
    root_inverse_interp = newton(interp_poly, initial_guess)
    
    return root_inverse_interp

# Find the greatest root using direct interpolation
def find_greatest_root_direct_interpolation():
    initial_guess = 5.0
    root_direct_interp = newton(equation, initial_guess)
    
    return root_direct_interp

# Main program
greatest_root_inverse_interp = find_greatest_root_inverse_interpolation()
greatest_root_direct_interp = find_greatest_root_direct_interpolation()

print("Greatest root using inverse interpolation:", greatest_root_inverse_interp)
print("Greatest root using direct interpolation:", greatest_root_direct_interp)

# Plot the function and the interpolated polynomial
x_range = (-2, 5)
n = 10
cheb_nodes = chebyshev_nodes(n)
f_values = equation(np.array(cheb_nodes))
interp_poly = lagrange(cheb_nodes, f_values)
plot_function_and_interpolation(equation, interp_poly, cheb_nodes, x_range, greatest_root_direct_interp, greatest_root_inverse_interp)