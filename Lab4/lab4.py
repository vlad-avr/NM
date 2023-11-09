import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Задане рівняння
def equation(x):
    return x**3 - 4*x**2 - 4*x + 13 - np.sin(x)

# Функція для побудови поліному Чебишова
def chebyshev_poly(n):
    x = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
    return x

# Знаходимо нулі поліному Чебишова
cheb_nodes = chebyshev_poly(10)

# Виводимо нулі поліному Чебишова
print("Чебишовські вузли:", cheb_nodes)

# Побудова графіка рівняння та поліному Чебишова
x_values = np.linspace(-2, 2, 1000)
y_values_equation = equation(x_values)
y_values_cheb = equation(cheb_nodes)

plt.plot(x_values, y_values_equation, label='Рівняння')
plt.scatter(cheb_nodes, y_values_cheb, color='red', label='Чебишовські вузли')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.title('Рівняння та його Чебишовські вузли')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Пряма інтерполяція за допомогою поліному Ньютона
def direct_interpolation(x, y, target_y):
    n = len(x)
    coefficients = np.zeros(n)
    coefficients[0] = y[0]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])

    result = coefficients[0]

    for i in range(1, n):
        result += coefficients[i] * (target_y - x[i - 1])
        coefficients[i] = result

    return result

# Обернена інтерполяція за допомогою поліному Ньютона
def inverse_interpolation(x, y, target_y):
    n = len(x)
    coefficients = np.zeros(n)
    coefficients[0] = y[0]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])

    result = coefficients[0]

    for i in range(1, n):
        result += coefficients[i] * (target_y - x[i - 1])
        coefficients[i] = result

    # Знаходимо корінь поліному методом Ньютона
    initial_guess = x[-1] + 0.1
    root = newton(lambda z: direct_interpolation(x, y, z) - target_y, initial_guess)

    return root

# Знаходимо корінь методом прямої інтерполяції
direct_root = direct_interpolation(cheb_nodes, y_values_cheb, 0)

# Знаходимо корінь методом оберненої інтерполяції
inverse_root = inverse_interpolation(cheb_nodes, y_values_cheb, 0)

print("Корінь за допомогою прямої інтерполяції:", direct_root)
print("Корінь за допомогою оберненої інтерполяції:", inverse_root)