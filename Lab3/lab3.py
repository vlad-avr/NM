import numpy as np
import math as m

def modified_newton_method(f, jacobian, x0, tol=1e-6, max_iter=100):
    x = x0

    for i in range(max_iter):
        f_x = f(x)
        jacobian_x = jacobian(x)
        delta_x = -np.dot(np.linalg.inv(jacobian_x), f_x)
        x = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            return x
    print("NOT FOUND\n")
    return x

def f(x):
    f1 = m.sin(2*x[0] - x[1]) - 1.2*x[0] - 0.4
    f2 = 0.8*x[0]**2 + 1.5*x[1]**2 - 1
    return np.array([f1, f2])

def jacobian_matrix(x):
    J = np.array([[2.*m.cos(2*x[0] - x[1]) - 1.2, -1.*m.cos(2*x[0]-x[1])], [1.6*x[0], 3.*x[1]]])
    return J

# Initial guess
x0 = np.array([0.5, 0.5])

result = modified_newton_method(f, jacobian_matrix, x0)

print("Solution:", result)
print("\nResut: f(x) = ", f(result))