import numpy as np
import math

def f(x):
    return 1 / (2 + x)

def f_second_der(x):
     return 2/(2 + x)**3
 
def get_h(a, b, eps):
    M2 = max(abs(f_second_der(x)) for x in range(a, b + 1))
    print("Maximum value of 2nd derivative : ", M2)
    h = np.sqrt((24*eps)/(M2 * (b-a)))
    return h

def midpoint_rule(a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        xi = a + (i + 0.5) * h
        result += f(xi)
    result *= h
    return result


a = 1
b = 5
h = get_h(a, b, 0.05)
print("h = ", h)
n = math.floor((b-a)/h)
print("n = ", n)
result = midpoint_rule(a, b, n)
print(f"The result of the definite integral is: {result}")
