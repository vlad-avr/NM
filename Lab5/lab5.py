import numpy as np
import math as math

def f(x):
    return 1 / (2 + x)

def f_second_der(x):
    return 2/(2 + x)**3


def get_h(a, b, eps):
    M2 = max(abs(f_second_der(x)) for x in range(a, b + 1))
    print(M2)
    h = np.sqrt((24*eps)/(M2 * (b-a)))
    return h

def get_f_interval(a, n, h):
    f_x = []
    cur = a
    next = a+h
    for i in range(1, n):
        f_x.append(f((cur + next)/2))
        cur = next
        next += h
    return f_x
        

def calculate_midpoint(f_x, h):
    res = 0
    for i in f_x:
        res += i
    res *= h
    return res       
        

a = 1
b = 5
eps = 0.05

# Result using regular integral computation : 0.8473
#h = math.floor(get_h(a, b, eps))
h = 0.1
n = (int)((b-a)/h)
f_x = get_f_interval(a, n, h)
result = calculate_midpoint(f_x, h)
print("The result of the integral is: ", result)