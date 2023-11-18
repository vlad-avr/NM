def f(x):
    return 1 / (2 + x)

def f_second_der(x):
    return 2/(2 + x)**3

def midpoint_rule(a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        xi = a + (i + 0.5) * h
        result += f(xi)
    result *= h
    return result

def estimate_error(a, b, n):
    M2 = max(abs(f_second_der(x)) for x in range(a, b + 1))
    h = (b - a) / n
    return (M2 * (b - a) * h**2) / 24

def integrate_midpoint_rule(a, b, tolerance):
    n = 1
    integral = midpoint_rule(a, b, n)
    while True:
        n *= 2
        new_integral = midpoint_rule(a, b, n)
        error = abs(new_integral - integral)
        if error < tolerance:
            break
        integral = new_integral
    return new_integral

a = 1
b = 5
eps = 0.05

# Result using regular integral computation : 0.8473

result = integrate_midpoint_rule(a, b, eps)
print("The result of the integral is: ", result)