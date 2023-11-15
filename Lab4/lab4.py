import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.plotting import plot
from sympy.abc import x

np.set_printoptions(precision=4, suppress=True)

f = "x**3 - 4*x**2 - 4*x + 13 - sin(x)"
func = sp.sympify(f)

roots = sp.nsolve(func, x, 5, dict=True)

points_on_graph = list()
for root in roots:
    point = {
        'args': [root[x], 0],
        'color': "black",
        'ms': 5,
        'marker': "o"
    }
    points_on_graph.append(point)

print("f(x): ")
sp.pprint(func)
print("\n f(x) root : ", roots)

#Interval
a = 0
b = 5

# Number of nodes
m = 10
# Polynomial degree
n = m - 1


def get_chebs(a, b, n):
    res = []
    for i in range(n + 1):
        res.append(((a + b) / 2) + ((b - a) / 2) * sp.cos((2 * i + 1) * sp.pi / (2 * (n + 1))).evalf()
    )
    return res

def get_divided_differences(n, func, ch_nodes):
    res = np.ndarray(shape=(n + 1, n + 1), dtype=float)

    for j in range(n + 1):
        for j in range(n + 1):
            res[j][j] = 0

    for i in range(n + 1):
        all_zero = True
        for j in range(n + 1 - i):
            if i == 0:
                res[j][i] = func.subs(x, ch_nodes[j])
            else:
                res[j][i] = ((res[j + 1][i - 1] - res[j][i - 1]) / (ch_nodes[j + i] - ch_nodes[j]))
            if abs(res[j][i]) > np.power(10., -16):
                all_zero = False
        if all_zero:
            break
    return res

def forward_interpolation(divided_differences, n):
    polynom_forward = 0.
    for k in range(n + 1):
        term_k = divided_differences[0][k]
        for i in range(k):
            term_k *= (x - ch_nodes[i])
        polynom_forward += term_k
    return sp.poly(polynom_forward).as_expr()

def backward_interpolation(divided_differences, n):
    polynom_rev = 0.
    for k in range(n + 1):
        term_k = divided_differences[n - k][k]
        for i in range(k):
            term_k *= (x - ch_nodes[n - i])
        polynom_rev += term_k
    return sp.poly(polynom_rev).as_expr()

ch_nodes = get_chebs(a, b, m)

for zero in ch_nodes:
    point = {
        'args': [zero, func.subs(x, zero)],
        'color': "purple",
        'ms': 5,
        'marker': "o",
        'label': 'bib'
    }
    points_on_graph.append(point)

print("\n Chbyshov zeros on [", a, ",", b, "] interval : ")
print(ch_nodes)

divided_differences = get_divided_differences(n, func, ch_nodes)
print("\n\nDivided Differences : \n")
print(divided_differences)


forward_pol = forward_interpolation(divided_differences, n)
print("\nResult using forward interpolation :")
sp.pprint(forward_pol)

backward_pol = backward_interpolation(divided_differences, n)
print("\nResult using backward interpolation :")
sp.pprint(backward_pol)

forward_solution = sp.nsolve(forward_pol, x, 5, dict=True)
for root in forward_solution:
    point = {
        'args': [root[x], 0],
        'color': "orange",
        'ms': 5,
        'marker': "o"
    }
    points_on_graph.append(point)

backward_solution = sp.nsolve(backward_pol, x, 4, dict=True)
for root in backward_solution:
    point = {
        'args': [root[x], 0],
        'color': "yellow",
        'ms': 5,
        'marker': "o"
    }
    points_on_graph.append(point)

print("\nGreatest root of P(x) (forward) : ", forward_solution)
print("\nGreatest root of P(x) (backward) : ", backward_solution)

plt.style.use('_mpl-gallery')
plot_poly = plot(func, line_color="red", label="Function", legend=True, xlim=(-15, 15), ylim=(-15, 15), markers=points_on_graph, show=False)
plot_poly.append(plot(forward_pol, line_color="green", label="Forward Interpolation", show=False)[0])
plot_poly.append(plot(backward_pol, line_color="blue", label="Backward Interpolation", show=False)[0])
plot_poly.show()
