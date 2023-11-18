import sympy as sp
import numpy as np


def read_file(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    input_data = []
    for line in lines:
        values = line.strip().split()
        row = []
        for value in values:
            row.append(float(value))
        input_data.append(row)
    return np.array(input_data).T

def get_derivative(y_vec, b_vec, b_values):
    derivs = []
    for y_i in y_vec:
        for b_i in b_vec:
            d = sp.diff(y_i, b_i)
            d = d.subs(b_values)
            derivs.append(d)

    cols_n = len(b_vec)
    der_matr = []
    for i in range(0, len(derivs), cols_n):
        der_matr.append(derivs[i:i + cols_n])   
    return sp.Matrix(der_matr)


def get_u_matr(a_matr, b_matr, u_matr, h):
    b_arrayed = np.array(b_matr.tolist())
    k1 = h * (np.dot(a_matr, u_matr) + b_arrayed)
    k2 = h * (np.dot(a_matr, u_matr + k1 / 2) + b_arrayed)
    k3 = h * (np.dot(a_matr, u_matr + k2 / 2) + b_arrayed)
    k4 = h * (np.dot(a_matr, u_matr + k3) + b_arrayed)
    return u_matr + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def get_y(a_matr, y_cur, h):
    k1 = h * np.dot(a_matr, y_cur)
    k2 = h * np.dot(a_matr, y_cur + k1 / 2)
    k3 = h * np.dot(a_matr, y_cur + k2 / 2)
    k4 = h * np.dot(a_matr, y_cur + k3)
    return y_cur + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def init_matr():
    c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1 c2 c3 c4 m1 m2 m3')
    matr = [
        [0, 1, 0, 0, 0, 0],
        [-(c2 + c1) / m1, 0, c2 / m1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [c2 / m2, 0, -(c2 + c3) / m2, 0, c3 / m2, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, c3 / m3, 0, -(c4 + c3) / m3, 0]
    ]
    return sp.Matrix(matr)

def approximate(y_matr, params, beta_symbols, beta_values, eps, h=0.2):
    a_matrix = init_matr().subs(params)
    beta_vector = np.array([beta_values[beta_symbols[0]], beta_values[beta_symbols[1]], beta_values[beta_symbols[2]]])
    while True:
        a_complete = np.array((a_matrix.subs(beta_values)).tolist())
        u_matr = np.zeros((6, 3))
        delta_integral = 0
        integral_part_inverse = np.zeros((3, 3))
        integral_part_mult = np.zeros((1, 3))
        y_approximation = y_matr[0]
        for i in range(len(y_matr)):
            b_derivative_matr = get_derivative(a_matrix * sp.Matrix(y_approximation), beta_symbols, beta_values)

            integral_part_inverse = (integral_part_inverse + np.dot(u_matr.T, u_matr)).astype('float64')

            integral_part_mult = (integral_part_mult + np.dot(u_matr.T, y_matr[i] - y_approximation)).astype('float64')

            delta_integral = delta_integral + np.dot(y_matr[i] - y_approximation, y_matr[i] - y_approximation)
            
            u_matr = get_u_matr(a_complete, b_derivative_matr, u_matr, h)
            y_approximation = get_y(a_complete, y_approximation, h)
            
        integral_part_inverse = integral_part_inverse * h
        integral_part_mult = integral_part_mult * h
        delta_integral = delta_integral * h
        
        delta_beta = np.dot(np.linalg.inv(integral_part_inverse), integral_part_mult.flatten())
        beta_vector = beta_vector + delta_beta
        
        beta_values = {
            beta_symbols[0]: beta_vector[0],
            beta_symbols[1]: beta_vector[1],
            beta_symbols[2]: beta_vector[2]
        }
        print("Current approximated values : ", beta_vector)
        print("Delta : ", delta_integral)
        if delta_integral < eps:
            return beta_values
        print("Delta is greater than ", eps, " -> next iteration")
        
        
input = read_file('D:\\python\\NM\\MS_Lab3\\y1.txt')
c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1 c2 c3 c4 m1 m2 m3')
to_approx = {c1: 0.1, m1: 11, m2: 23}
result = approximate(input, {c2: 0.3, c3: 0.2, c4: 0.12, m3: 18}, [c1, m1, m2], to_approx, 1e-6)
print("Approximation : ", result)