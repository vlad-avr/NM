import numpy as np
import sys
import math as math

a = np.array([[2.0, 3.0, -1.0, 4.0, 5.0], [3.0, 2.0, 4.0, -1.0, 6.0], [1.0, 5.0, 6.0, 7.0, -8.0], [6.0, 4.0, 5.0, 2.0, -3.0], [4.0, -7.0, 3.0, 5.0, 2.0]])
print("Matrix A : \n", a)

b = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
print("Vector b : ", b)

def check_matr(matr):
    print("\nChecking if this system of linear computations can be solved:\n")
    det = np.linalg.det(matr)
    print("\nDeterminant : ", det)
    if(det == 0):
        print("\nDeterminant is 0 -> cannot be solved\n")

# a_x = np.array([[1.0,2.0,3.0],[2.,5.,5.],[3.,5.,6.]])
# b_x = np.array([1.,2.,3.])

def gauss(A, b):
    matr_len = len(b)
    for i in range(matr_len):
        p = np.identity(matr_len)
        m = np.identity(matr_len)
        print("\nCurrent A on step ", i, " : \n", A)
        print("\nCurrent b on step ", i, " : \n", b)
        max_row = np.argmax(np.abs(A[i:matr_len, i])) + i
        print("\n Max Row Ind : ", max_row)
        p[[i, max_row]] = p[[max_row, i]]
        print("\nP matrix : \n", p)
        A = np.matmul(p, A)
        b = np.matmul(p, b.transpose())
        for j in range(matr_len):
            if(j < i):
                m[j, i] = 0
            elif(j == i):
                m[j, i] = 1/A[j, i]
            else :
                m[j, i] = -A[j, i]/A[i, i]
        
        print("\nM matrix : \n", m)
        A = np.matmul(m, A)
        b = np.matmul(m, b.transpose())
        
    print("\n Result A : \n", A)
    print("\n Result b : ", b)
    x = np.zeros(matr_len)
    for i in range(matr_len - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, matr_len):
            x[i] -= A[i, j] * x[j]
    
    return x

# s = gauss(a_x, b_x)
# print("Example Solution : ", s)
        
my_solution = gauss(a, b)
print("My Solution : ", my_solution)