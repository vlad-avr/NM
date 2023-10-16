import numpy as np
import sys
import math as math

A = np.array([[2, 3, -1, 4, 5], [3, 2, 4, -1, 6], [1, 5, 6, 7, -8], [6, 4, 5, 2, -3], [4, -7, 3, 5, 2]])
print("Matrix A : ", A)

b = np.array([10, 15, 20, 25, 30])
print("Vector b : ", b)

def check_matr(matr):
    print("\nChecking if this system of linear computations can be solved:\n")
    det = np.linalg.det(matr)
    print("\nDeterminant : ", det)
    if(det == 0):
        print("\nDeterminant is 0 -> cannot be solved\n")
    
check_matr(A)