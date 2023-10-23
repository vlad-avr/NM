import numpy as np
import sys
import math as math

#a = np.array([[2.0, 3.0, -1.0, 4.0, 5.0], [3.0, 2.0, 4.0, -1.0, 6.0], [1.0, 5.0, 6.0, 7.0, -8.0], [6.0, 4.0, 5.0, 2.0, -3.0], [4.0, -7.0, 3.0, 5.0, 2.0]])
a = np.array([[10.0, 2.0, -1.0, -3.0, 4.0], [2.0, 12.0, 3.0, 1.0, -5.0], [-1.0, 3.0, 14.0, -4.0, 6.0], [-3.0, 1.0, -4.0, 15.0, 2.0], [4.0, -5.0, 6.0, 2.0, 17.0]])
print("Matrix A : \n", a)

#b = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
b = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
print("Vector b : ", b)

def check_gauss(matr):
    print("\nChecking if this system of linear computations can be solved with Gauss method:\n")
    det = np.linalg.det(matr)
    print("\nDeterminant : ", det)
    if(det == 0):
        print("\nDeterminant is 0 -> cannot be solved\n")
        return False
    print("\n All good\n")
    return True

# a_x = np.array([[1.0,2.0,3.0],[2.,5.,5.],[3.,5.,6.]])
# b_x = np.array([1.,2.,3.])

def gauss(A, b):
    if(check_gauss(A) == False):
        return np.zeros(len(b))
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

def check_jacobi(matr, eps):
    print("\nChecking if this system of linear computations can be solved with Jacobi method:\n")    
    for i in range(len(matr)):
        if(matr[i,i] == 0):
            print("\nUnable to solve : one of diagonal elements is 0")
            return False
        s = 0
        for j in range(len(matr)):
            if(j != i):
                s += abs(matr[i,j])
        if(s > abs(matr[i,i])):
            print("\nUnable to solve : sum of absolutes of elements in row is greater than diagonal element in row ", i)
            return False         
    print("\n All good\n")  
    print("\nChecking the Theorem about convergence condition for any starting approximation for Jacobi method:\n")
    matr_upper = np.zeros((len(matr), len(matr)))
    matr_lower = np.zeros((len(matr), len(matr)))
    diag = np.zeros((len(matr), len(matr)))
    for i in range(len(matr)):
        diag[i,i] = matr[i,i]
        for j in range(i+1, len(matr)):
            matr_upper[i,j] = matr[i,j]
            matr_lower[j,i] = matr[j,i]
            
    print("\n", matr_upper, "\n", matr_lower, "\n", diag)
    upper_lower = np.add(matr_upper, matr_lower)
    print("\n", upper_lower)
    d_m = -1*np.linalg.inv(diag)
    print("\n", d_m)
    B = np.matmul(d_m, upper_lower)
    print("\nB : \n", B)
    u, sig, v = np.linalg.svd(B)
    for i in range(len(B)):
        diag[i][i] = -sig[i]
    
    B = np.add(B, diag)
    det = np.linalg.det(B)
    print("\nSignature numbers of B matrix: ", sig)
    print("\nMatrix (A1 + A2 + lD) : \n", B, "\nAnd its determinant : ", det)
    if abs(det) <= eps:
        print("\nConditions are met : Jacobi method converges for all approximations\n")
    else:
        print("\nConditions are not met : Jacobi method may not converge for all approximations")
    return True
            
    
def jacobi(A, b, eps):
    if(check_jacobi(A, eps) == False):
        return np.zeros(len(b))
    n = len(b)
    x = np.zeros(len(b))
    print("\nStarting approximation : ", x)
    while True:
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        print("\n Current approximation with Jacobi : ", x_new)
            
        #Using Frobenius norm
        frob = np.linalg.norm(x_new - x)
        if frob < eps:
            print("\n Frobenius norm : ", frob," is less than ", eps, " -> result obtained\n")
            return x_new
        print("\n Frobenius norm : ", frob, " is not less than ", eps, " -> continuing iterations\n")
        x = x_new
    
def get_error(x_actual, x_approx):
    print("\nCalculating error between actual result (from gauss method) and approximated result (from jacobi method) : \n")
    er = np.linalg.norm(x_approx - x_actual)/np.linalg.norm(x_approx)
    print("\nError : ", er)
    

def get_cond(matr):
    u, sig, v = np.linalg.svd(matr)
    c = np.linalg.cond(matr)
    print("\nCondition number of a matrixx : ", c)
    print("\n", max(abs(sig)), " ", min(abs(sig)))
    if(c <= (max(abs(sig))/min(abs(sig)))+0.001 and c >= (max(abs(sig))/min(abs(sig)))-0.001):
        print("\n Condition number ", c, " is more or equal to ", max(abs(sig))/min(abs(sig)))
        
get_cond(a);        
gauss_solution = gauss(a, b)
print("Solution (Gauss): ", gauss_solution)
jacobi_solution = jacobi(a, b, 0.001)
print("Solution (Jacobi): ", jacobi_solution)
get_error(gauss_solution, jacobi_solution)