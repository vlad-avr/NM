import numpy as np
import sys
import math as math

#Epsilon that defines how precise our answer to be
EPS = 0.0001

#Equation
equation = np.array([-16, 20, -6, 1, 1])

#Derivative

der = np.array([20, -12, 3, 4])

#Start\end of interval
a = 0.0
b = 3.0

#For a posteriori evaluation of iterations quantity
step_count = 0


def compute(val, equat):
    res = equat[0]
    for i in range(1, len(equat)):
        res += equat[i]*(val**i)
    return res

def compute_bisection(start, end):
    global step_count
    step_count += 1
    print("\nStart: " + str(start) + " End: " + str(end))
    f_a = compute(start, equation)
    f_b = compute(end, equation)
    if(f_a*f_b >= 0):
        print("No roots in this interval")
        return
    
    mid = (start+end)/2.0
    f_mid = compute(mid, equation)
    print("f(" + str(mid) + ") = " + str(f_mid))
    if(abs(f_mid) <= EPS or abs(start - mid) <= EPS or abs(mid - end) <= EPS):
        return mid
    else:
        if (f_mid * f_a <= 0):
            return compute_bisection(start, mid)
        else:
            return compute_bisection(mid, end)
        
def compute_newton(val):
    if(val < 0.781):
        print("No roots in this interval")
        return
    global step_count
    step_count += 1
    f_val = compute(val, equation)
    print("\nf(" + str(val) + ") = " + str(f_val))
    if(abs(f_val) <= 2*EPS):
        return val
    else:
        print("Value is not the root -> resuming search")
        f_der = compute(val, der)
        print("f'(" + str(val) + ") = " + str(f_der))
        div = f_val/f_der
        new_val = val - div
        print("n+1 value : ", new_val, " = [n value :]", val, " - [f value/ f derivative value at n value]", div)
        return compute_newton(new_val)
    
def compute_iteration(val, q):
    global step_count
    step_count += 1
    print("\nX = " + str(val))
    #new_x = (equation[0] + equation[2]*val**2 + equation[3]*val**3 + equation[4]*val**4)/(-1.0*equation[1])
    new_x = (-1.0*equation[0])/(equation[1] + equation[2]*val + equation[3]*val**2 + equation[4]*val**3)
    print("New X = ", new_x)
    if(abs(new_x - val) < (1-q)*EPS/q):
        return new_x
    else:
        return compute_iteration(new_x, q)
print("Dichotomy method: \n")
print("A priori step count: n >= ", math.floor(math.log2((b-a)/EPS)) + 1)
print("\nResult: ", compute_bisection(a, b))
print("A posteriori step count: n = ", step_count)
step_count = 0
print("\nNewton's method: \n")
m1 = 14.3634
m2 = 24
q = ((m2 * abs(b/2.0 - 1.0))/(2.0*m1))
print("q = ", q, "\n")
print("A priori step count: n >= ", math.floor(math.log2(((math.log(abs(b/2.0 - 1.0)/EPS))/math.log(1/q)))+ 1.0) + 1)
print("\nResult: ", compute_newton(b/2.0))
print("A posteriori step count: n = " , step_count)
step_count = 0
print("\nSimple Iteration method: \n")
# max of fi`
q = 0.24681
print("q = ", q, "\n")
print("A priori step count: n >= " + str(math.floor((math.log(abs(0.24-a)/((1-q)*EPS))/math.log(1/q))) + 1))
print("\nResult: ", compute_iteration(a, q))
print("A posteriori step count: n = " , step_count)
step_count = 0