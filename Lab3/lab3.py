import numpy as np
import math as m

def f1(x,y):
    return m.sin(2*x-y) - 1.2*x - 0.4

def f2(x,y):
    return 0.8*x**2 + 1.5*y**2 - 1

def df1dx(x,y):
    return 2.*m.cos(2*x-y) - 1.2

def df1dy(x,y):
    return -1.*m.cos(2*x-y)

def df2dx(x):
    return 1.6*x

def df2dy(y):
    return 3.*y