import numpy as np
import sys
import math as math
import random as rnd
import time

rnd.seed(time.time())

class Matrix:
    matr = [[]]
    def __init__(self, N, max, min):
        self.matr = np.zeros(shape=(N,N), dtype=float)
        for i in range(N):
            for j in range(N):
                self.matr[i][j] = rnd.randint(min, max)
    
    def printf(self):
        print('\n'.join([''.join(['{:8}'.format(item) for item in row]) for row in self.matr]))    
class Vector:
    vec = []
    def __init__(self, N, max, min):
        self.vec = np.zeros(shape=(N), dtype=float)
        for i in range(N):
            self.vec[i] = rnd.randint(min, max)
    
    def printf(self):
        print('\n','\t'.join(['{:5}'.format(item) for item in self.vec]) )
        
        
matrix = Matrix(5, 10, 1)     
matrix.printf()
vec = Vector(5, 10, 1)
vec.printf()
    
            