import numpy as np
import PIL
from PIL import Image

def get_pixel_map(path):
    img = Image.open(path)    
    img = img.convert("L")
    cols, rows = img.size
    print("\n Width : ", cols, " Height : ", rows)
    p_map = list(img.getdata())
    p_map = [p_map[i * cols:(i+1) * cols] for i in range(rows)]
    p_map = np.array(p_map)
    return p_map

x_1 =  get_pixel_map("D:/python/NM/MS/x1.bmp")
y_1 = get_pixel_map("D:/python/NM/MS/y1.bmp")

def get_z(matr, matr_inv):
    return np.eye(len(matr), dtype=int) - np.matmul(matr_inv, matr)

def greville(A):
    transposed = False
    if(len(A) > len(A[0])):
        transposed = True
        A = np.transpose(A)
    A_inv = list()
    for i in range(len(A)):
        cur_a = np.transpose(A[i])
        cur_a_inv = 0
        scal = np.transpose(cur_a) * cur_a
        if(scal == 0):
            A_inv.append(cur_a) 
        else:
            A_inv.append(cur_a/scal)
    
    
    