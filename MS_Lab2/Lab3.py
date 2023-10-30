import numpy as np
from PIL import Image

def get_pixel_map(path):
    img = Image.open(path)
    img = img.convert("L")
    pixel_map = list(img.getdata())
    cols, rows = img.size
    print("\n Size : ", rows, "x", cols)
    pixel_map = [pixel_map[i * cols:(i + 1) * cols] for i in range(rows)]
    return pixel_map

def convert_to_image(path, pixel_map):
    img = Image.new('L', (len(pixel_map[0]), len(pixel_map)))
    img.putdata([pixel for row in pixel_map for pixel in row])
    img.save(path)

def mult(*args):
    if len(args) == 0:
        return 0
    if len(args) == 1:
        return args[0]

    result = np.dot(args[0], args[1])
    for i in range(2, len(args)):
        result = np.dot(result, args[i])
    return result


def greville(A, eps):
    scalar = mult(A[0].T, A[0])
    inverse_A = None
    if scalar == 0:
        inverse_A = np.vstack(A[0])
    else:
        inverse_A = np.vstack(A[0] / scalar)
        
    cur_A = np.array([A[0]])
    n = A.shape[0]
    for i in range(1, n):
        a_i = A[i].reshape(-1, 1)
        z = np.identity(cur_A.shape[1]) - mult(inverse_A, cur_A)
        cur_A = np.vstack((cur_A, A[i]))
        denom = mult(a_i.T, z, a_i)[0, 0]
        if np.abs(denom) < eps:
            r_A = mult(inverse_A, inverse_A.T)
            denom_r = 1 + mult(a_i.T, r_A, a_i)
            inverse_A = inverse_A - mult(z, a_i, a_i.T, inverse_A) / denom_r
            col = mult(r_A, a_i) / denom_r
            inverse_A = np.hstack((inverse_A, np.vstack(col)))
        else:
            inverse_A = inverse_A - mult(z, a_i, a_i.T, inverse_A) / denom
            col = mult(z, a_i) / denom
            inverse_A = np.hstack((inverse_A, np.vstack(col)))

    return inverse_A

def moore(A, eps):
    d = 1
    width = A.shape[0]
    height = A.shape[1]
    cur = np.zeros([height, width])
    if width < height:
        while True:
            prev = cur.copy()
            cur = mult(A.T, np.linalg.inv(mult(A, A.T) + (d**2) * np.identity(width)))
            if np.sqrt(np.sum((prev - cur) ** 2)) < eps:
                break
            d /= 2
    else:
        while True:
            prev = cur.copy()
            cur = mult(np.linalg.inv(mult(A.T, A) + (d**2) * np.identity(height)), A.T)
            if np.sqrt(np.sum((prev - cur) ** 2)) < eps:
                break
            d /= 2
    return cur

def transform(operator, x, y):
    result = mult(operator, x)
    print("\nDistance:", np.sqrt(np.sum(((y - result)**2))))
    return result

def get_operator(input, input_inv, output):
    z = np.identity(input.shape[0]) - mult(input, input_inv)
    v = np.zeros((output.shape[0], input.shape[0]))
    return mult(output, input_inv) + mult(v, z.T)

input =  np.array(get_pixel_map("D:/python/NM/MS_Lab2/x1.bmp"))
output = np.array(get_pixel_map("D:/python/NM/MS_Lab2/y1.bmp"))
append_vec = np.ones((1, input.shape[1]))
input = np.vstack((input, append_vec))

inverse_greville = greville(input, 0.001)
inverse_moore = moore(input, 0.000001)

print("\nResult using Greville method of finding pseudo-inverse:\n")
operator = get_operator(input, inverse_greville, output)
g_res = transform(operator, input, output)
convert_to_image("D:/python/NM/MS_Lab2/res_greville.bmp", g_res)
print("\nResult using Moore-Penrose method of finding pseudo-inverse:\n")
operator = get_operator(input, inverse_moore, output)
m_res = transform(operator, input, output)
convert_to_image("D:/python/NM/MS_Lab2/res_moore.bmp", m_res)
