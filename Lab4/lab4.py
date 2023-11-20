import numpy as np
import matplotlib.pyplot as plt

def get_root(p, x, a, b):
  eps = 0.0001
  approx = a + (b - a) / 2
  value = polynom(approx, p, x)
  value_a = polynom(a, p, x)
  while(abs(0 - value) > eps):
    if value_a * value >= 0:
      a = approx
    else:
      b = approx
    approx = a + (b - a) / 2
    value = polynom(approx, p, x)
    value_a = polynom(a, p, x)
  return approx

def get_chebs(a, b, n):
    res = []
    for i in range(n + 1):
        res.append(((a + b) / 2) + ((b - a) / 2) * np.cos((2 * i + 1) * np.pi / (2 * (n + 1)))
    )
    return res

def f(x):
  return x**3 - 4*x**2 - 4*x + 13 - np.sin(x)

def f2(x, y):
  l = len(x)
  if l == 1:
    return y[0]
  if l > 2:
    return (f2(x[1:l], y[1:l]) - f2(x[0:l-1], y[0:l-1])) / (x[l-1] - x[0])

  return (y[1] - y[0]) / (x[1] - x[0])

def polynom(val, p, x):
  value = 0
  for i in range(n):
    add = p[i]
    for j in range(0, i):
      add *= val - x[j]
    value += add
  return value

n = 9
a, b = 4, 5
step = 4.0
x = []
y = []

f_x = []
f_y = []
for i in range(1000):
    f_x.append(step)
    f_y.append(f(step))
    step += 0.001
    
    
plt.plot(f_x, f_y)
plt.show()
x = get_chebs(a, b, n)
print("Chebs : ", x)
for i in x:
  y.append(f(i))

print(f"Interval: [{a};{b}]")
print(f"N Nodes: {n+1}")
print()
print(f"X: {x}")
print(f"Y: {y}")

koefs_straight = []
koefs_inverse = []

x_temp = []
y_temp = []
for i in range(len(x)):
  x_temp.append(x[i])
  y_temp.append(y[i])
  koefs_straight.append(f2(x_temp, y_temp))

x_temp = []
y_temp = []
for i in range(len(x)):
  x_temp.append(y[i])
  y_temp.append(x[i])
  koefs_inverse.append(f2(x_temp, y_temp))

print()
print("Straight interpolation koefs: ")
print(koefs_straight)
print("Inverse interpolation koefs: ")
print(koefs_inverse)
print()

f_x = []
f_y = []
step = 4.0
for i in range(1000):
    f_x.append(step)
    f_y.append(polynom(step, koefs_straight, x))
    step += 0.001 
plt.plot(f_x, f_y)

plt.show()

f_x = []
f_y = []
step = -1.0
for i in range(1000):
    f_y.append(step)
    f_x.append(polynom(step, koefs_inverse, y))
    step += 0.001 
plt.plot(f_x, f_y)

plt.show()
res_inverse = polynom(0, koefs_inverse, y)
res_striaght = get_root(koefs_straight,x,a,b)

print(f"Straight interpolation root = {res_striaght}")
print(f"Inverse interpolation root = {res_inverse}")

