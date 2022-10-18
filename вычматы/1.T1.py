# основная теорема алгебры
# локализация корней
# дихотомия

import math
import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return x**2 + (math.tan(x))**2 - 1

def root(a, b, err):
    if (F(a) * F(b) >= 0):
        raise Exception("Initial approximation error")
    while True:
        c = (a+b)/2
        if abs(F(c)) < err:
            return c

        if (F(a) * F(c) < 0):
            b = c
        else:
            a = c


# графическое определение корней
p = np.arange(-1, 1.01, 0.01)
y_p = np.array([F(p[i]) for i in range(len(p))])
plt.plot(p, y_p)
plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.3)
plt.show()

# начальное приближение
a = 0
b = 2
err = 10e-6
x1 = root(a, b, err)
a = -2
b = 0
x2 = root(a, b, err)
print("x1 = ", x1)
print("y1 = ", math.tan(x1))
print("x2 = ", x2)
print("y2 = ", math.tan(x2))