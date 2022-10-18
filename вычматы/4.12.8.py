import math
import numpy as np
import matplotlib.pyplot as plt

err = 10e-5

def F(x):
    return x*math.exp(-x**2)

def dF(x):
    return math.exp(-x**2) + (-2*x**2) * math.exp(-x**2)

def P1(x, p):
    return p*math.exp(x**2)

def P2(x, p):
    return math.sqrt(math.log(x/p))

def root(F, a, b, err):
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

def MSI(F, p, err, init_x):
    x = init_x
    while True:
        x_1 = F(x, p)
        if abs(x_1 - x) < err:
            return x_1
        else:
            x = x_1

# # построим график функции
# x = np.arange(0, 5, 0.01)
# y = np.array([F(x[i]) for i in range(len(x))])
# fig1 = plt.figure()
# plt.plot(x, y)
# plt.show()

# найдем пик (дихотомия, как в T1)
a = 0.5
b = 1
err_0 = 10e-9
x_c = root(dF, a, b, err)
h = F(x_c)
print("Пик:  x_c = ", x_c, ", h = ", F(x_c))

# поиск ширины функции на полувысоте
err = 10e-3
# p = h/2 # F*= F - p
p = F(1/math.sqrt(2))/2
x_1 = MSI(P1, p, err, p)
x_2 = MSI(P2, p, err, 0.8)
D = x_2 - x_1
print("x_1 = ", round(x_1, 3), "x_2 = ", round(x_2, 3))
print("Ширина функции на полувысоте: ", round(D, 3))


