import numpy as np
import math
import matplotlib.pyplot as plt

data = [92228496, 106021537, 123202624, 132164569, 151325798, 
       179323175, 203211926, 226545805, 248709873, 281421906]
x = np.arange(0, 10, 1)
h = x[1] - x[0]
n = len(x)
x_0 = 10

# интерполяция полиномом Ньютона:
def div_dif(x, f, k):
    res = 0
    for j in range(k + 1): # сумма от 0 до k
        m = 1 # на случай если i = j
        for i in range(k + 1): # цикл произведений
            if i != j:
                m *= x[j] - x[i]
        res += f[j]/m
    return res

def Newton_pol(x, f):
    div_diff = []
    for i in range(1, len(x)): # вычисляем все разделенные разности
        div_diff.append(div_dif(x, f, i)) 
    def polynom(p):
        res = f[0] # первый член полинома
        for k in range(1, len(f)):
            m = 1
            for j in range(k):
                m *= p-x[j]
            res += div_diff[k-1]*m
        return res
    return polynom


P_N = Newton_pol(x, data)
print(P_N(x_0))

Y = []
for i in range(0, 10, 1):
    Y.append(P_N(x[i]))

fig1 = plt.figure("Интерполяция полиномом Ньютона")
plt.plot(x, data, ".", color = 'b')
plt.plot(10, 308745538, '.', color = 'b')
# plt.plot(10, P_N(x_0), '.', color = 'r')
plt.plot(x, Y, '--', linewidth = 0.6, color = 'r')

# data.append(P_N(x_0))
# x = np.append(x, 10)


# plt.plot(x, data, ".")
plt.show()
