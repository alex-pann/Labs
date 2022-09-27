import numpy as np
import matplotlib.pyplot as plt

def divided_differences(x, f, k):
    res = 0
    for j in range(k + 1): #сумма от 0 до k
        mul = 1 #на случай если i = j
        for i in range(k + 1):
            if i != j:
                mul *= x[j] - x[i]
        res += f[j]/mul
    return res

def Newton_pol(x, f, x_0):
    div_diff = []
    for i in range(1, len(x)): # вычисляем все частичные суммы
        div_diff.append(divided_differences(x, f, i)) 
    res = f[0] # первый член полинома
    for k in range(1, len(f)):
        mul = 1
        for j in range(k):
            mul *= (x_0-x[j])
        res += div_diff[k-1]*mul
    return res


# дано
x = np.arange(0, 1.1, 0.1)
f = [0.000, 0.033, 0.067, 0.100, 0.134, 0.168, 0.203, 0.238, 0.273, 0.309, 0.346]
x_0 = 0.95

# интерполяция полиномом Ньютона:
P_N = Newton_pol(x, f, x_0)
print(P_N)

pr = np.arange(0, 1.1, 0.05)
P = Newton_pol(x, f, pr)
print(P)

# plt.plot(x, f, 'k.', markersize=8) 
# plt.plot(pr, P, '--', linewidth = 0.6, color = 'r')
# plt.show()

# интерполяция сплайнами:
