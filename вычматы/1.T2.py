#from sympy import * 
import math
import numpy as np

def f(x):
    res = math.sin(100*x) * math.exp(-x**2) * math.cos(2*x)
    return res

def I_k(k, h):
    res = h*(f(x[k]) + f(x[k+1]))/2 
    return res

def I_k_38(k, h):
    res = h*(f(x[k-1]) + 3*f(2*x[k-1]/3 + x[k]/3) + 3*f(x[k-1]/3 +2*x[k]/3) + f(x[k]))/8
    return res

a = 0
b = 3
n = 100000
x = np.array([a + (b-a)* i / n for i in range(n)])
h = x[1] - x[0]

I = 0
I_38 = 0

for k in range (n-1):
    I += I_k(k, h)

for k in range (1, n, 1):
    I_38 += I_k_38(k,h)

print("Результат вычисления интеграла методом трапеций: ", I)
print("Результат вычисления интеграла методом 3/8: ", I_38)

#проверка
from scipy.integrate import quad  
def f( x ): 
    return np.exp(-x**2)*np.sin(100*x)*np.cos(2*x)
I_0=quad(f, 0, 3) 
print("Результат вычисления с помощью scipy: ", I_0) 