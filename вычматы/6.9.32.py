import numpy as np
import math
import matplotlib.pyplot as plt

data = [92228496, 106021537, 123202624, 132164569, 151325798, 
       179323175, 203211926, 226545805, 248709873, 281421906]
x = np.arange(1910, 2010, 10)
h = x[1] - x[0]
n = len(x)
x_0 = 2010


# функция трехдиагональной прогонки: ( x[i] = alpha[i+1] * x[i+1] + beta[i+1] )
def TDMA(a,b,c,g):
    alpha = [-c[0] / b[0]] # начальные значения коэффициентов прогонки
    beta = [g[0] / b[0]]
    n = len(g)
    res = [0]*n

    for i in range(1, n): # рассчет коэффициентов прогонки по формулам
        alpha.append( -c[i]/(a[i]*alpha[i-1] + b[i]) )
        beta.append( (g[i] - a[i]*beta[i-1])/(a[i]*alpha[i-1] + b[i]) )

    res[n-1] = beta[n-1]

    for i in range(n-1, 0, -1): # обратная прогонка
        res[i - 1] = alpha[i - 1]*res[i] + beta[i - 1]

    return res


# введем коэффициенты сплайнов a, b, c, d:
a = data

g = [] # вектор свободных членов в СЛАУ для поиска с
for i in range(1, n-1):
    g.append( 3*(a[i-1] - 2*a[i] + a[i+1])/h**2 ) 
a_t = [1]*(n-2) #на равномерной сетке коэффициенты в СЛАУ для с получаются постоянные
b_t = [4]*(n-2)
c_t = [1]*(n-2)
c = TDMA(a_t, b_t, c_t, g) # трехдиагональная прогонка
c = np.insert(c,[0,len(c)],[0,0]) # учет естественных граничных условий

b = [] # b и d находим по формулам
for i in range(1, n):
    b.append( (a[i] - a[i-1])/h + (2*c[i] + c[i-1])*h/3 ) 

d = []
for i in range(1, n):
    d.append( (c[i] - c[i-1])/(3*h) )

# вычисляем значение функции в искомой точке
k = 8
P_S = a[k] + b[k]*(x_0-x[k]) + c[k]*(x_0-x[k])**2 + d[k]*(x_0-x[k])**3

print("Результат интерполяции сплайном: ", P_S)

# визуализация:
fig1 = plt.figure("Интерполяция трикубическим сплайном")
P_S_ = []
pr = np.arange(1910, 2000, 1)
for p in pr:
    i = math.floor((p-1910)/10)
    if i <=8:
        P_S_.append( a[i] + b[i]*(p-x[i]) + c[i]*(p-x[i])**2 + d[i]*(p-x[i])**3 )

# x = np.append(x)
# data.append(P_S)
plt.plot(x, data, ".", color = 'b')
plt.plot(2010, 308745538, '.', color = 'b')
plt.plot(2010, P_S, '.', color = 'r')
plt.plot(pr, P_S_, '--', linewidth = 0.6, color = 'r')
plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.3)
plt.show()