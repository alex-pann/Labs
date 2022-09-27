import numpy as np
import math
import matplotlib.pyplot as plt

# дано:
x = np.arange(0, 1.1, 0.1)
f = [0.000, 0.033, 0.067, 0.100, 0.134, 0.168, 0.203, 0.238, 0.273, 0.309, 0.346]
x_0 = 0.95
n = len(x)
h = 0.1 #шаг сетки
df_11 = 15
df_4 = 0.1

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

def Newton_pol(x, f, x_0):
    div_diff = []
    for i in range(1, len(x)): # вычисляем все разделенные разности
        div_diff.append(div_dif(x, f, i)) 
    res = f[0] # первый член полинома
    for k in range(1, len(f)):
        m = 1
        for j in range(k):
            m *= (x_0-x[j])
        res += div_diff[k-1]*m
    return res

P_N = Newton_pol(x, f, x_0)
R_N = (h**n)/n * df_11 # оценка погрешности
print("Результат интерполяции полиномом Ньютона: ", round(P_N, 11), "±", round(R_N, 11))

# визуализация:
fig1 = plt.figure("Интерполяция полиномом Ньютона")
pr = np.arange(0, 1.01, 0.01)
P = Newton_pol(x, f, pr)
plt.plot(x, f, 'k.', markersize=8) 
plt.plot(pr, P, '--', linewidth = 0.6, color = 'r')
plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.3)



#----------------------------------------------------------------------------------


# интерполяция сплайнами:
# S = a + b(x-xi) + c(x-xi)^2 + d(x-xi)^3
# ai = f(xi)
# bi = (ai - a(i-1))/h + (2ci + c(i-1))h/3
# c(i-1) + 4ci + c(i+1) = 3(a(i+1) - 2ai + a(i-1))/h^2
# di = (ci - c(i-1))/3h

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
a = f

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
k = math.floor(x_0 * 10)
P_S = a[k] + b[k]*(x_0-x[k]) + c[k]*(x_0-x[k])**2 + d[k]*(x_0-x[k])**3
R_S = (df_4 * (x_0 - x[k])**2 * (x_0 - x[k+1])**2 ) / (2*3*4) # оценка погрешности
print("Результат интерполяции сплайном: ", round(P_S, 8), "±", round(R_S, 8))

# визуализация:
fig1 = plt.figure("Интерполяция трикубическим сплайном")
P_S = []
pr = np.arange(0, 1, 0.01)
for p in pr:
    i = math.floor(p*10)
    if i <=9:
        P_S.append( a[i] + b[i]*(p-x[i]) + c[i]*(p-x[i])**2 + d[i]*(p-x[i])**3 )

plt.plot(x, f, 'k.', markersize=8) 
plt.plot(pr, P_S, '--', linewidth = 0.6, color = 'r')
plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.3)
plt.show()