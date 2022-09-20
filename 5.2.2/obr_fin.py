from asyncio import Handle
from ctypes.wintypes import HGDIOBJ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def func1(x, a, b, c):
    return a + b/(x+c)

def func2(x, k):
    return k*x


data = pd.read_excel("C:\\Users\\hp\\0-Alex\\BOTAY\\0-TOTAL\\5\\ФИЗИКА\\Лабы\\2.2\\data_obr.xlsx")
x = data.x #градуировка по неоновой лампе
A = data.A
x1 = data.x1 # градуировка по ртутной лампе
A1 = data.A1
x_s = data.x_s # общая таблица данных
A_s = data.A_s

# !!!! a + b/(x+c)

popt, _ = curve_fit(func1, x_s, A_s, p0=[2400,-600,3900])
print(popt)



z = np.polyfit(x_s, A_s, 4) # калибровочная функция
# A_fit = np.poly1d(z)(x_s)
print("Коэффициенты калибровочной функции: ", z)

p = list(range(600, 2950))
A_fit = np.poly1d(z)(p) # построение калибровочной функции

so=round(sum([abs(A_s[i]-A_fit[i]) for i in range(0,len(x_s))])/(len(x_s)*sum(A_s))*100,4) # средняя ошибка
print('Средняя ошибка аппроксимации, %: ', so) 

Ha=round(np.poly1d(z)(2798), 1)
Hb=round(np.poly1d(z)(1792), 1)
Hg=round(np.poly1d(z)(1146), 1)
Hd=round(np.poly1d(z)(721), 1)
print("Измеренные длины волн серии Бальмера:", Ha, Hb, Hg, Hd)

I1 = round(np.poly1d(z)(2678), 1)
I2 = round(np.poly1d(z)(2588), 1)
I3 = round(np.poly1d(z)(1964), 1)
print("Измеренные характеристики диссоциации для йода:", I1, I2, I3)

fig1 = plt.figure("Figure 1")
plt.plot(x_s, func1(x_s, *popt), 'm_')
plt.plot(x, A, 'b.', markersize=7, label = "Неоновая лампа") 
plt.plot(x1, A1, 'k.', markersize=7, label = "Ртутная лампа")
plt.plot(p, A_fit, '--', linewidth = 0.6, color = 'g')
plt.minorticks_on()
# plt.grid(True)
plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.5)
plt.grid(which = 'minor', color='gray', linestyle='--', linewidth=0.3)

plt.xlabel('x, °', fontsize = 11)
plt.ylabel('λ, Å', fontsize = 11)
plt.legend(fontsize = 11)

fig1.set_figwidth(9)
fig1.set_figheight(6)
# plt.show()

#----------------------------------------------------------------------------------------------------------------------

fig1 = plt.figure("Figure 2")
y = [1/Ha, 1/Hb, 1/Hg, 1/Hd]
x3 = [1/4 - 1/9, 1/4 - 1/16, 1/4 - 1/25, 1/4 - 1/36]

# !!!! y = kx
d = np.polyfit(x3, y, 1) # аппроксимация
y_fit = np.poly1d(d)(x3) 
print("Коэффициенты полученной линейной функцииЖ ", d)

plt.plot(x3, y, 'k.', markersize=8) 
plt.plot(x3, y_fit, '--', linewidth = 0.6, color = 'k')
plt.minorticks_on()

plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.5)
plt.grid(which = 'minor', color='gray', linestyle='--', linewidth=0.3)

plt.errorbar(x3, y, yerr=[4e-7, 10e-7, 16e-7, 1e-6], fmt='r.', ecolor='r', elinewidth=1)

plt.xlabel('1/4 - 1/m^2', fontsize = 11)
plt.ylabel('1/λ, 1/Å', fontsize = 11)

fig1.set_figwidth(9)
fig1.set_figheight(6)
plt.show()