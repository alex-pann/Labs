from asyncio import Handle
from ctypes.wintypes import HGDIOBJ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d


data = pd.read_excel("C:\\Users\\hp\\0-Alex\\BOTAY\\0-TOTAL\\5\\ФИЗИКА\\Лабы\\2.2\\data_obr.xlsx")
x = data.x
A = data.A
x1 = data.x1
A1 = data.A1
x_s = data.x_s
A_s = data.A_s
z = np.polyfit(x_s, A_s, 4)
A_fit = np.poly1d(z)(x_s)
print(z)

Ha=np.poly1d(z)(2798)
Hb=np.poly1d(z)(1792)
Hg=np.poly1d(z)(1146)
Hd=np.poly1d(z)(721)
print("H:", Ha, Hb, Hg, Hd)

I1 = np.poly1d(z)(2678)
I2 = np.poly1d(z)(2588)
I3 = np.poly1d(z)(1964)
print("I:", I1, I2, I3)
# f = interp1d(x_s, A_s, bounds_error=False, kind  = 'cubic')
# print(f)
fig1 = plt.figure("Figure 1")
plt.plot(x, A, 'b.', markersize=7, label = "Неоновая лампа") 
plt.plot(x1, A1, 'k.', markersize=7, label = "Ртутная лампа")
# plt.plot(x_s, f(x_s), '--', linewidth = 0.4, color = 'g')
plt.plot(x_s, A_fit, '--', linewidth = 0.6, color = 'g')
plt.minorticks_on()
# plt.grid(True)
plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.5)
plt.grid(which = 'minor', color='gray', linestyle='--', linewidth=0.3)

# plt.errorbar(I, B, xerr=0.01, yerr=0.01, fmt='b.', ecolor='b', elinewidth=0.8)
plt.xlabel('x, °', fontsize = 11)
plt.ylabel('λ, Å', fontsize = 11)
plt.legend(fontsize = 11)

fig1.set_figwidth(9)
fig1.set_figheight(6)
plt.show()

#----------------------------------------------------------------------------------------------------------------------

fig1 = plt.figure("Figure 2")
y = [1/Ha, 1/Hb, 1/Hg, 1/Hd]
x3 = [1/4 - 1/9, 1/4 - 1/16, 1/4 - 1/25, 1/4 - 1/36]

d = np.polyfit(x3, y, 1)
y_fit = np.poly1d(d)(x3)
print(d)

plt.plot(x3, y, 'k.', markersize=8) 
plt.plot(x3, y_fit, '--', linewidth = 0.6, color = 'k')
plt.minorticks_on()

plt.grid(which = 'major', color='gray', linestyle='-', linewidth=0.5)
plt.grid(which = 'minor', color='gray', linestyle='--', linewidth=0.3)

plt.errorbar(x3, y, yerr=[4e-7, 10e-7, 16e-7, 1e-6], fmt='r.', ecolor='r', elinewidth=1)

plt.xlabel('1/4 - 1/m^2', fontsize = 11)
plt.ylabel('1/λ, 1/Å', fontsize = 11)
plt.legend()

fig1.set_figwidth(9)
fig1.set_figheight(6)
plt.show()