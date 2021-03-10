import numpy as np
from scipy.optimize import fsolve
from sympy import *
import matplotlib.pyplot as plt

#\delta t
h = 0.01
#Time width
T = 10
#number of time steps
N = int(T/h)

#object fucntion(McCormick Function)
f = lambda x,y : sin(x+y) + (x-y)**2 -1.5*x +2.5*y+1

#variable_z (z = (x,y))
z = np.ones((N+1,2))
#variable_x
z[0,0] = 0
#variable_y
z[0,1] = 0

expr1 = lambda x1,y1,x2,y2 : x1 - x2 + h * ((-1.5*x1 + 1.5*x2 + (x1 - y1)**2 - (x2 - y1)**2 + sin(x1 + y1) - sin(x2 + y1))/(x1 - x2))
expr2 = lambda x1,y1,x2,y2 : y1 - y2 + h * ((2.5*y1 - 2.5*y2 + (x2 - y1)**2 - (x2 - y2)**2 + sin(x2 + y1) - sin(x2 + y2))/(y1 - y2))

df_x = lambda x,y,x2 : x - x2 + h * (2*x - 2*y + cos(x + y) - 1.5)
df_y = lambda x,y,y2 : y - y2 + h * (-2*x + 2*y + cos(x + y) + 2.5)

for i in range(N):

    def fun(x):
        return [expr1(x[0],x[1],z[i,0],z[i,1]), expr2(x[0],x[1],z[i,0],z[i,1])]

    x = fsolve(fun,z[i])

    def func(x):
        return [df_x(x[0],x[1],z[i,0]),df_y(x[0],x[1],z[i,1])]
    
    X = fsolve(func,z[i])

    if abs(x[0] - z[i,0]) < 1.0e-14 :
        z[i+1,0] = X[0]
    else :
        z[i+1,0] = x[0]

    if abs(x[1] - z[i,1]) < 1.0e-14 :
        z[i+1,1] = X[1]
    else :
        z[i+1,1] = x[1]

F = np.ones((N+1, 1))
t = np.ones((N+1,1))

for i in range(N+1):
    F[i] = (f(z[i,0],z[i,1]))
    t[i] = h*i

plt.plot(t,F)
plt.xlabel('t')
plt.ylabel('log(f)')

# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()