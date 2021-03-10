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

#object fucntion(Booth Function)
f = lambda x,y : (x + 2*y -7)**2 + (2*x + y -5)**2

#variable_z (z = (x,y))
z = np.ones((N+1,2))
#variable_x
z[0,0] = 10
#variable_y
z[0,1] = 10

expr1 = lambda x1,y1,x2,y2 : x1 - x2 + h * ((x1 + 2*y1 - 7)**2 + (2*x1 + y1 - 5)**2 - (x2 + 2*y1 - 7)**2 - (2*x2 + y1 - 5)**2)/(x1 - x2)
expr2 = lambda x1,y1,x2,y2 : y1 - y2 + h * ((x2 + 2*y1 - 7)**2 - (x2 + 2*y2 - 7)**2 + (2*x2 + y1 - 5)**2 - (2*x2 + y2 - 5)**2)/(y1 - y2)

df_x = lambda x,y,x2 : x - x2 + h * (10*x + 8*y - 34)
df_y = lambda x,y,y2 : y - y2 + h * (8*x + 10*y - 38)

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