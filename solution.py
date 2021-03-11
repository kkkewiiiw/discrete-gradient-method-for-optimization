import numpy as np
from scipy.optimize import fsolve
from sympy import *

#\delta t
h = 0.01
#Time width
T = 10
#number of time steps
N = int(T/h)

f = lambda x,y : #object function

#variable_z (z = (x,y))
z = np.ones((N+1,2))
#variable_x
z[0,0] = 10 #initial value x
#variable_y
z[0,1] = 10 #initial value y

expr1 = lambda x1,y1,x2,y2 : x1 - x2 + h *(#calculate by "Ito_Abe_method.py")
expr2 = lambda x1,y1,x2,y2 : y1 - y2 + h * (#calculate by "Ito_Abe_method.py")

df_x = lambda x,y,x2 : x - x2 + h * (#calculate by "partial_differencial.py")
df_y = lambda x,y,y2 : y - y2 + h * (#calculate by "partial_differencial.py")

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
