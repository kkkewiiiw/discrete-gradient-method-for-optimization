import numpy as np
from sympy import *


#x_m,y_m
x2 = Symbol('x2')
y2 = Symbol('y2')

#x_m+1,y_m+1
x1 = Symbol('x1')
y1 = Symbol('y1')

def f(x, y):
    expr = (x + 2*y -7)**2 + (2*x + y -5)**2
    return expr

df_x = (f(x1, y1) - f(x2, y1))/(x1-x2)
df_y = (f(x2, y1) - f(x2, y2))/(y1-y2)

print(df_x)
print(df_y)
