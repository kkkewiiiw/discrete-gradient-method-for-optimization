import numpy as np
import math
from scipy.optimize import fsolve
from sympy import *

f = lambda x,y :  #object function

x=Symbol('x')
y=Symbol('y')

df_x = diff(f(x,y), x)
df_y = diff(f(x,y), y)

print(df_x)
print(df_y)