import numpy as np
from scipy.optimize import fsolve
from sympy import *
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import wand.image


h = #\delta t

T = #Time width

N = #number of time steps

f = lambda x,y : #object function

#variable_z (z = (x,y))
z = np.ones((N+1,2))
z[0,0] = #initial value x
z[0,1] = #initial value y

expr1 = lambda x1,y1,x2,y2 : #calculate by "Ito_Abe_method.py"
expr2 = lambda x1,y1,x2,y2 : #calculate by "Ito_Abe_method.py"

df_x = lambda x,y,x2 : #calculate by "partial_differencial.py"
df_y = lambda x,y,y2 : #calculate by "partial_differencial.py"

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


#graph
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.xlabel('x')
plt.ylabel('y')

plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


#initial value
plt.scatter(z[0,0], z[0,1], color = "blue", s = 40)
plt.text(z[0,0],z[0,1],"初期値", fontname="MS Gothic", size ='xx-large')

#Optimal solution
plt.scatter(z[N,0], z[N,1], color = "blue", s = 40)
plt.text(z[N,0],z[N,1],"最適解", fontname="MS Gothic", size ='xx-large')

#contour
x = np.arange(0, 5, 0.1) # x axis
y = np.arange(0, 5, 0.1) # y axis

X, Y = np.meshgrid(x, y)

cont = plt.contour(X, Y, f(X,Y), colors='slategray', levels=15)

#animation
graph_list = []
for i in range(N+1):
    x = z[i,0]
    y = z[i,1]
    graph = plt.scatter(x,y,color = "black", s = 40)
    title = ax.text(0.5, 1.01, 'step={:}'.format(i), ha='center', va='bottom',transform=ax.transAxes, fontsize='large')
    graph_list.append([graph]+ [title])

ani = anm.ArtistAnimation(fig, graph_list, interval = 200)


plt.show()

#save
ani.save("Booth_Function_Ani.gif", writer="imagemagick")
