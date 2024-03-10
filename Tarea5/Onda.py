# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:14:20 2024

@author: THINKBOOK
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib as mpl
import matplotlib.animation as animation
from tqdm import tqdm

Nt = 200
Nx = 20
Ny = 20

x = np.linspace(0,2,Nx)
y = np.linspace(0,2,Nx)
t = np.linspace(0,np.pi*0.5,Nt)

deltax = x[1] - x[0]
deltay = y[1] - y[0]
deltat = t[1] - t[0]

v = 2.

lambda_ = v*deltat/deltax
mu_ = v*deltat/deltay

print(lambda_,mu_)

def ui(x,y):
    
    return np.sin(np.pi*x)*np.sin(np.pi*y)


u = np.zeros((Nt,Nx,Ny))
for i in range(len(x)):
    for j in range(len(y)):
        u[0,i,j] = ui(x[i],y[j])
        
        
def GetSolution():
    
    gamma = 0*deltat #amortiguamiento
    
    for l in tqdm(range(1,len(t))):
        
        if l == 1:
            u[l,:,:] = u[l-1,:,:]
        else:
        
            for i in range(1,len(x)-1):
                for j in range(1,len(y)-1):
                    u[l,i,j] = 2*(1-lambda_**2-mu_**2)*u[l-1,i,j] \
                    + lambda_**2*( u[l-1,i+1,j] + u[l-1,i-1,j] ) \
                    + mu_**2*( u[l-1,i,j+1] + u[l-1,i,j-1] ) \
                    - u[l-2,i,j] \
                    - gamma*u[l-1,i,j] + gamma*u[l-2,i,j] 
 
GetSolution()


fig = plt.figure(figsize=(5,6))
ax = fig.add_subplot(111, projection='3d')


X,Y = np.meshgrid(x,y)

def init():
    
    ax.view_init(elev=20, azim=45)
    ax.set_xlim3d(0,2)
    ax.set_ylim3d(0,2)
    ax.set_zlim3d(-1,1)

def Update(i):

    ax.clear()
    init()
    
    ax.plot_surface(X,Y,u[i,:,:],cmap='viridis')
    
Animation = animation.FuncAnimation(fig,Update,frames=len(t),init_func=init)

writer = animation.PillowWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
Animation.save('Onda.gif', writer=writer)