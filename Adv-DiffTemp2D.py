#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:58:47 2018

@author: alex

Solution to the advection-difussion problem in a two dimensional space in transitory state:

\frac{{\partial T}}{{\partial t}} + \frac{{\partial }}{{\partial x}}( \rho u  T) +\frac{{\partial }}{{\partial y}}( \rho u  T)= \frac{{\partial }}{{\partial x}}\left(\Gamma\frac{{\partial  T}}{{\partial x}}\right) + \frac{{\partial }}{{\partial y}}\left(\Gamma\frac{{\partial  T}}{{\partial y}}\right)
"""

#----------------------------------------------------------------------------------------
                               #Library Imports
#----------------------------------------------------------------------------------------

import numpy as np
from RBF import Multiquadric2D
from RBF import GrammMatrix
from RBF import Solver
from Plotter import Plotter
from matplotlib.animation import FuncAnimation
from Knots import RegularMesh2D
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------
                           #Diffusion parameter definition
#----------------------------------------------------------------------------------------

Dx = .02
Dy = .02
T1 = 100
T2 = 100
T4 = 20
A = 1
alfa = 1
rho = 1
dt = 0.005
Tmax = .25
Tsteps = int(Tmax/dt)

#----------------------------------------------------------------------------------------
                           #Velocity functions definition
#----------------------------------------------------------------------------------------
def u(x,y,A,alfa):
    return -A*np.cos(alfa*np.pi*y)*np.sin(alfa*np.pi*x)

def v(x,y,A,alfa):
    return A*np.sin(alfa*np.pi*y)*np.cos(alfa*np.pi*x)

#----------------------------------------------------------------------------------------
                          #Creation of points cloud (knots)
#----------------------------------------------------------------------------------------

mesh = RegularMesh2D(15,15,1,2)
mesh.create2Dmesh()

#----------------------------------------------------------------------------------------
                           #Velocity vectors
#----------------------------------------------------------------------------------------
Ax = mesh.Ax()
Ay = mesh.Ay()
Ux = np.zeros(mesh.N())
Uy = np.zeros(mesh.N())

for i in range(mesh.N()):
    Ux[i] = u(Ax[i],Ay[i],A,alfa)
    Uy[i] = v(Ax[i],Ay[i],A,alfa)

#----------------------------------------------------------------------------------------
                               #Kernel selection
#----------------------------------------------------------------------------------------

kernel = Multiquadric2D(1/np.sqrt(mesh.N()))

#----------------------------------------------------------------------------------------
                             #Gramm matrx allocation
#----------------------------------------------------------------------------------------

matrix = GrammMatrix(mesh)
matrix.fillMatrixAdvDiffTime2D(kernel,Dx,Dy,Ux,Uy,rho)

#----------------------------------------------------------------------------------------
                          #Dirichlet boundary condition 
#----------------------------------------------------------------------------------------

#matrix.setDirichletRegular(T1,1)
matrix.setDirichletRegular(T2,2)
matrix.setDirichletRegular(T4,4)

#----------------------------------------------------------------------------------------
                                #Gram matrix initial solution
#----------------------------------------------------------------------------------------
    
solv = Solver(matrix,'bicgstab')
solv.solve()

#----------------------------------------------------------------------------------------
                           #Plot and animation initializtion
#----------------------------------------------------------------------------------------
title = 'Transient State Advection-Diffusion Heat Transfer with RBF'
xlabel = 'Lx [m]'
ylabel = 'Ly [m]'
barlabel = 'Temparature °C'
plot = Plotter(solv,kernel)

x = np.linspace(0,mesh.Lx(),mesh.Nx())
y = np.linspace(0,mesh.Ly(),mesh.Ny()) 
X, Y = np.meshgrid(x, y)
Z = solv.interpolate(kernel)
        
fig, ax = plt.subplots()
CS = plt.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin='lower')
cbar = plt.colorbar(CS)
label = ax.text(2.6, 0.5, 'Time = {:>8.5f}'.format(0),
            ha='center', va='center',
            fontsize=12)

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
cbar.ax.set_ylabel(barlabel)

#----------------------------------------------------------------------------------------
                                #Function Solver
#----------------------------------------------------------------------------------------

def implicitSolver(i):
    time_step = i * dt
    solv = Solver(matrix,'gmres')
    solv.solve()
    solv.evaluate(kernel)
    u = solv.getSol()
    Z = solv.interpolate(kernel)
    matrix.update(u,dt)
    #matrix.setDirichletRegular(T1,1)
    matrix.setDirichletRegular(T2,2)
    matrix.setDirichletRegular(T4,4)
    plt.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin='lower')
    if i >= Tsteps:
        C = plt.contour(X, Y, Z, 10, colors='black')
        plt.clabel(C, inline=1, fontsize=7.5)
        plt.savefig('TempSolution.png')
    label.set_text('Step = {:>8d} \n Time = {:>8.5f}'.format(i, time_step))
    print(i, '\n')
    
#----------------------------------------------------------------------------------------
                                #Solution animation
#----------------------------------------------------------------------------------------

anim = FuncAnimation(fig,               # La figura
                     implicitSolver,    # la función que cambia los datos
                     interval=1,        # Intervalo entre cuadros en milisegundos
                     frames=Tsteps+1,   # Cuadros
                     repeat=False)       # Permite poner la animación en un ciclo

plt.show()



