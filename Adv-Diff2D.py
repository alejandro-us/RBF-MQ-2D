#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:20:39 2018

@author: alex

Solution to the advection-difussion problem in a two dimensional space in stationary state:

\frac{{\partial }}{{\partial x}}( \rho u  T) +\frac{{\partial }}{{\partial y}}( \rho u  T)= \frac{{\partial }}{{\partial x}}\left(\Gamma\frac{{\partial  T}}{{\partial x}}\right) + \frac{{\partial }}{{\partial y}}\left(\Gamma\frac{{\partial  T}}{{\partial y}}\right)

       
"""
#----------------------------------------------------------------------------------------
                               #Library Imports
#----------------------------------------------------------------------------------------

import numpy as np
from Knots import RegularMesh2D
from RBF import Multiquadric2D
from RBF import GrammMatrix
from RBF import Solver
from Plotter import Plotter

#----------------------------------------------------------------------------------------
                           #Diffusion parameter definition
#----------------------------------------------------------------------------------------

Dx = .02
Dy = .02
T1 = 100
T2 = 100
T4 = 20
A = 2
alfa = 1
rho = 1

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

mesh = RegularMesh2D(25,25,1,2)
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
matrix.fillMatrixAdvDiff2D(kernel,Dx,Dy,Ux,Uy,rho)

#----------------------------------------------------------------------------------------
                          #Dirichlet boundary condition 
#----------------------------------------------------------------------------------------

#matrix.setDirichletRegular(T1,1)
matrix.setDirichletRegular(T2,2)
matrix.setDirichletRegular(T4,4)
fb= matrix.getfv()
#matrix.setFv(100,17,18)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution
#----------------------------------------------------------------------------------------

solv = Solver(matrix,'gmres')
solv.solve()
lam = solv.lam()
solv.evaluate(kernel)

#----------------------------------------------------------------------------------------
                               #Solution storage(optional)
#----------------------------------------------------------------------------------------

zx =  solv.interpolate(kernel)
u = solv.getSol()
lam = solv.lam()
fa = matrix.getfv()
#----------------------------------------------------------------------------------------
                           #Solution and point cloud plotting
#----------------------------------------------------------------------------------------
title = 'Steady State Advection-Diffusion Heat Transfer with RBF'
xlabel = 'Lx [m]'
ylabel = 'Ly [m]'
barlabel = 'Temparature °C'
plot = Plotter(solv,kernel)
plot.regularMesh2D(title = 'Spatial created grid', xlabel = xlabel, ylabel = ylabel)
plot.surface3D(title = title, xlabel = xlabel, ylabel = ylabel, barlabel = barlabel)
plot.levelplot(title = title, xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'Stationary.png')
plot.vectorFieldVelocity(A,alfa, fileName = "Laplacelevel.png")



