#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:35:21 2018

@author: alex

Solution to the Laplace equation in two dimensions.

the boundary conditions are given by:
    
       
                                  L = 1 m
                                  T = 0°C 
                          ______________________
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |  H = 2L
                         |                      |
                         |                      |
                 T= 0°C  |                      |  T = 0°C
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |______________________|
                                
                                  T = 100°C
                        

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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------
                           #Diffusion parameter definition
#----------------------------------------------------------------------------------------

D = 1
T1 = 100
#T2 = -25
#T3 = 34
#T4 = 7

#----------------------------------------------------------------------------------------
                          #Creation of points cloud (knots)
#----------------------------------------------------------------------------------------

mesh = RegularMesh2D(20,20,1,2)
mesh.create2Dmesh()

#----------------------------------------------------------------------------------------
                               #Kernel selection
#----------------------------------------------------------------------------------------

kernel = Multiquadric2D(1/np.sqrt(mesh.N()))

#----------------------------------------------------------------------------------------
                             #Gramm matrx allocation
#----------------------------------------------------------------------------------------

matrix = GrammMatrix(mesh)
matrix.fillMatrixLaplace2D(kernel,D)

#----------------------------------------------------------------------------------------
                          #Dirichlet boundary condition 
#----------------------------------------------------------------------------------------

matrix.setDirichletRegular(T1,1)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution
#----------------------------------------------------------------------------------------

solv = Solver(matrix,'gmres')
solv.solve()
solv.evaluate(kernel)

#----------------------------------------------------------------------------------------
                               #Solution storage(optional)
#----------------------------------------------------------------------------------------

zx =  solv.interpolate(kernel)
u = solv.getSol()
lam = solv.lam()

#----------------------------------------------------------------------------------------
                           #Solution and point cloud plotting
#----------------------------------------------------------------------------------------
title = 'Heat difussion in two dimensional domain'
xlabel = 'Lx [m]'
ylabel = 'Ly [m]'
barlabel = 'Temparature °C'
plot = Plotter(solv,kernel)
plot.regularMesh2D(title = 'Spatial created grid', xlabel = xlabel, ylabel = ylabel, fileName = "Laplace2DRM.png")
plot.surface3D(title = title, xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = "Laplace2D3D.png")
plot.levelplot(title = title, xlabel = xlabel, ylabel = ylabel, barlabel = barlabel, fileName = "Laplacelevel.png")

#----------------------------------------------------------------------------------------
                           #Analytical SOlution Plotting
#----------------------------------------------------------------------------------------

#axd = 0
#bxd = 1
#ayd = 0
#byd = 2
#boundA = 100
#boundB = 0
#boundC = 0
#boundD = 0
#def solucionAnalitica(x,y,N):
#   T=np.zeros((len(y),len(x)))
#   L=bxd-axd
#   H=byd-ayd
#   pi=np.pi
#   for i in range(0,len(x)):
#       for j in range(0,len(y)):
#           suma=0
#           for n in range(1,N):
#               suma += (1-(-1)**n)*np.sinh((n*pi*(H-y[j]))/L)*np.sin(n*pi*x[i]/L)/(n*pi*np.sinh(n*pi*H/L))
#           T[j,i]=boundA*2*suma
#       return T
#    
#Nx = 20
#Ny = 20
#x = np.arange(0, Nx)/(Nx-1)
#y = np.arange(0, Ny)/(Ny-1)
#
#X, Y = np.meshgrid(x, y)
#Z = solucionAnalitica(X, Y, 20)
#fig, ax = plt.subplots()
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#cbar = plt.colorbar(surf)
#plt.title("Analytical Solution")
#plt.xlabel("x [m]")
#plt.ylabel("y [m]")
#cbar.ax.set_ylabel("Temperature [°C]")
#plt.savefig("analytical_Laplace.png")
#        
#plt.show()

