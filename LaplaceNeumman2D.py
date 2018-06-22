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
            dT/dx = 0°C  |                      |  dT/dx = 0°C
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

mesh = RegularMesh2D(25,25,1,2)
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

matrix.setNeummanRegular(kernel,0,2)
matrix.setNeummanRegular(kernel,0,4)
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
plot.regularMesh2D(title = 'Spatial created grid', xlabel = xlabel, ylabel = ylabel)
plot.surface3D(title = title, xlabel = xlabel, ylabel = ylabel, barlabel = barlabel)
plot.levelplot(title = title, xlabel = xlabel, ylabel = ylabel, barlabel = barlabel, fileName = "LaplaceNeumman.png")