#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:14:36 2018

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from Decorators import timePass

class RegularMesh2D():
    '''
    Class that creates a rectangular cloud of coordinates at constant steps using number of nodes and lenght.
    
    Class Varibles:
            Nx: Number of nodes in the x euclidean direction.\n
            Ny: Number of nodes in the y euclidean direction.\n
            Lx: Lenght of domain in the x euclidean direction.\n
            Ly: Lenght of domain in the y euclidean direction.\n
            N: Number of domain nodes.\n
            NB: Number if border nodes.\n
            NI: number of internal nodes.\n
            Ax: Coordinates in x of overall domain nodes.\n
            Ax: Coordinates in x of overall domain nodes.\n
            dx: Constant grid step in the x direction.\n
            dy: Constant grid step in the y direction.\n
            
    '''
    
    def __init__(self, Nx, Ny, Lx, Ly):
        '''
        Class Costructor:
        
        Param:
            Nx: Number of nodes in the x euclidean direction.\n
            Ny: Number of nodes in the y euclidean direction.\n
            Lx: Lenght of domain in the x euclidean direction.\n
            Ly: Lenght of domain in the y euclidean direction.\n
        Return:
            -
        '''
        
        self.__Nx = Nx
        self.__Ny = Ny
        self.__Lx = Lx
        self.__Ly = Ly
        self.__N = Nx*Ny
        self.__NB = 2*Nx + 2*Ny - 4
        self.__NI = self.__N - self.__NB
        self.__Ax = np.zeros(self.__N)
        self.__Ay = np.zeros(self.__N)
        self.__dx = Lx/(Nx-1)
        self.__dy = Ly/(Ny-1)
        
    def __del__(self):
        '''
        Class Destructor:
        
        Param:
            -
        Return:
            -
        '''
        
        del (self.__Nx)
        del (self.__Ny)
        del (self.__Lx)
        del (self.__Ly)
        
#----------------------------------------------------------------------------------------
                               #RegularMesh Setters
#----------------------------------------------------------------------------------------

    def N(self):
        return self.__N
        
    def Nx(self):
        return self.__Nx
    
    def Ny(self):
        return self.__Ny
    
    def Lx(self):
        return self.__Lx
    
    def Ly(self):
        return self.__Ly
    
    def Ax(self):
        return self.__Ax
    
    def Ay(self):
        return self.__Ay
    
    def NI(self):
        return self.__NI
    
    def NB(self):
        return self.__NB
    
    def aNI(self):
        return self.__Ax[0:self.__NI], self.__Ay[0:self.__NI]
    
    def aNB(self):
        return self.__Ax[self.__NI:self.__N], self.__Ay[self.__NI:self.__N]
        
    @timePass
    def create2Dmesh(self):
        '''
        Methds that allocates memory space with the ecuclidiean coordinates for the regular mesh in the calss variables Ax and Ay.
        
        Param:
            -
        Return:
            -
        '''
        count = 0
        for i in range(1,self.__Ny-1):
            for j in range(1,self.__Nx-1):
                self.__Ax[count] = self.__dx*j
                self.__Ay[count] = self.__dy*i
                count += 1 
        for i in range(self.__Nx):
            self.__Ax[count] = self.__dx*i
            self.__Ay[count] = 0
            count += 1 
        for j in range(self.__Ny-2):
            self.__Ax[count] = self.__Lx
            self.__Ay[count] = self.__dy*(j+1)
            count += 1 
        for i in range(self.__Nx-1,-1,-1):
            self.__Ax[count] = self.__dx*i
            self.__Ay[count] = self.__Ly
            count += 1 
        for j in range(self.__Ny-2,0,-1):
            self.__Ax[count] = 0
            self.__Ay[count] = self.__dy*(j)
            count += 1 
         
            
if __name__ == '__main__':
    
    mesh = RegularMesh2D(5,5,12,12)
    mesh.create2Dmesh()
    plt.title('Created 2 Dimensional Grid')
    plt.xlabel('Lx [m]')
    plt.ylabel('Ly [m]')
    NIx,NIy = mesh.aNI()
    NBx,NBy = mesh.aNB()
    plt.scatter(NIx,NIy,c='b', marker = '.')
    plt.scatter(NBx,NBy,c='r', marker = '.')
    plt.xlabel('Lx [m]')
    plt.ylabel('Ly [m]')
    plt.show()
    
        
    
        
        