#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:58:03 2018

@author: alex
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from Decorators import timePass

class Multiquadric1D():
    '''
    Class that contains mathemathical models and differential operators of Multicuadric Radial Based Function Kernel in one dimension:
        
        Class Variables:
            c: c parameter of multicuadric function.\n
    '''
    
    def __init__(self, c):
        '''
        Class constructor.
        
        Param:
             c: c parameter of multicuadric function.\n
        Return:
            -
        '''
        self.__c = c
        
    def mq(self, x, xj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
        Return:
            ((x - xj) * (x - xj) + c * c)
            
        '''
        c = self.__c
        return np.sqrt((x - xj) * (x - xj) + c * c)

    def d1x(self, x, xj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function first derivative on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
        Return:
            (x-xj) / sqrt( (x-xj) * (x-xj) + c * c )
        '''
        c = self.__c
        return (x-xj) / np.sqrt( (x-xj) * (x-xj) + c * c )

    def d2x(self, x, xj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function second derivative on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
        Return:
            c**2 / ( np.sqrt(r**2 + c**2) * (r**2 + c**2) )
            
        '''
        c2 = self.__c * self.__c
        r2 = (x-xj) * (x-xj)
        return c2 / ( np.sqrt(r2 + c2) * (r2 + c2) )

    def c(self):
        return self.__c
    

class Multiquadric2D():
    '''
    Class that contains mathemathical models and differential operators of Multicuadric Radial Based Function Kernel in two dimensions:
        
        Class Variables:
            c: c parameter of multicuadric function.\n
    '''
    
    def __init__(self, c):
        '''
        Class constructor.
        
        Param:
             c: c parameter of multicuadric function.\n
        Return:
             -
        '''
        self.__c = c
        
    def r(self, x, y, xj, yj):
        '''
        Method that returns the euclidean distance of a paier of coordinates.
        
        Param:
            x: Reference x coordinate.\n
            y: Reference y coordinate.\n
            xj: Evaluated x coordinate.\n
            yj: Evaluated y coordinate.\n
        Return:
            sqrt((x-xj)**2 + (y-yj)**2)
        '''
        return np.sqrt((x-xj)**2 + (y-yj)**2)
        
    def mq(self, x, y, xj, yj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
            y: Reference evaluated distance point y coordinate.\n
            yj: Evaluated distance point y coordinate.\n
        Return:
            (sqrt(r**2 + c**2)
            
        '''
        c2 = self.__c * self.__c
        r2 = self.r(x,y,xj,yj)**2
        return np.sqrt(r2 + c2)

    def d1x(self, x, y, xj, yj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function x first partial derivative on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
            y: Reference evaluated distance point y coordinate.\n
            yj: Evaluated distance point y coordinate.\n
        Return:
            (x-xj) / np.sqrt(r**2 + c**2)
        '''
        c2 = self.__c * self.__c
        r2 = self.r(x,y,xj,yj)**2
        return (x-xj) / np.sqrt(r2 + c2)
    
    def d1y(self, x, y, xj, yj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function y first partial derivative on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
            y: Reference evaluated distance point y coordinate.\n
            yj: Evaluated distance point y coordinate.\n
        Return:
            (y-yj) / np.sqrt(r**2 + c**2)
        '''
        c2 = self.__c * self.__c
        r2 = self.r(x,y,xj,yj)**2
        return (y-yj) / np.sqrt(r2 + c2)

    def d2x(self, x, y, xj, yj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function x second partial derivative on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
            y: Reference evaluated distance point y coordinate.\n
            yj: Evaluated distance point y coordinate.\n
        Return:
            (r**2 + c**2 - (x-xj)**2) / (sqrt(r**2 + c**2) * (r**2 + c**2) )
        '''
        c2 = self.__c * self.__c
        r2 = self.r(x,y,xj,yj)**2
        return (r2 + c2 - (x-xj)**2) / ( np.sqrt(r2 + c2) * (r2 + c2) )
    
    def d2y(self, x, y, xj, yj):
        '''
        Method that returns the evaluation of the Multicuadric Kernel function y second partial derivative on specified coordinates.
        
        Param:
            x: Reference evaluated distance point x coordinate.\n
            xj: Evaluated distance point x coordinate.\n
            y: Reference evaluated distance point y coordinate.\n
            yj: Evaluated distance point y coordinate.\n
        Return:
            (r**2 + c**2 - (y-yj)**2) / (sqrt(r**2 + c**2) * (r**2 + c**2) )
        '''
        c2 = self.__c * self.__c
        r2 = self.r(x,y,xj,yj)**2
        return (r2 + c2 - (y-yj)**2) / ( np.sqrt(r2 + c2) * (r2 + c2) )


    def c(self):
        return self.__c
    
class GrammMatrix():
    '''
    Class that represent the linear system  Gramm's matrix resulting of the Radial Based Funtion lambda evaluation.
    
    Class Variables:
        N: Number of domain nodes.\n
        NI: Number of border nodes.\n
        x: x coordinates of domain nodes.\n
        y: y coordinates of domain nodes.\n
        matrix: Allocated matrix array.\n
        fv: Right hand lienar system vector.\n
        knots: Object of Knots class.\n
        
    '''
    
    def __init__(self, knots):
        '''
        Class Constructor.
        
        Param:
            knots: Object of Knots class.\n
        Return:
            -
        '''
        self.__N = knots.N()
        self.__NI = knots.NI()
        self.__x = knots.Ax()
        self.__y = knots.Ay()
        self.__matrix = np.eye(self.__N)
        self.__fv = np.zeros(self.__N)
        self.__knots = knots

#----------------------------------------------------------------------------------------
                               #GrammMatrix Setters and Getters
#----------------------------------------------------------------------------------------
        
    def knots(self):
        return self.__knots
    
    def N(self):
        return self.__N
        
    def getMatrix(self):
        return self.__matrix
    
    def getfv(self):
        return self.__fv
        
    def setFv(self, f, i = None, j = None ):
        if i is not None and j is not None:
            self.__fv[i:j] += f
        else:
            self.__fv += f
            
    def update(self,solAnt, dt):
        '''
        Method that updated the right hand vector of the lienar system with previous results in temporal driven iterations.
        
        Param:
            solAnt: Prevoius solution vector.\n
            dt: Time step legnght.\n
        Return:
            -
        '''
        for i in range(self.__fv.size):
            self.__fv[i] = self.__fv[i]*dt + solAnt[i]
        
    @timePass    
    def setDirichletRegular(self,f,a):
        '''
        Method that updates the right hand vector of linear equations system with dirichelt border conditions of regular coordinates domain.
        
        Param:
            f: Right hand lienar equation system vector:\n
            a: Number of the rectangular border contition, 1 = inferior, 2 = Right, 3 = Superior, 4 = Left.\n
        Return:
            -
        '''
        knots = self.__knots
        i = (knots.Nx()-2)*(knots.Ny()-2)
        if a == 1:
            self.__fv[i: i + knots.Nx()] = f
        elif a == 2:
            self.__fv[i + knots.Nx() - 1: i + knots.Nx() + knots.Ny() -1] = f
        elif a == 3:
            self.__fv[i + knots.Nx() + knots.Ny() -2: i + knots.Nx()*2 + knots.Ny() -2] = f
        elif a == 4:
            self.__fv[i + knots.Nx()*2 + knots.Ny() - 3: i + knots.Nx()*2 + knots.Ny()*2 -4] = f
            self.__fv[i] = f
    
    @timePass       
    def setNeummanRegular(self,kernel,f,a):
        '''
        Method that updates the right hand vector of linear equations system with Neumman border conditions of regular coordinates domain.
        
        Param:
            kernel: Kernel class object instance.\n
            f: Right hand lienar equation system vector:\n
            a: Number of the rectangular border contition, 1 = inferior, 2 = Right, 3 = Superior, 4 = Left.\n
        Return:
            -
        '''
        N = self.__N
        x = self.__x
        y = self.__y
        knots = self.__knots
        q = (knots.Nx()-2)*(knots.Ny()-2)
        if a == 1:
            for i in range(q+1, q + knots.Nx()-1):
                self.__fv[i] = f 
                for j in range(N):
                    self.__matrix[i,j] = -1*kernel.d1y(x[i], y[i], x[j], y[j])
                    
        elif a == 2:
            for i in range(q +  knots.Nx() , q + knots.Nx() + knots.Ny() -2):
                self.__fv[i] = f 
                for j in range(N):
                    self.__matrix[i,j] = kernel.d1x(x[i], y[i], x[j], y[j])
                    
        elif a == 3:
            for i in range(q + knots.Nx() + knots.Ny() -1, q + knots.Nx()*2 + knots.Ny() -3):
                self.__fv[i] = f 
                for j in range(N):
                    self.__matrix[i,j] = kernel.d1y(x[i], y[i], x[j], y[j])
                    
        elif a == 4:
            for i in range(q + knots.Nx()*2 + knots.Ny() - 2, q + knots.Nx()*2 + knots.Ny()*2 -4):
                self.__fv[i] = f 
                for j in range(N):
                    self.__matrix[i,j] = -1*kernel.d1x(x[i], y[i], x[j], y[j])
            
                    
    @timePass
    def fillMatrixLaplace2D( self, kernel , D):
         '''
         Method that fills the linear system Gramm's Matrix based on the Stationary Diffusive mathemathical model.
        
         Param:
            kernel: Kernel class object instance.\n
            D: Diffusive Conductivity Constant (Gamma).\n
            
         Return:
            -
         '''
         N = self.__N
         NI = self.__NI
         x = self.__x
         y = self.__y
    
    # ----- WL block matrix
         for i in range(NI):
            for j in range(N):
                self.__matrix[i,j] = D * (kernel.d2x(x[i], y[i], x[j], y[j])) + D * (kernel.d2y(x[i], y[i], x[j], y[j]))
            
    # ----- WB block matrix
         for i in range(NI,N):
            for j in range(N):
                self.__matrix[i,j] = kernel.mq(x[i], y[i], x[j], y[j])
                
    @timePass
    def fillMatrixAdvDiff2D( self, kernel , Dx, Dy, Ux, Uy, rho):
         '''
         Method that fills the linear system Gramm's Matrix based on the Stationary Diffusive-Advection mathemathical model.
        
         Param:
            kernel: Kernel class object instance.\n
            Dx: Diffusive Conductivity Constant in x (Gamma).\n
            Dy: Diffusive Conductivity Constant in y (Gamma).\n
            Ux: Fluid Velocity in x.\n
            Uy: Fluid Velocity in y.\n
            tho: Fluid density.\n
         Return:
            -
         '''
         N = self.__N
         NI = self.__NI
         x = self.__x
         y = self.__y
    
    # ----- WL block matrix
         for i in range(NI):
            for j in range(N):
                self.__matrix[i,j] = (rho*Ux[i]*kernel.d1x(x[i], y[i], x[j], y[j]) + rho*Uy[i]*kernel.d1y(x[i], y[i], x[j], y[j])
                - Dx * (kernel.d2x(x[i], y[i], x[j], y[j])) - Dy * (kernel.d2y(x[i], y[i], x[j], y[j])))
            
    # ----- WB block matrix
         for i in range(NI,N):
            for j in range(N):
                self.__matrix[i,j] = kernel.mq(x[i], y[i], x[j], y[j])
       
    @timePass         
    def fillMatrixAdvDiffTime2D( self, kernel , Dx, Dy, Ux, Uy, rho):
         '''
         Method that fills the linear system Gramm's Matrix based on the Transient Diffusive-Advection mathemathical model.
        
         Param:
            kernel: Kernel class object instance.\n
            Dx: Diffusive Conductivity Constant in x (Gamma).\n
            Dy: Diffusive Conductivity Constant in y (Gamma).\n
            Ux: Fluid Velocity in x.\n
            Uy: Fluid Velocity in y.\n
            tho: Fluid density.\n
         Return:
            -
         '''
         N = self.__N
         NI = self.__NI
         x = self.__x
         y = self.__y
    
    # ----- WL block matrix
         for i in range(NI):
            for j in range(N):
                self.__matrix[i,j] = ( kernel.mq(x[i], y[i], x[j], y[j])
                    + rho*Ux[i]*kernel.d1x(x[i], y[i], x[j], y[j]) + rho*Uy[i]*kernel.d1y(x[i], y[i], x[j], y[j])
                - Dx * (kernel.d2x(x[i], y[i], x[j], y[j])) - Dy * (kernel.d2y(x[i], y[i], x[j], y[j])))
            
    # ----- WB block matrix
         for i in range(NI,N):
            for j in range(N):
                self.__matrix[i,j] = kernel.mq(x[i], y[i], x[j], y[j])

class Solver():
    '''
    Class that contains the information for the lanbda linear equation system solution.
    
    Class Variables:
        lam: Vector thet contains the lanba values of solution.\n
        u: Vector of evaluated solution in the referecne coordinate domain.\n
        algorithm: String used to select linear system solution method, linalg, gmres, bcgstab, bicg, cg.\n
        matrix: GrammMatric class object instance.\n
        
    '''
    def __init__(self, matrix, algorithm = None):
        '''
        Class Cosntructor.
        
        Param:
            algorithm: String used to select linear system solution method, linalg, gmres, bcgstab, bicg, cg.\n
            matrix: GrammMatric class object instance.\n
        Return:
            -
        '''
        self.__lam = np.zeros(matrix.N())
        self.__u = np.zeros(matrix.N())
        if algorithm:
            self.__algorithm = algorithm
        else:
            self.__algorithm = 'linalg'
        self.__matrix = matrix

#----------------------------------------------------------------------------------------
                               #Solver Getters
#----------------------------------------------------------------------------------------
        
    def getMatrix(self):
        return self.__matrix
    
    def getKnots(self):
        return self.__matrix.knots()
    
    def lam(self):
        return self.__lam
    
    def getSol(self):
        return self.__u
    
    @timePass
    def solve(self):
        '''
        Method that solves the linear equations system.
        
        Param:
            -
        Return:
            -
        '''
        G = self.__matrix.getMatrix()
        f = self.__matrix.getfv()
        
        if self.__algorithm == 'linalg':
            self.__lam = np.linalg.solve(G,f)
        elif self.__algorithm == 'bicgstab':
            A = sps.csr_matrix(G)
            self.__lam = spla.bicgstab(A,f)[0]
        elif self.__algorithm == 'bicg':
            A = sps.csr_matrix(G)
            self.__lam = spla.bicg(A,f)[0]
        elif self.__algorithm == 'cg':
            A = sps.csr_matrix(G)
            self.__lam = spla.cg(A,f)[0]
        elif self.__algorithm == 'gmres':
            A = sps.csr_matrix(G)
            self.__lam = spla.gmres(A,f)[0]
           # print(self.lam())
    
    @timePass
    def evaluate(self, kernel): 
        '''
        Method that evaluates the solution in euclidean domain with calculated lambda values.
        
        Param:
            kernel: Kernel object instance.\n
        Return:
            -
        '''
        knots = self.__matrix.knots()
        x = knots.Ax()
        y= knots.Ay()
        for i in range(self.__matrix.N()):
            for j in range(self.__matrix.N()):
                self.__u[i] += self.__lam[j] * kernel.mq(x[i],y[i],x[j],y[j])
      
    @timePass      
    def interpolate(self,kernel):
        '''
        Method that calculates the solution interpolation in a coordinate domain using calculated lambda values and returns the values in the form of a meshgrid.
        
        Param:
            kernel: Kernel object instance.\n
        Return:
            z: meshgrid solution values in the euclidean domain.
        '''
        knots = self.__matrix.knots()
        x = np.linspace(0,knots.Lx(),knots.Nx())
        y = np.linspace(0,knots.Ly(),knots.Ny()) 
        x2 = knots.Ax()
        y2 = knots.Ay()
        z = np.eye(knots.Ny(),knots.Nx())
        for i in range(knots.Nx()):
            for j in range(knots.Ny()):
                for p in range(knots.N()):
                    z[j,i] += self.__lam[p] * kernel.mq(x[i],y[j],x2[p],y2[p])
                         
        return z

if __name__ == '__main__':
    
    from Knots import RegularMesh2D
    
    mesh = RegularMesh2D(6,6,2,2)
    mesh.create2Dmesh()
    
    kernel = Multiquadric2D(1)
    
    GM = GrammMatrix(mesh)
    GM.fillMatrixLaplace2D(kernel,1)
    Ga = GM.getMatrix()
    print (mesh.Ax())
    print(mesh.Ay())
    print(Ga)
    print(kernel.d2x(1,1,0,0) + kernel.d2y(1,1,0,0))
    print('-'*20,'\n')
    GM.setNeummanRegular(kernel,0,1)