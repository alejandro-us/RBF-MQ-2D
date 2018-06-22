#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:19:54 2018

@author: alex
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from Decorators import timePass
 
class Plotter():
    '''
    Class that contains solution information for plotting and display.
    
    Class Variables:
        solver: Solver class object instance.\n
        kernel: Kernel class object Instance.\n
    '''
    
    def __init__(self,solv, kernel):
        '''
        Class Cosntructor.
        
        Param:
            solver: Solver class object instance.\n
            kernel: Kernel class object Instance.\n
        Return:
            -
        '''
        self.__solver = solv
        self.__kernel = kernel
    
    @timePass
    def levelplot(self, title = None, xlabel = None, ylabel = None, barlabel = None, fileName = None):
        '''
        Method for level plotting and isolines along the domain in two dimensions.
        
        Param:
            title: Title of the plott.\n
            xlablel: x axis label.\n
            ylabel: y axis label.\n
            barlabel: Label for the color bar.\n
            filename: Name of the output file.\n
        Return:
            -
        '''
        knots = self.__solver.getKnots()
        x = np.linspace(0,knots.Lx(),knots.Nx())
        y = np.linspace(0,knots.Ly(),knots.Ny()) 
        X, Y = np.meshgrid(x, y)
        Z = self.__solver.interpolate(self.__kernel)
        
        fig, ax = plt.subplots()
        CS = plt.contourf(X, Y, Z, 10, alpha=.75, cmap=plt.cm.bone)
        cbar = plt.colorbar(CS)
        C = plt.contour(X, Y, Z, 10, colors='black')
        plt.clabel(C, inline=1, fontsize=7.5)
        
        
        #cbar.add_lines(CS)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if barlabel:
            cbar.ax.set_ylabel(barlabel)
        if fileName:
            plt.savefig(fileName)
        plt.show()
     
    @timePass    
    def levelplotAnimation(self, i , title = None, xlabel = None, ylabel = None, barlabel = None, fileName = None):
        knots = self.__solver.getKnots()
        x = np.linspace(0,knots.Lx(),knots.Nx())
        y = np.linspace(0,knots.Ly(),knots.Ny()) 
        X, Y = np.meshgrid(x, y)
        Z = self.__solver.interpolate(self.__kernel)
        
        fig, ax = plt.subplots()
        CS = plt.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin='lower')
        cbar = plt.colorbar(CS)
        label = ax.text(2.6, 0.5, 'Time = {:>8.5f}'.format(0),
                ha='center', va='center',
                fontsize=12)
        label.set_text('Step = {:>8d} \n Time = {:>8.5f}'.format(i, time_step))
        
        #cbar.add_lines(CS)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if barlabel:
            cbar.ax.set_ylabel(barlabel)
        plt.savefig(fileName)
        plt.show()
        
    @timePass
    def surface3D(self, title = None, xlabel = None, ylabel = None, barlabel = None, fileName = None):
        '''
        Method for 3D scatter plotting along the domain in two dimensions.
        
        Param:
            title: Title of the plott.\n
            xlablel: x axis label.\n
            ylabel: y axis label.\n
            barlabel: Label for the color bar.\n
            filename: Name of the output file.\n
        Return:
            -
        '''
        knots = self.__solver.getKnots()
        x = np.linspace(0,knots.Lx(),knots.Nx())
        y = np.linspace(0,knots.Ly(),knots.Ny()) 
        X, Y = np.meshgrid(x, y)
        Z = self.__solver.interpolate(self.__kernel)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar = plt.colorbar(surf)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if barlabel:
            cbar.ax.set_ylabel(barlabel)
        if fileName:
            plt.savefig(fileName)
        
        plt.show()
     
    @timePass    
    def regularMesh2D(self,title = None, xlabel = None, ylabel = None, fileName = None):
        '''
        Method for regular mesh plotting along the domain in two dimensions.
        
        Param:
            title: Title of the plott.\n
            xlablel: x axis label.\n
            ylabel: y axis label.\n
            filename: Name of the output file.\n
        Return:
            -
        '''

        NIx,NIy = self.__solver.getKnots().aNI()
        NBx,NBy = self.__solver.getKnots().aNB()
        plt.scatter(NIx,NIy,c='b', marker = '.')
        plt.scatter(NBx,NBy,c='r', marker = '.')
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if fileName:
            plt.savefig(fileName)
        plt.show()
        
    def u(x,y,A,alfa):
        return -A*np.cos(alfa*np.pi*y)*np.sin(alfa*np.pi*x)

    def v(x,y,A,alfa):
        return A*np.sin(alfa*np.pi*y)*np.cos(alfa*np.pi*x)
    
    @timePass
    def vectorFieldVelocity(self,A,alfa,fileName = None):
        '''
        Method for level plotting and isolines along the domain in two dimensions.
        
        Param:
            A: Forced velocity ammplitude parameter.\n
            alfa: Forced velocity function parameter.\n
            filename: Name of the output file.\n
        Return:
            -
        '''
        knots = self.__solver.getKnots()
        x = np.linspace(0,knots.Lx(),knots.Nx())
        y = np.linspace(0,knots.Ly(),knots.Ny())
        X, Y = np.meshgrid(x, y)
        U = -A*np.cos(alfa*np.pi*Y)*np.sin(alfa*np.pi*X)
        V = A*np.sin(alfa*np.pi*Y)*np.cos(alfa*np.pi*X)
        fig, ax = plt.subplots()
        q = ax.quiver(X, Y, U, V)
        ax.quiverkey(q, X=0.3, Y=1.1, U=1,
                 label='Velocity vector field', labelpos='E')
        if fileName:
            plt.savefig(fileName)

        plt.show()

            