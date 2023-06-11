#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:50:44 2023

@author: Tommaso Giacometti
"""
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

#Plots
def plot_solution_scipy(time : ArrayLike, sol : ArrayLike, sol2 : ArrayLike = None) -> None:
    '''
    Function to plot the solution of the differentail equation computed by scipy odeint.
    It can plot in a subplot also the 'normalized' solution to check if they are equal.

    Parameters
    ----------
    time : ArrayLike
        Time array for the solutions.
    sol : ArrayLike
        odeint solution of the differential equation.
    sol2 : ArrayLike, optional
        odeint solution of the 'normalized' solution of the diff. eq. The default is None 
        and will be plotted only the first solution.

    Returns
    -------
    None
    '''
    if sol2 is None:
        fig, ax = plt.subplots()
        ax.set_title('Solution of the differentail equation')
        ax.plot(time, sol[:,0], label = 'x1')
        ax.plot(time, sol[:,1], label = 'x2')
        ax.plot(time, sol[:,2], label = 'y1')
        ax.plot(time, sol[:,3], label = 'z')
        ax.legend()
        plt.show()
    else:
        fig, ax = plt.subplots(1,2, figsize = (8,4))
        fig.suptitle('Solution of the differentail equation')
        ax[0].plot(time, sol[:,0], label = 'x1')
        ax[0].plot(time, sol[:,1], label = 'x2')
        ax[0].plot(time, sol[:,2], label = 'y1')
        ax[0].plot(time, sol[:,3], label = 'z')
        ax[0].legend()
        ax[0].set_title('Solution')
        ax[1].plot(time, sol2[:,0], label = 'x1')
        ax[1].plot(time, sol2[:,1], label = 'x2')
        ax[1].plot(time, sol2[:,2], label = 'y1')
        ax[1].plot(time, sol2[:,3], label = 'z')
        ax[1].legend()
        ax[0].set_title('Normalized solution')
        plt.show()
    pass
