#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:50:44 2023

@author: Tommaso Giacometti
"""
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import numpy as np
import torch

class Bcolors:
    #Class to print on terminal with different colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
        fig.suptitle('Solution of the differentail equation by scipy')
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
        ax[1].set_title('Normalized solution')
        plt.show()
    pass


def plot_loss(lossi : list, mean = 20, tit = None) -> None:
    '''
    Plot the loss history in log scale.
    
    Parameters
    ----------
    lossi : list, ArrayLike
    
    mean : Optional
        The plot will show the mean of the loss for this number of steps.
        
    tit : str, optional
        Title to put on the plot.

    Returns
    -------
    Show the plot
    '''
    try:
        lossi = np.array(lossi)
        y = lossi.reshape(-1,mean).mean(axis=1)
        x = np.linspace(1, len(y), num=len(y))
        fig, ax = plt.subplots()
        ax.plot(x,y)
        if tit is None:
            ax.set_title(f'Mean of {mean} losses steps')
        else:
            ax.set_title(tit)
        ax.set_ylabel('loss')
        ax.set_xlabel(f'epoch/{mean}')
        ax.set_yscale('log')
        plt.show()
        pass
    except:
        print(f'{Bcolors.WARNING}WARNING : {Bcolors.ENDC}the shape of lossi is not multiple of {mean}!')
        print('The loss track plot will not be shown')
        pass
    
    
def plot_solution_pinn(model, time):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    pred = model(torch.from_numpy(time).to(device).float().view(-1,1))
    x1 = pred[:,0].detach().cpu().numpy()
    x2 = pred[:,1].detach().cpu().numpy()
    y1 = pred[:,2].detach().cpu().numpy()
    z = pred[:,3].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(time, x1, label='x1')
    ax.plot(time, x2, label='x2')
    ax.plot(time, y1, label='y1')
    ax.plot(time, z, label='z')
    ax.legend()
    ax.set_title('PINN solution of the differentail equation')
    plt.show()
    
    
def plot_solution_pinn_inverse(model, time, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    pred = model(torch.from_numpy(time).to(device).float().view(-1,1))
    x1 = pred[:,0].detach().cpu().numpy()
    x2 = pred[:,1].detach().cpu().numpy()
    y1 = pred[:,2].detach().cpu().numpy()
    z = pred[:,3].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(time, x1, label='x1')
    ax.plot(time, x2, label='x2')
    ax.plot(time, y1, label='y1')
    ax.plot(time, z, label='z')
    ax.scatter(time, data[:,3], label='data used z', s = 10)
    ax.scatter(time, data[:,2], label='data used y1', s = 10)
    ax.legend()
    ax.set_title('PINN solution of the differentail equation')
    plt.show()
    
