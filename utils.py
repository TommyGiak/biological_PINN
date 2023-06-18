#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:35:27 2023

@author: tommygiak
"""
from numpy.typing import ArrayLike
from scipy.integrate import odeint
from numpy.random import normal

#Differential equation system
def pde_scipy(states : ArrayLike, t : float, params : ArrayLike) -> ArrayLike:
    assert len(states) == 4 #Input states must be of lenght 4
    assert len(params) == 4 #Input params as well
    if min(states) < 0:
        raise ValueError('The states of the system is cannot be negative')
    if min(params) < 0:
        raise ValueError('The parameters of the system must be positive')
    dx1 = 0. #According to the model x1 is constant
    dx2 = params[0]*states[0] + states[1]*(params[0]-params[1])
    dy1 = states[1]*params[1] - states[2]*params[2] 
    dz = 2*states[2]*params[2] - states[3]*params[3]
    return ([dx1,dx2,dy1,dz]) 


def inverse_pinn_data_gen(init : ArrayLike, time : ArrayLike, params : ArrayLike, noise : bool, std = 0.1) -> ArrayLike:
    '''
    Function to generate the data needed for the inverse problem in the PINN fit.
    The data is generated with odeint and in will have the same number of points as the time in input.
    Can be added gaussian noise with a specific standard deviation (the mean is by default zero).

    Parameters
    ----------
    init : ArrayLike
        Initial state of problem.
    time : ArrayLike
        Time points where to solve the system.
    params : ArrayLike
        Parameters of the system.
    noise : bool
        Whether or not to add gaussian noise to the generated data (mean = 0).
    std : TYPE, optional
        Standar deviation of the noise. The default is 0.1.

    Returns
    -------
    ArrayLike
        Data for the inverse problem generated with scipy.integrate.odeint 

    '''
    y = odeint(pde_scipy, init, time, args=(params,))
    assert len(init) == 4 #Input states must be of lenght 4
    assert len(params) == 4 #Input params as well
    if noise:
        size = len(time)
        gauss = normal(scale=std, size=(size,4))
        y += gauss
    return y


