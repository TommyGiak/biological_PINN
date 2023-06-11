# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:42:45 2023

@author: Tommaso Giacometti
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import odeint
import plots

#Differential equation system
def pde(states : ArrayLike, t : float, params : ArrayLike) -> ArrayLike:
    assert len(states) == 4 #Input states must be of lenght 4
    assert len(params) == 4 #Input params as well
    if min(states) < 0:
        raise ValueError('The states of the system is cannot be negative')
    if min(params) < 0:
        raise ValueError('The parameters of the system must be positive')
    dx1 = 0. #According to the model x1 is constant
    dx2 = params[0]*states[0] + states[1]*(params[0]-params[1])
    dy1 = states[1]*params[1] - states[2]*params[2] 
    dz = states[2]*params[2] - states[3]*params[3]
    return ([dx1,dx2,dy1,dz])


def main():
    #Parameters 
    lam = 0.2
    nu = 0.33
    gamma = 2.
    delta = 0.3
    params = np.array([lam,nu,gamma,delta])
    
    #Initial conditions
    x1 = 6
    x2 = 10
    y1 = 0
    z = 0
    init = np.array([x1,x2,y1,z])
    
    #Time domain
    ub_time = 20 #Upper bound for the time domain during the integration
    time = np.linspace(0, ub_time, 1000)
    
    #Normalization factor
    cons = ub_time
    
    #Solution
    y = odeint(pde, init, time, args=(params,))
    #Normalized solution: limited time range (in [0,1]) to numerical stability of the neural network
    y_norm = odeint(pde, init, time/cons, args=(params*cons,)) #Simply a change of variable in a differential equation
    
    #Plot
    plots.plot_solution_scipy(time, y, y_norm)


if __name__ == '__main__':
    main()