# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:42:45 2023

@author: Tommaso Giacometti
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import odeint
import plots
import torch
from torch import nn
from model import PINN

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
    dz = states[2]*params[2] - states[3]*params[3]
    return ([dx1,dx2,dy1,dz])              


def main():
    ##Real solution with scipy
    #Parameters 
    lam = 0.2
    nu = 0.33
    gamma = 2.
    delta = 0.3
    params = np.array([lam,nu,gamma,delta])
    
    #Initial conditions
    x1 = 6
    x2 = 5
    y1 = 0
    z = 0
    init = np.array([x1,x2,y1,z])
    
    #Time domain
    ub_time = 20. #Upper bound for the time domain during the integration
    time = np.linspace(0, ub_time, 100)
    
    #Normalization factor
    cons = ub_time
    
    #Solution
    y = odeint(pde_scipy, init, time, args=(params,))
    #Normalized solution: limited time range (in [0,1]) to numerical stability of the neural network
    y_norm = odeint(pde_scipy, init, time/cons, args=(params*cons,)) #Simply a change of variable in a differential equation
    
    
    ##PINN solution    
    # Create PINN instance
    torch.manual_seed(0)
    pinn = PINN()
    pinn.add_pde_parameters(params) # Add the parameters of the pde as buffers
    pinn.add_initial_condition(init) # Add the initial conditions, starting time by default is zero
    
    # Convert data to tensors
    # The input and target must be float numbers
    t = torch.from_numpy(time).float().view(-1, 1).requires_grad_(True)
  
    # Set training parameters
    epochs = 1000
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loss_history = []
    
    #Training performed with Adam (or the one setted above)
    print(f'Adam training ({epochs} epochs):')
    for epoch in range(epochs):
        loss = pinn.train_step(t, optimizer, loss_fn).item() # Single train step defined in the PINN class
        loss_history.append(loss)
        if (epoch + 1) % (epochs / 20) == 0:
            print(f"{epoch/epochs*100:.1f}% -> Loss: ", loss)
            
    # Setting up the network for the LBFGS training
    optimizer, closure = pinn.set_up_lbfgs(t)
    
    # Training performed with LBFGS
    print(f'LBFGS training ({epochs} epochs):')
    for epoch in range(epochs):
        optimizer.step(closure)
        loss = closure().item()
        loss_history.append(loss)
        if (epoch + 1) % (epochs / 20) == 0:
            print(f"{epoch/epochs*100:.1f}% -> Loss: ", loss)
    
    # Plots
    plots.plot_solution_scipy(time, y, y_norm)
    plots.plot_loss(loss_history)
    plots.plot_solution_pinn(pinn, time)



if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    