# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:42:45 2023

@author: Tommaso Giacometti
"""
import numpy as np
from scipy.integrate import odeint
import plots
import torch
from model import PINN
from utils import pde_scipy            


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##Real solution with scipy
    #Parameters 
    lam = 0.2
    nu = 0.33
    gamma = 2.
    delta = 0.33
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
    pinn = PINN().to(device)
    pinn.add_pde_parameters(params) # Add the parameters of the pde as buffers
    pinn.add_initial_condition(init) # Add the initial conditions, starting time by default is zero
    
    # Convert data to tensors
    # The input and target must be float numbers
    t = torch.from_numpy(time).float().view(-1, 1).to(device).requires_grad_(True)
  
    # Set training parameters
    epochs = 1000
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    loss_history = []
    
    #Training performed with Adam (or the one setted above)
    print(f'Adam training ({epochs} epochs):')
    for epoch in range(epochs):
        loss = pinn.train_step(t, optimizer).item() # Single train step defined in the PINN class
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
    plots.plot_solution_pinn(pinn, time, y)



if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    