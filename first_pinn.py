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
from torch.autograd import grad
import matplotlib.pyplot as plt


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
    x2 = 10
    y1 = 0
    z = 0
    init = np.array([x1,x2,y1,z])
    
    #Time domain
    ub_time = 2. #Upper bound for the time domain during the integration
    time = np.linspace(0, ub_time, 1000)
    
    #Normalization factor
    cons = ub_time
    
    #Solution
    y = odeint(pde_scipy, init, time, args=(params,))
    #Normalized solution: limited time range (in [0,1]) to numerical stability of the neural network
    y_norm = odeint(pde_scipy, init, time/cons, args=(params*cons,)) #Simply a change of variable in a differential equation
    
    #Plot
    plots.plot_solution_scipy(time, y, y_norm)
    
    
    ##PINN solution
    
    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(1, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 4)
            )
    
        def forward(self, inp):
            return self.seq(inp)
    
    
    # Convert data to tensors
    t = np.linspace(0, 5, 1000)  # Example time data, adjust according to your problem
    t_tensor = torch.from_numpy(t).float().view(-1, 1).requires_grad_(True)
    t_ic = torch.zeros(1, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    ic = torch.from_numpy(init).float().view(1, -1)
    
    # Create PINN instance and define loss function
    pinn = PINN()
    loss_fn = nn.MSELoss()
    
    # Set training parameters
    epochs = 5000
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    loss_history = []
    
    pinn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        ind = torch.randperm(len(t_tensor))
        time = t_tensor[ind]
    
        # Forward pass
        out_ic = pinn(t_ic)
        loss_ic = loss_fn(out_ic, ic)
    
        out = pinn(time)
        x1 = out[:, 0].view(-1, 1)
        x2 = out[:, 1].view(-1, 1)
        y1 = out[:, 2].view(-1, 1)
        z = out[:, 3].view(-1, 1)
        grad_out = torch.ones_like(x1)
    
        # Compute gradients
        gradient1 = grad(x1, time, grad_outputs=grad_out, create_graph=True)[0]
        gradient2 = grad(x2, time, grad_outputs=grad_out, create_graph=True)[0]
        gradient3 = grad(y1, time, grad_outputs=grad_out, create_graph=True)[0]
        gradient4 = grad(z, time, grad_outputs=grad_out, create_graph=True)[0]
    
        # Compute partial differential equation (PDE) losses
        pde_x2 = 0.2 * x1 + (0.2 - 0.33) * x2 - gradient2
        pde_y1 = 0.33 * x2 - 2. * y1 - gradient3
        pde_z = 2. * y1 - 0.3 * z - gradient4
        loss_pdex1 = loss_fn(gradient1, torch.zeros_like(gradient1))
        loss_pdex2 = loss_fn(pde_x2, torch.zeros_like(pde_x2))
        loss_pdey1 = loss_fn(pde_y1, torch.zeros_like(pde_y1))
        loss_pdez = loss_fn(pde_z, torch.zeros_like(pde_z))
    
        # Total loss
        loss = loss_pdex1 + loss_pdex2 + loss_pdey1 + loss_pdez + loss_ic
        loss.backward()
        optimizer.step()
    
        loss_history.append(loss.item())
        if (epoch + 1) % (epochs / 20) == 0:
            print("Epoch:", epoch + 1, "Loss:", loss.item())

    plt.plot(loss_history)


    pinn.eval()
    print(pinn(t_ic))
    with torch.no_grad():
        pred = pinn(torch.from_numpy(t).float().view(-1,1))
        x1 = pred[:,0].detach().numpy()
        x2 = pred[:,1].detach().numpy()
        y1 = pred[:,2].detach().numpy()
        z = pred[:,3].detach().numpy()
    fig, ax = plt.subplots()
    ax.plot(t, x1)
    ax.plot(t, x2)
    ax.plot(t, y1)
    ax.plot(t, z)
    plt.show()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    