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


#Neural Network structure for the PINN
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), # In PINN, dropout, can NOT be use
            nn.Linear(64, 32), nn.Tanh(), # and the activation function must have a smooth derivative
            nn.Linear(32, 16), nn.Tanh(), # so ReLU doesn't work
            nn.Linear(16, 4)
        )

    def forward(self, inp):
        return self.seq(inp)
    
    def add_pde_parameters(self, params : ArrayLike) -> None:
        '''
        Add the parameters of the pde as buffer to the neural network structure

        Parameters
        ----------
        params : ArrayLike
            numpy array containing the parameters
        Returns
        -------
        None
        '''
        assert len(params) == 4
        params = torch.from_numpy(params).float()
        self.register_buffer('params', params)
        pass
    
    def add_initial_condition(self, init : ArrayLike, t_ic_zero = None) -> None:
        '''
        Add the initial condition as buffers to the network structure.
        If t_in_zero is none the initial time will be automatically set to zero.

        Parameters
        ----------
        init : ArrayLike
            numpy array of the initial condition for the four states
        t_ic_zero : optional
            The default is None.

        Returns
        -------
        None
        '''
        assert len(init) == 4
        if t_ic_zero is None:
            self.register_buffer('t_ic', torch.zeros(1, dtype=torch.float, requires_grad=True))
        else:
            self.register_buffer('t_ic', torch.tensor(t_ic_zero, dtype=torch.float, requires_grad=True))
        init = torch.from_numpy(init).float()
        self.register_buffer('init', init)
        pass
    
    def train_step(self, t : torch.Tensor, optimizer, loss_fn) -> float:
        '''
        Perform a step for the training.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor for the domain. It must have the shape : Nx1 where N is the number of training points.
        optimizer : torch.optim
            Optimizer to use for the training.
        loss_fn : torch.nn
            Loss function to use for the training.

        Returns
        -------
        float
            The loss of the training step.
        '''
        self.train()
        optimizer.zero_grad()

        # Forward pass for the initial conditions
        out_ic = self(self.t_ic)
        loss_ic = loss_fn(out_ic, self.init)
        
        # Forwward pass in time domain
        out = self(t)
        x1 = out[:, 0].view(-1, 1)
        x2 = out[:, 1].view(-1, 1)
        y1 = out[:, 2].view(-1, 1)
        z = out[:, 3].view(-1, 1)
        grad_out = torch.ones_like(x1) # Define the starting gradient for backprop. using autograd.grad
        
        # Compute gradients
        dx1 = grad(x1, t, grad_outputs=grad_out, create_graph=True)[0]
        dx2 = grad(x2, t, grad_outputs=grad_out, create_graph=True)[0]
        dy1 = grad(y1, t, grad_outputs=grad_out, create_graph=True)[0]
        dz = grad(z, t, grad_outputs=grad_out, create_graph=True)[0]
        
        # Compute partial differential equation (PDE) and their losses
        pde_x1 = 0. - dx1
        pde_x2 = 0.2 * x1 + (0.2 - 0.33) * x2 - dx2
        pde_y1 = 0.33 * x2 - 2. * y1 - dy1
        pde_z = 2. * y1 - 0.3 * z - dz
        loss_pdex1 = loss_fn(pde_x1, torch.zeros_like(dx1))
        loss_pdex2 = loss_fn(pde_x2, torch.zeros_like(pde_x2))
        loss_pdey1 = loss_fn(pde_y1, torch.zeros_like(pde_y1))
        loss_pdez = loss_fn(pde_z, torch.zeros_like(pde_z))
        
        # Total loss and optim step
        loss = loss_pdex1 + loss_pdex2 + loss_pdey1 + loss_pdez + loss_ic
        loss.backward()
        optimizer.step()
        
        return loss.item() # Return the computed loss as float


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
    time = np.linspace(0, ub_time, 100)
    
    #Normalization factor
    cons = ub_time
    
    #Solution
    y = odeint(pde_scipy, init, time, args=(params,))
    #Normalized solution: limited time range (in [0,1]) to numerical stability of the neural network
    y_norm = odeint(pde_scipy, init, time/cons, args=(params*cons,)) #Simply a change of variable in a differential equation
    
    #Plot
    plots.plot_solution_scipy(time, y, y_norm)
    
    
    
    
    
    ##PINN solution    
    # Create PINN instance
    pinn = PINN()
    pinn.add_pde_parameters(params) # Add the parameters of the pde as buffers
    pinn.add_initial_condition(init) # Add the initial conditions, starting time by default is zero
    
    # Convert data to tensors
    # The input and target must be float numbers
    t = torch.from_numpy(time).float().view(-1, 1).requires_grad_(True)
  
    # Set training parameters
    epochs = 5000
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loss_history = []
    
    for epoch in range(epochs):
        
        loss = pinn.train_step(t, optimizer, loss_fn)
        loss_history.append(loss)
        
        if (epoch + 1) % (epochs / 20) == 0:
            print(f"{epoch/epochs*100:.1f}% -> Loss: ", loss)


    plots.plot_loss(loss_history)    

    plots.plot_solution_pinn(pinn, time)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    