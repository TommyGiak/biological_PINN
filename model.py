# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:30:08 2023

@author: Tommaso Giacometti
"""
import torch
from torch import nn, Tensor
from torch.autograd import grad
from numpy.typing import ArrayLike



#Neural Network structure for the PINN for the direct problem
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
    
    
    def train_step(self, t : Tensor, optimizer, loss_fn = None, lbfgs : bool = False) -> float:
        '''
        Perform a step for the training.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor for the domain. It must have the shape : Nx1 where N is the number of training points.
        optimizer : torch.optim
            Optimizer to use for the training.
        loss_fn : torch.nn
            Loss function to use for the training, by default (None) MSELoss will be used.
        lbfgs : bool, optional
            Parameter to use this function as closure() for the LBFGS optimizer training. DEFAULT False,
            set to True ONLY for the LBFGS otherwise the training will fail.

        Returns
        -------
        float
            The loss of the training step.
        '''
        if loss_fn is None:
            loss_fn = nn.MSELoss()
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
        pde_x2 = self.params[0] * x1 + (self.params[0] - self.params[1]) * x2 - dx2
        pde_y1 = self.params[1] * x2 - self.params[2] * y1 - dy1
        pde_z = 2 * self.params[2] * y1 - self.params[3] * z - dz
        loss_pdex1 = loss_fn(pde_x1, torch.zeros_like(dx1))
        loss_pdex2 = loss_fn(pde_x2, torch.zeros_like(pde_x2))
        loss_pdey1 = loss_fn(pde_y1, torch.zeros_like(pde_y1))
        loss_pdez = loss_fn(pde_z, torch.zeros_like(pde_z))
        
        # Total loss and optim step
        loss = loss_pdex1 + loss_pdex2 + loss_pdey1 + loss_pdez + loss_ic
        loss.backward()
        if not lbfgs: # In the LBFGS, the optimizer step is permormed in the LBFGS class
            optimizer.step()
        
        return loss # Return the computed loss as tensor
    
    
    def set_up_lbfgs(self, t : Tensor):
        self.register_buffer('time', t)
        optimizer = torch.optim.LBFGS(self.parameters())
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        
        def closure():
            return self.train_step(self.time, self.optimizer, self.loss_fn, lbfgs = True)
        
        return optimizer, closure


#Neural Network structure for the PINN for the inverse problem
#In this case we want to infer 2 parameters: lambda and gamma.
class PINN_inverse(nn.Module):
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
    
    
    def add_pde_parameters_known(self, params : ArrayLike) -> None:
        '''
        Add the parameters of the pde as buffer to the neural network structure.
        In this case the parameters must be nu and delta

        Parameters
        ----------
        params : ArrayLike
            numpy array containing the parameters nu and delta
        Returns
        -------
        None
        '''
        assert len(params) == 2
        params = torch.from_numpy(params).float()
        self.register_buffer('params_known', params)
        pass
    
    def add_pde_parameters_to_infer(self, params : ArrayLike) -> None:
        '''
        Add the parameters of the pde to infer as torch.Parameters to the neural network structure.
        In this case the parameters must be lambda (lam) and gamma.

        Parameters
        ----------
        params : ArrayLike
            numpy array containing the parameters lambda and gamma
        Returns
        -------
        None
        '''
        assert len(params) == 2
        params = torch.from_numpy(params).float()
        params = nn.Parameter(params)
        self.register_parameter('params_to_infer', params)
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
    
    
    def train_step(self, t : Tensor, data : Tensor, optimizer, loss_fn = None, lbfgs : bool = False) -> float:
        '''
        Perform a step for the training.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor for the domain. It must have the shape : Nx1 where N is the number of training points.
        optimizer : torch.optim
            Optimizer to use for the training.
        loss_fn : torch.nn
            Loss function to use for the training, by default (None) MSELoss will be used.
        lbfgs : bool, optional
            Parameter to use this function as closure() for the LBFGS optimizer training. DEFAULT False,
            set to True ONLY for the LBFGS otherwise the training will fail.

        Returns
        -------
        float
            The loss of the training step.
        '''
        if loss_fn is None:
            loss_fn = nn.MSELoss()
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
        pde_x2 = self.params_to_infer[0] * x1 + (self.params_to_infer[0] - self.params_known[0]) * x2 - dx2
        pde_y1 = self.params_known[0] * x2 - self.params_to_infer[1] * y1 - dy1
        pde_z = 2 * self.params_to_infer[1] * y1 - self.params_known[0] * z - dz
        loss_pdex1 = loss_fn(pde_x1, torch.zeros_like(dx1))
        loss_pdex2 = loss_fn(pde_x2, torch.zeros_like(pde_x2))
        loss_pdey1 = loss_fn(pde_y1, torch.zeros_like(pde_y1))
        loss_pdez = loss_fn(pde_z, torch.zeros_like(pde_z))
        
        # Data loss
        data_t = data[:,0:1]
        data_points = data[:,1:]
        out_data = self(data_t)
        loss_data = loss_fn(out_data[:,2:4], data_points[:,2:4])
        
        # Total loss and optim step
        loss = loss_pdex1 + loss_pdex2 + loss_pdey1 + loss_pdez + loss_ic + 0.5*loss_data # The 0.5 factor is to get smoother solution and avoid overfitting with datapoints
        loss.backward()
        if not lbfgs: # In the LBFGS, the optimizer step is permormed in the LBFGS class
            optimizer.step()
        
        return loss # Return the computed loss as tensor
    
    
    def set_up_lbfgs(self, t : Tensor, data : Tensor):
        self.register_buffer('time', t)
        self.register_buffer('data', data)
        #With noisy data lbfgs is likely to return nan result due to convergence problem
        #for this reason I reduced the history size, but it can't always solve the problem.
        #If the problem persist the lbfgs training can be simpli avoided
        optimizer = torch.optim.LBFGS(self.parameters(), history_size=10) 
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        
        def closure():
            return self.train_step(self.time, self.data, self.optimizer, self.loss_fn, lbfgs = True)
        
        return optimizer, closure


