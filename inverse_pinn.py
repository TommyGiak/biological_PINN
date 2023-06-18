# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:33:15 2023

@author: Tomamso Giacometti
"""
import torch
from torch import nn
import numpy as np
import plots
from scipy.integrate import odeint
from model import PINN_inverse
from utils import pde_scipy, inverse_pinn_data_gen


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Data generation for the inverse problem
#Parameters 
lam = 0.2
nu = 0.33
gamma = 2.
delta = 0.33
params = np.array([lam,nu,gamma,delta])
params_known = np.array([nu,delta])
params_to_infer = np.array([1,1]) # Starting values for the parameters to infer 

#Initial conditions
x1 = 6
x2 = 5
y1 = 0
z = 0
init = np.array([x1,x2,y1,z])

#Time domain
ub_time = 20. #Upper bound for the time domain during the integration
time = np.linspace(0, ub_time, 200)

#Solution
y = odeint(pde_scipy, init, time, args=(params,))


##PINN solution    
# Create PINN instance
torch.manual_seed(0)
pinn = PINN_inverse().to(device)
pinn.add_pde_parameters_known(params_known) # Add the parameters of the pde as buffers
pinn.add_pde_parameters_to_infer(params_to_infer) # Add the parameters of the pde as torch.Parameters
pinn.add_initial_condition(init) # Add the initial conditions, starting time by default is zero

# Convert data to tensors
# The input and target must be float numbers
t = torch.from_numpy(time).float().view(-1, 1).to(device).requires_grad_(True)
np.random.seed(123)
data = inverse_pinn_data_gen(init, time, params, True, std=0.2) # Data with noise
data = torch.from_numpy(data).float()

# Set training parameters
epochs = 10000
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
loss_history = []

#Training performed with Adam (or the one setted above)
print(f'Adam training ({epochs} epochs):')
for epoch in range(epochs):
    loss = pinn.train_step(t, data, optimizer, loss_fn).item() # Single train step defined in the PINN class
    loss_history.append(loss)
    if (epoch + 1) % (epochs / 20) == 0:
        print(f"{epoch/epochs*100:.1f}% -> Loss: ", loss)
        
# Setting up the network for the LBFGS training
optimizer, closure = pinn.set_up_lbfgs(t, data)

# Training performed with LBFGS
epochs = 500
print(f'LBFGS training ({epochs} epochs):')
for epoch in range(epochs):
    optimizer.step(closure)
    loss = closure().item()
    loss_history.append(loss)
    if (epoch + 1) % (epochs / 20) == 0:
        print(f"{epoch/epochs*100:.1f}% -> Loss: ", loss)
        
#Inferred parameters
print(f'lambda = {pinn.params_to_infer[0].item():.5f}, gamma = {pinn.params_to_infer[1].item():.5f}')

# Plots
plots.plot_solution_scipy(time, y)
plots.plot_loss(loss_history)
plots.plot_solution_pinn_inverse(pinn, time, data)
    
    
    
    
