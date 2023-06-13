# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:42:45 2023

@author: Tommaso Giacometti
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import odeint
import plots
import deepxde as dde
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
    dz = 2*states[2]*params[2] - states[3]*params[3]
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
    
    
    ##PINN solution
    # Definition of the pde system
    def pde_pinn(t, states):
        x1 = states[:,0:1]
        x2 = states[:,1:2]
        y1 = states[:,2:3]
        z = states[:,3:4]
        dx2 = dde.grad.jacobian(states, t, i = 1)
        dx1 = dde.grad.jacobian(states, t, i = 0)
        dy1 = dde.grad.jacobian(states, t, i = 2)
        dz = dde.grad.jacobian(states, t, i = 3)
        return [dx1 - 0.,
                dx2 - params[0] * x1 - x2 * (params[0]-params[1]),
                dy1 - x2 * params[1] + y1 * params[2],
                dz - 2 * params[2] * y1 + params[3] * z
                ]  
    
    #Geometry of the problem and definition
    t = dde.geometry.TimeDomain(0, ub_time)
    data = dde.data.PDE(t, pde_pinn, [], 3000, 2, num_test=3000)

    #Neural Network structure
    layer_size = [1] + [64] * 2 + [4]
    activation = "tanh"
    initializer = "Glorot normal"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    #Setup of the initial conditions
    def boundary(_, on_initial): # function to return the right value (it makes sense when there is another variable more than the time)
        return on_initial
    ic1 = dde.IC(t, lambda _: np.full(1, init[0]), boundary, component=0)
    ic2 = dde.IC(t, lambda _: np.full(1, init[1]), boundary, component=1)
    ic3 = dde.IC(t, lambda _: np.full(1, init[2]), boundary, component=2)
    ic4 = dde.IC(t, lambda _: np.full(1, init[3]), boundary, component=3)
    data = dde.data.PDE(t, pde_pinn, [ic1, ic2, ic3, ic4], 2000, 2, num_test=100)
    
    #Model definition and training with Adam and LBFGS
    model = dde.Model(data, net)
    model.compile("adam", lr=0.002)
    model.train(epochs=2000)
    model.compile('L-BFGS-B')
    model.train()
    
    #Plot
    plots.plot_solution_scipy(time, y, y_norm)
    t = np.linspace(0, 2, 100)
    t = t.reshape(100, 1)
    sol_pred = model.predict(t)
    x1_pred = sol_pred[:, 0:1]
    x2_pred = sol_pred[:, 1:2]
    y1_pred = sol_pred[:, 2:3]
    z_pred = sol_pred[:, 3:4]
    
    plt.plot(t, x1_pred, color="red", linestyle="dashed", label="x1_pred")
    plt.plot(t, x2_pred, color="orange", linestyle="dashed", label="x2_pred")
    plt.plot(t, y1_pred, color="orange", linestyle="dashed", label="y1_pred")
    plt.plot(t, z_pred, color="orange", linestyle="dashed", label="z_pred")
    plt.legend()
    plt.show()
    
    
    

if __name__ == '__main__':
    main()