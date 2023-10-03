# PINN for biological systems
This repository tries to solve systems of differential equation for biological systems using PINNs (Physics Informed Neural Networks).\
The scripts are implemented using [Pytorch](https://pytorch.org) or the <em>ad hoc</em> library [DeepXDE](https://github.com/lululxvi/deepxde).

## Organization of the repository
Up to now therepository contain three main files: <em>first_pinn.py, pinn_deepxde.py</em> and <em>inverse_pinn.py</em>.\
The first two scripts solve the exact same problem with same parameters, the only difference is that in <em>first_pinn.py</em> I solved the ploblem 'from scratch' using only Pytorch, while in <em>pinn_deepxde.py</em> the problem is solved using the DeepXDE library.\
The third file (<em>inverse_pinn.py</em>) inplements the inverse pinn problem where the aim is to infer some parameters not known using data.

## Problem
The differential equation system that I solved in the two scripts is the following:
```math
\begin{cases}
\frac{d}{dt}x_1 = 0\\
\frac{d}{dt}x_2 = \lambda x_1 + (\lambda - \nu) x_2\\
\frac{d}{dt}y_1 = \nu x_2 - \gamma y_1\\
\frac{d}{dt}z = 2\gamma y_1 - \delta z
\end{cases}
```
This system of equation is a semplified version of the evolution of stem cells. For further details you can refer to this [paper](https://pubmed.ncbi.nlm.nih.gov/28616066/).
The values of the parameters that where used are:
- $\lambda = 0.2$
- $\nu = 0.33$
- $\gamma = 2.0$
- $\delta = 0.33$

With the following initial conditions:
- $x_1 = 6$
- $x_2 = 5$
- $y_1 = 0$
- $z = 0$

### Inverse problem
In the inverse problem, so in the <em>inverse_pinn.py</em> file, $\nu$ and $\delta$ were assumed already known and $\lambda$ and $\gamma$ unknown. The aim was to infer this two parameter using noisy data generated with [scipy.integrate.odeint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html), the noise follows a Gaussian distribution with $\mu = 0$ and $\sigma = 0.5$.\
The generated data used by the neural network for the training include only the solution for $y_1$ and $z$. Nonetheless the parameters were succesfully inferred.

## How to use
To run the simulation you have to clone this repository in an empty folder: from the terminal (inside an empty folder) you can use the following command
```shell
git clone https://github.com/TommyGiak/biological_PINN.git
```
then you can just run with python the script:
```shell
python first_pinn.py
```
or
```shell
python pinn_deepxde.py
```
for the direct problem,
```shell
python inverse_pinn.py
```
for the inverse one.
