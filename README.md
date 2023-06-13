# PINN for biological systems
In Tthis repository I will try to solve systems of differential equation for biological systems using PINNs (Physics Informed Neural Networks).\
The scripts are implemented using [Pythorc](https://pytorch.org) or the <em>ad hoc</em> library [DeepXDE](https://github.com/lululxvi/deepxde).

## Organization of the repository
Up to now therepository contain two main files: <em>first_pinn.py</em> and <em>pinn_deepxde.py</em>.\
They solve the exact same problem with same parameters, the only difference is that in <em>first_pinn.py</em> I solved the ploblem 'from scratch' using only Pytorch, while in <em>pinn_deepxde.py</em> the problem is solved using the DeepXDE library.

## Problem
The differential equation system that I solved in the two scripts is the following:
```math
\begin{cases}
\frac{d}{dt}x_1 = 0 \\
\frac{d}{dt}x_2 = \lambda x_1 + (\lambda - \nu x_2\\
\frac{d}{dt}y_1 = \nu x_2 - \gamma y_1 \\
\frac{d}{dt}z = 2\gamma y_1 - \delta z
\end{cases}
```
This system of equation is a semplified version of the evolution of stem cells. For further details you can refer to this [paper](https://pubmed.ncbi.nlm.nih.gov/28616066/).

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
