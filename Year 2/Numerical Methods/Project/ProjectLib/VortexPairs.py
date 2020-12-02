from methods import GPE as solver
from visualizer import plotter as cplt
import numpy as np

dim = 2
xmin, xmax, ymin, ymax = -5, 5, -5, 5
bounds = [[xmin, xmax], [ymin, ymax]]
tmin = 0
tmax = 5
dt = 0.01
N = [100, 100]
im_iters = 100
interC = 500
boundary_conditions = 'periodic'
alpha = 30
beta = 3
v = 2

def vortex_potential(grids, t, alpha, beta, v, ymin, ymax):
    x, y = grids
    return (1/4 * (x**2 + y**2) + alpha * np.exp(- beta* x**2 - beta*(y - (ymin + (v * t - ymin)%(ymax - ymin)) )**2))

GPE = solver.GPE_Solver(dim, bounds, tmin, tmax, dt, N, im_iters, \
    lambda grids, t : vortex_potential(grids, t, alpha, beta, v, ymin, ymax), interC, boundary_conditions)

mutab, utab, jtab = GPE.full_solve()

ani = cplt.plot(utab)