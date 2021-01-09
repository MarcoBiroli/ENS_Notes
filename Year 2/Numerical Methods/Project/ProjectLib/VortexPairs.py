from methods import GPE as solver
from visualizer import plotter as cplt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

'''
Start by defining the relevant parameters of the problem.
'''

dim = 2
xmin, xmax, ymin, ymax = -5, 5, -5, 5
bounds = [[xmin, xmax], [ymin, ymax]]
tmin = 0
tmax = 5
dt = 0.01
N = [100, 100]
im_iters = 2000
interC = 500
boundary_conditions = 'periodic'
alpha = 30
beta = 3
v = 2

def vortex_potential(grids, t, alpha, beta, v, ymin, ymax):
    '''
    Define the potential considered.
    '''
    x, y = grids
    return (1/4 * (x**2 + y**2) + alpha * np.exp(- beta* x**2 - beta*(y - (ymin + (v * t - ymin)%(ymax - ymin)) )**2))

# Define the GPE with the relevant parameters.
GPE = solver.GPE_Solver(dim, bounds, tmin, tmax, dt, N, im_iters, \
    lambda grids, t : vortex_potential(grids, t, alpha, beta, v, ymin, ymax), interC, boundary_conditions)

# Solve the problem.
#mutab, utab, jtab = GPE.full_solve(save=(True, 'vortex_pairs_init.csv'))

GPE.load_init_state('vortex_pairs_init.csv')
utab, jtab = GPE.real_time_evolution()

# Plot the evolution of the chemical potential.
#plt.plot(mutab)
#plt.show()

#print(mutab[-1])

# Plot the evolution of the wavefunction.
ani = cplt.plot(utab, dim = 2)

# Plot the evolution of the probability current.
x = np.linspace(xmin, xmax, N[0])
y = np.linspace(ymin, ymax, N[1])
xx, yy = np.meshgrid(x, y)

ani2 = cplt.quiver(xx, yy, jtab, grain = 2)
