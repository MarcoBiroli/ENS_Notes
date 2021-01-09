from methods import GPE as solver
from visualizer import plotter as cplt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

'''
Start by defining the relevant parameters of the problem.
'''

dim = 2
xmin, xmax, ymin, ymax = -8, 8, -8, 8
bounds = [[xmin, xmax], [ymin, ymax]]
tmin = 0
tmax = 0
dt = 0.001
N = [129, 129]
im_iters = 2000

epsilon = 1.0
gamma = 1.0
interC = 200
v = 2.0
r0 = 2.0/50.0 ** (1.0/4.0)
omega_s = 1.0
V_s = 1.0/50.0**(1.0/4.0)
W_f = np.sqrt(2.0)

alpha = 4.0
beta = 1.0
x0 = 1.0

boundary_conditions = 'hard'

def potential(grids, t, alpha, beta, gamma, x0):
    return alpha * np.exp(-beta*((grids[0]-x0)**2+(grids[1])**2)) + 0.5*(grids[0]**2+gamma**2*grids[1]**2) # units of Adams: 0.25 instead of 0.5

print('epsilon: ', epsilon)
print('gamma: ', gamma)
print('kappa2: ', interC)

# Define the GPE with the relevant parameters.
GPE = solver.GPE_Solver(dim, bounds, tmin, tmax, dt, N, im_iters, \
    lambda grids, t : potential(grids, t, alpha, beta, gamma, x0), interC, boundary_conditions)

# Solve the problem.
mutab, utab, jtab = GPE.full_solve(save = (True, 'test_save.csv'))

# Plot the evolution of the chemical potential.
plt.plot(mutab)
plt.show()

print(mutab[-1])

# Plot the evolution of the wavefunction.
ani = cplt.plot(utab, dim = 2)

# Plot the evolution of the probability current.
x = np.linspace(xmin, xmax, N[0])
y = np.linspace(ymin, ymax, N[1])
xx, yy = np.meshgrid(x, y)

ani2 = cplt.quiver(xx, yy, jtab, grain = 2)
