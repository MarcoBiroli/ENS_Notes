from methods import GPE as solver
from visualizer import plotter as cplt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

'''
Start by defining relevant parameters. (I put them at random here because I didn't check them in the paper yet).
'''

dim = 1
xmin, xmax = -5, 5
bounds = [[xmin, xmax]]
tmin = 0
tmax = 100
dt = 0.01
N = [1000]
im_iters = 1000
interC = 6000
boundary_conditions = 'periodic'
U0 = 39
Us = 6
w0 = 5
lambd = 0.812
v = 0.21

def hawking_potential(grids, t, U0, Us, w0, lambd, v, xmin, xmax):
    '''
    Define the potential of the problem.
    '''
    x = grids
    x0 = np.pi * w0**2 / lambd
    wx = w0 * np.sqrt( 1 + (x/x0)**2 )
    pos = (v*t - xmin)%(xmax - xmin) + xmin
    if t != 0: # At time not 0 the laser is on.
        return U0 * (1 - (w0/wx)**2) - Us * (x > pos)
    else: # Initially the laser is off.
        return U0 * (1 - (w0/wx)**2)

# Define the GPE with the relevant parameters.
GPE = solver.GPE_Solver(dim, bounds, tmin, tmax, dt, N, im_iters, \
    lambda grids, t : hawking_potential(grids, t, U0, Us, w0, lambd, v, xmin, xmax), interC, boundary_conditions)

# Solve the problem.
mutab, utab, jtab = GPE.full_solve()

# Plot the evolution of the chemical potential.
plt.plot(mutab)
plt.show()

# Plot the evolution of the wavefunction.
ani = cplt.plot(utab, dim = 1)