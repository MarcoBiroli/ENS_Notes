from methods import GPE as solver
from visualizer import plotter as cplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from Quant import wavefunc as wf
from methods import TSSP as solver2
from pylab import cm
'''
Start by defining tdhe relevant parameters of the problem.
'''


dim = 2
xmin, xmax, ymin, ymax = -5, 5, -5, 5
bounds = [[xmin, xmax], [ymin, ymax]]
tmin = 0
tmax = 5
dt = 0.01
N = [200, 200]
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

#GPE = solver2.GPE_Solver(dim, bounds, tmin, tmax, dt, N, im_iters, \
#    lambda grids, t : vortex_potential(grids, t, alpha, beta, v, ymin, ymax), interC, boundary_conditions)

# Solve the problem.
mutab, utab, jtab = GPE.full_solve(save=(False, './init_states/vortex_pairs_init.npy'))

np.save('./data_save/mutab.npy', np.array(mutab))

#GPE.load_init_state('vortex_pairs_init.csv')
#mu, E = GPE.compute_mu()
#print("mu = {}, energy = {}".format(mu, E))

#np.save('./data_save/totu1000-60.npy', np.array(utab))
