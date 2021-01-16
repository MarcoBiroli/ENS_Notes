from methods import GPE as solver
from visualizer import plotter as cplt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from Quant import wavefunc as wf
from methods import TSSP as solver2

'''
Start by defining the relevant parameters of the problem.
'''

dim = 2
xmin, xmax, ymin, ymax = -5, 5, -5, 5
bounds = [[xmin, xmax], [ymin, ymax]]
tmin = 0
tmax = 5
dt = 0.01
N = [300, 300]
im_iters = 2000
interC = 500
boundary_conditions = 'periodic'
alpha = 60
beta = 3
v = 2
    
def vortex_potential(grids, t, alpha, beta, v, ymin, ymax):
    '''
    Define the potential considered.
    '''
    x, y = grids
    return (1/4 * (x**2 + y**2) + alpha * np.exp(- beta* x**2 - beta*(y - (ymin + (v * t - ymin)%(ymax - ymin)) )**2))

# Define the GPE with the relevant parameters.
#GPE = solver.GPE_Solver(dim, bounds, tmin, tmax, dt, N, im_iters, \
#    lambda grids, t : vortex_potential(grids, t, alpha, beta, v, ymin, ymax), interC, boundary_conditions)

GPE = solver2.GPE_Solver(dim, bounds, tmin, tmax, dt, N, im_iters, \
    lambda grids, t : vortex_potential(grids, t, alpha, beta, v, ymin, ymax), interC, boundary_conditions)

#cplt.plot_cur(wf.Wavefunction(GPE.U, GPE.dim, GPE.N, GPE.normalization), 2, bounds)

mutab, utab = GPE.full_solve()
#jtab = [psi.computeCurrent(GPE.Ops) for psi in utab]

mutab = np.real(mutab)

print(mutab[-1])
plt.loglog(mutab)
plt.show()

#cplt.plot_cur(utab[0], 2, bounds)

#ani = cplt.plot(utab, dim = 2)

#x = np.linspace(xmin, xmax, N[0])
#y = np.linspace(ymin, ymax, N[1])
#xx, yy = np.meshgrid(x, y)

#ani2 = cplt.quiver(xx, yy, jtab, grain = 5)