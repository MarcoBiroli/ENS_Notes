from methods import GPE as solver
from visualizer import plotter as cplt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from Quant import wavefunc as wf

'''
Start by defining the relevant parameters of the problem.
'''

dim = 2
xmin, xmax, ymin, ymax = -5, 5, -5, 5
bounds = [[xmin, xmax], [ymin, ymax]]
tmin = 0
tmax = 3
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
mutab, utab, jtab = GPE.full_solve(save=(True, './init_states/vortex_pairs_init.npy'))

#GPE.load_init_state('vortex_pairs_init.csv')
#mu, E = GPE.compute_mu()
#print("mu = {}, energy = {}".format(mu, E))

#utab, jtab = GPE.real_time_evolution()

#reu = [np.real(psi.u) for psi in utab]
#imu = [np.imag(psi.u) for psi in utab]

#reu = np.array(reu)
#imu = np.array(imu)

#np.save('./data_save/reu.npy', reu)
#np.save('./data_save/imu.npy', imu)

#np.save('./data_save/totu.npy', np.array(utab))

#reu = np.load('reu.npy')
#imu = np.load('imu.npy')

#utab = [wf.Wavefunction(reu[i] + 1j*imu[i], GPE.dim, GPE.N, GPE.normalization) for i in range(len(reu))]

#utab = np.load('./data_save/totu.npy', allow_pickle=True)

#utab = [wf.Wavefunction(u, GPE.dim, GPE.N, GPE.normalization) for u in tmp]

#np.savetxt('Re(psi_t=3s).csv', np.real(GPE.psi.u), delimiter=',')
#np.savetxt('Im(psi_t=3s).csv', np.imag(GPE.psi.u), delimiter=',')
#GPE.load_init_state('Re(psi_t=3s).csv', 'Im(psi_t=3s).csv')

cplt.plot_cur(GPE.psi, 2, bounds)

x = np.linspace(xmin, xmax, N[0])
y = np.linspace(ymin, ymax, N[1])
xx, yy = np.meshgrid(x, y)
cplt.quiver_cur(xx, yy, GPE.psi.computeCurrent(GPE.Ops, fact = 1/40))

integral, path = GPE.winding_number([0, 0], [1, 1])

cplt.plot_cur(GPE.psi, 2, bounds, path, GPE.normalization)


# Plot the evolution of the chemical potential.
plt.plot(mutab)
plt.show()

print(mutab[-1])

# Plot the evolution of the wavefunction.
#ani = cplt.plot(utab, dim = 2)

# Plot the evolution of the probability current.
#x = np.linspace(xmin, xmax, N[0])
#y = np.linspace(ymin, ymax, N[1])
#xx, yy = np.meshgrid(x, y)

#ani2 = cplt.quiver(xx, yy, jtab, grain = 2)
