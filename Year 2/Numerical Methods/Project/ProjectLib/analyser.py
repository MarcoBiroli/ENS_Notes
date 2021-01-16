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
N = [300, 300]
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

mutab = np.load('./data_save/mutab.npy')
utab = np.load('./data_save/utab.npy', allow_pickle=True)
utab = utab[:500]
jtab = [psi.computeCurrent(GPE.Ops) for psi in utab]
phitab = [np.angle(psi.u) for psi in utab]

'''
phase_slip = []
for psi in utab:
    phi = np.angle(psi.u).reshape(GPE.N[0], GPE.N[1])
    dS = np.max([phi[(i+1)%GPE.N[1], GPE.N[0]//2] - phi[i, GPE.N[0]//2] for i in range(GPE.N[1])])
    phase_slip.append(dS)

plt.plot(GPE.t, phase_slip)
plt.show()
'''

def plot_quiver(t = None, grain = 2, fact = 1/40, show = True):
    x = np.linspace(xmin, xmax, N[0])
    y = np.linspace(ymin, ymax, N[1])
    xx, yy = np.meshgrid(x, y)

    if t is None:
        ani2 = cplt.quiver(xx, yy, jtab, grain = grain, show = show)
    else:
        cplt.quiver_cur(xx, yy, utab[t].computeCurrent(GPE.Ops, fact = fact), grain = grain, show = show)

def plot_state(t = None, path = None, show = False):
    if t is None:
        ani = cplt.plot(utab, dim = 2)
    else:
        cplt.plot_cur(utab[t], 2, bounds, path=path, normalization=GPE.normalization, show = show)

def compute_winding(center1, center2, rad):
    vortex_left = []
    vortex_right = []

    for u in utab[:500]:
        integral, path1 = GPE.winding_number(center1, rad, psi = u)
        vortex_left.append(2*integral/np.pi)
        integral, path2 = GPE.winding_number(center2, rad, psi = u)
        vortex_right.append(2*integral/np.pi)
    
    return vortex_left, vortex_right, path1 + path2


#plt.title(r'Contour plot of $|\psi(t = 3s)|^2$ with the winding number integration contour.')

def plot_winding(vortex_left, vortex_right, center, rad, show = True, legend = None):
    fig = plt.figure(figsize=(6, 6))

    plt.xlabel('Time (s)', labelpad=10)
    plt.ylabel(r'$w/(\pi/2)$', labelpad=10)
    plt.title(r'Evolution of the winding number around $(\pm {}, {})$'.format(center[0], center[1]) + '\nin a rectangular box of size ({}, {}).'.format(rad[0], rad[1]))
    if legend is None:
        plt.plot(GPE.t, vortex_left, label = 'vortex left')
        plt.plot(GPE.t, vortex_right, label = 'vortex right')
    else:
        plt.plot(GPE.t, vortex_left, label = legend[0])
        plt.plot(GPE.t, vortex_right, label = legend[1])
    #plt.plot(GPE.t, vortex_left2, label = 'vortex left 2')
    #plt.plot(GPE.t, vortex_right2, label = 'vortex right 2')
    plt.legend(loc='best', frameon=False, fontsize=14)
    if show:
        plt.show()


def plot_cut(ycut):
    fig, ax1 = plt.subplots(figsize =(6,6))

    nsq = utab[299].normsq_pointwise.reshape(GPE.N[0], GPE.N[1])
    cur = utab[299].computeCurrent(GPE.Ops, fact = 1).reshape(GPE.N[0], GPE.N[1], 2)


    yidx = int((ycut - GPE.bounds[1][0])/GPE.normalization[1])

    color = 'tab:red'

    ax1.set_xlabel('x position')
    ax1.set_ylabel(r'$|\psi|^2$', color = color)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.title(r'$|\psi(t = 3s)|^2$ and $|j(t = 3s)|^2$ along a cut at $y = {}$'.format(ycut))

    top = np.max(nsq[yidx, :])
    ax1.vlines([-1, 1], 0, top, linestyle = '--', label = 'vortices center')
    ax1.plot(GPE.spaces[0], nsq[yidx, :], label = r'$|\psi|^2$', color = color)

    ax1.legend()

    ax2 = ax1.twinx() 
    color = 'tab:blue'

    ax2.set_ylabel(r'|j|^2', color = color)
    ax2.plot(GPE.spaces[0], cur[yidx, :, 0]**2 + cur[yidx, :, 1]**2, label = r'$|j|^2$')
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.legend(frameon = False)

    plt.show()

print(mutab[-1])
plt.title('Evolution of the energy and chemical potential during the imaginary time evolution.')
plt.loglog(mutab[:, 0], label = r'$\mu$')
plt.loglog(mutab[:, 1], label = r'$E$')
plt.xlabel('Iterations')
plt.ylabel('Energy')
plt.legend()
plt.show()

'''
## Vortex nb.1
vleft, vright, path = compute_winding([-1, 2], [1, 2], [0.5, 1.25])
plot_winding(vleft, vright, [1,2], [0.5, 1.25], show = False, legend = [r'v_1^L', r'v_1^R'])

print('1 Done')

## Vortex nb.2
vleft, vright, path = compute_winding([-0.75, 3.75], [0.75, 3.75], [0.5, 0.5])
plot_winding(vleft, vright, [0.75, 3.75], [0.55, 0.5], show = False, legend = [r'v_2^L', r'v_2^R'])

print('2 Done')

## Vortex nb.3
vleft, vright, path = compute_winding([-0.5, -2.5], [0.5, -2.5], [0.75, 0.75])
plot_winding(vleft, vright, [0.5, -2.5], [0.75, 0.75], show = False, legend = [r'v_3^L', r'v_3^R'])

print('3 Done')

plt.title('Winding numbers of all three pairs of vortices along time.')
plt.show()

'''

'''
vortex1_pos = [\
    [0.8, 1.1], \
    [0.9, 1.4], \
    [0.8, 1.8], \
    [0.8, 2], \
    [1, 2.5], \
    [1, 2.7], \
    [1.25, 3.36]]

vortex2_pos = [\
    [0.35, 2.9],\
    [0.8, 3.5], \
    [1.35, 4.1]]

vortex3_pos = [\
    [0.25, -3.1],\
    [0.75, -2.4]]

color = 'red'
plt.text(vortex1_pos[0][0],vortex1_pos[0][1], 't = 1s ', horizontalalignment = 'right', color = color)
plt.text(-vortex1_pos[0][0],vortex1_pos[0][1], ' t = 1s', horizontalalignment = 'left', color = color)
plt.plot([p[0] for p in vortex1_pos], [p[1] for p in vortex1_pos], label = 'vortex 1', color = color)
plt.plot([-p[0] for p in vortex1_pos], [p[1] for p in vortex1_pos], color = color)
plt.text(vortex1_pos[-1][0],vortex1_pos[-1][1], 't = 4s ', horizontalalignment = 'right', color = color)
plt.text(-vortex1_pos[-1][0],vortex1_pos[-1][1], ' t = 4s', horizontalalignment = 'left', color = color)

color = 'blue'
plt.text(0,vortex2_pos[0][1], 't = 1.5s ', horizontalalignment = 'center', color = color)
plt.plot([p[0] for p in vortex2_pos], [p[1] for p in vortex2_pos], label = 'vortex 2', color = color)
plt.plot([-p[0] for p in vortex2_pos], [p[1] for p in vortex2_pos], color = color)
plt.text(vortex2_pos[-1][0],vortex2_pos[-1][1], 't = 2.5s ', horizontalalignment = 'right', color = color)
plt.text(-vortex2_pos[-1][0],vortex2_pos[-1][1], ' t = 2.5s', horizontalalignment = 'left', color = color)

color = 'green'
plt.text(0,vortex3_pos[0][1], 't = 3.5s ', horizontalalignment = 'center', color = color)
plt.plot([p[0] for p in vortex3_pos], [p[1] for p in vortex3_pos], label = 'vortex 3', color = color)
plt.plot([-p[0] for p in vortex3_pos], [p[1] for p in vortex3_pos], color = color)
plt.text(vortex3_pos[-1][0],vortex3_pos[-1][1], 't = 4s ', horizontalalignment = 'right', color = color)
plt.text(-vortex3_pos[-1][0],vortex3_pos[-1][1], ' t = 4s', horizontalalignment = 'left', color = color)

plt.title('Vortices creation, destruction and trajectories.')
plt.xlabel('x position')
plt.ylabel('y position')

plt.legend(loc = 'best')
plt.show()

'''

'''
#plot_state(t = 100, show = False)
#plt.title(r'Probability density $|\psi(t = 1s)|^2$')
plot_quiver(t = 100, grain = 2, fact = 1/20, show = False)
plt.title(r'Velocity field $j(t = 1s)$')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()

#plot_state(t = 150, show = False)
#plt.title(r'Probability density $|\psi(t = 1.5s)|^2$')
plot_quiver(t = 150, grain = 2, fact = 1/20, show = False)
plt.title(r'Velocity field $j(t = 1.5s)$')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()

#plot_state(t = 200, show = False)
#plt.title(r'Probability density $|\psi(t = 2s)|^2$')
plot_quiver(t = 200, grain = 2, fact = 1/20, show = False)
plt.title(r'Velocity field $j(t = 2s)$')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()

#plot_state(t = 250, show = False)
#plt.title(r'Probability density $|\psi(t = 2.5s)|^2$')
plot_quiver(t = 250, grain = 2, fact = 1/20, show = False)
plt.title(r'Velocity field $j(t = 2.5s)$')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()

#plot_state(t = 300, show = False)
#plt.title(r'Probability density $|\psi(t = 3s)|^2$')
plot_quiver(t = 300, grain = 2, fact = 1/20, show = False)
plt.title(r'Velocity field $j(t = 3s)$')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()

#plot_state(t = 350, show = False)
#plt.title(r'Probability density $|\psi(t = 3.5s)|^2$')
plot_quiver(t = 350, grain = 2, fact = 1/20, show = False)
plt.title(r'Velocity field $j(t = 3.5s)$')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()

#plot_state(t = 400, show = False)
#plt.title(r'Probability density $|\psi(t = 4s)|^2$')
plot_quiver(t = 400, grain = 2, fact = 1/20, show = False)
plt.title(r'Velocity field $j(t = 4s)$')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()
'''