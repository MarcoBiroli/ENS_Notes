import numpy as np
import matplotlib as mpl
from pylab import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')

def plot_cur(psi, dim, bounds, path = None, normalization = None, show = True):
    
    fig, ax = plt.subplots(1,1, figsize=(6, 6))

    xtab = np.linspace(bounds[0][0], bounds[0][1], psi.N[0])
    ytab = np.linspace(bounds[1][0], bounds[1][1], psi.N[1])

    X, Y = np.meshgrid(xtab, ytab)

    p = psi.normsq_pointwise.reshape(psi.N[0], psi.N[1])

    print(p.shape)

    if dim == 2:
        #img = ax.imshow(psi.normsq_pointwise.reshape(psi.N[0], psi.N[1]), origin = 'lower', \
        #    extent=[bounds[0][0],bounds[0][1],bounds[1][0], bounds[1][1]])
        CS = ax.contour(X, Y, psi.normsq_pointwise.reshape(psi.N[0], psi.N[1]), levels = 20, origin = 'lower', \
            extent=[bounds[0][0],bounds[0][1],bounds[1][0], bounds[1][1]])
    if dim == 1:
        ax.plot(psi.normsq_pointwise)

    if path != None:
        #print(path, normalization)
        xtab = [bounds[0][0] + point[0]*normalization[0] for point in path]
        ytab = [bounds[0][0] + point[1]*normalization[1] for point in path]
        #print(xtab, ytab)
        ax.scatter(xtab, ytab, color = 'red', s = 1)

    if dim == 2:
        fig.colorbar(CS)
        #ax.clabel(CS, inline=1, fontsize=8)
    if show:
        plt.show()

def plot(psi_list, dim):
    '''
    Plot the evolution in time of the norm squared of the wavefunction.
    '''
    fig = plt.figure(figsize = (6, 6))

    if dim == 2:
        im = plt.imshow(psi_list[0].normsq_pointwise.reshape(psi_list[0].N[0], psi_list[0].N[1]), origin = 'lower', animated = True)
    if dim == 1:
        im, = plt.plot(psi_list[0].normsq_pointwise, animated = True)

    def updatefig(frame, *args):
        if dim == 2:
            im.set_array(psi_list[frame].normsq_pointwise.reshape(psi_list[frame].N[0], psi_list[frame].N[1]))
        if dim == 1:
            im.set_data(list(range(len(psi_list[frame].normsq_pointwise))), psi_list[frame].normsq_pointwise)
        return im,
    
    ani = animation.FuncAnimation(fig, updatefig, frames = len(psi_list), interval=50, blit=True)
    if dim == 2:
        plt.colorbar()
    plt.show()
    return ani

def quiver_cur(X, Y, j, grain = 1, show = True):
    _, ax = plt.subplots(1,1, figsize=(6, 6))

    ax.quiver(X[::grain, ::grain], Y[::grain, ::grain], j[:, 0].reshape(X.shape)[::grain, ::grain], \
                   j[:, 1].reshape(X.shape)[::grain, ::grain], units='inches', angles = 'xy', \
                       scale_units = 'xy', scale = 1)
    if show:
        plt.show()


def quiver(X, Y, jlist, grain = 1, show = True):
    '''
    Plot the evolution in time of the probability current.
    '''
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    Q = ax.quiver(X[::grain, ::grain], Y[::grain, ::grain], jlist[0][:, 0].reshape(X.shape)[::grain, ::grain], \
                   jlist[0][:, 1].reshape(X.shape)[::grain, ::grain], units='inches', angles = 'xy', \
                       scale_units = 'xy', scale = 1)

    def update_quiver(frame, Q, X, Y):
        print(frame)
        Q.set_UVC(jlist[frame][:, 0].reshape(X.shape)[::grain, ::grain], jlist[frame][:, 1].reshape(X.shape)[::grain, ::grain])
        return Q,

    ani = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y), frames = len(jlist),
                                  interval=50, blit=False)
    
    if show:
        plt.show()
    return ani