import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot(psi_list):

    fig = plt.figure(figsize = (6, 6))

    im = plt.imshow(psi_list[0].normsq_pointwise.reshape(psi_list[0].N[0], psi_list[0].N[1]))

    def updatefig(frame, *args):
        im.set_array(psi_list[frame].normsq_pointwise.reshape(psi_list[frame].N[0], psi_list[frame].N[1]))
        return im,
    ani = animation.FuncAnimation(fig, updatefig, frames = len(psi_list), interval=50, blit=True)
    plt.show()
    return ani