import matplotlib.pyplot as plt
import numpy as np


def start_fig(size):
    f = plt.figure()
    f.set_size_inches(size)
    return f.add_subplot(1, 1, 1)

def plot_model(model, label, ax):
    depths = np.cumsum(model.thickness)
    for bl in model.boundary_inds:
        ax.axhline(depths[bl], linestyle=':', color='#e0e0e0')
    line, = ax.plot(model.vsv, depths, 'o-', markersize=3)
    line.set_label(label)
    ax.set_ylim([depths[-1], 0])
    ax.set(xlabel='Vsv (km/s)', ylabel='Depth (km)')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.legend()

def plot_ph_vel(periods, c, label, ax):
    line, = ax.plot(periods, c, 'o-', markersize=4)
    line.set_label(label)
    ax.set(xlabel='Vsv (km/s)', ylabel='Depth (km)')
    ax.legend()
