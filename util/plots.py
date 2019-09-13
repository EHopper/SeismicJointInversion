import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import define_models
from util import mineos
from util import inversion
from util import partial_derivatives
from util import weights
from util import constraints

def plot_model(model, label, ax):
    depth = np.cumsum(model.thickness)
    line, = ax.plot(model.vsv, depth, '-o', markersize=2, label=label)
    for ib in model.boundary_inds:
        ax.axhline(depth[ib], linestyle=':', color='#e0e0e0')
    ax.set_ylim([depth[-1], 0])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set(xlabel='Vsv (km/s)', ylabel='Depth (km)')
    ax.legend()

def plot_ph_vel(periods, c, label, ax):
    line = ax.plot(periods, c, '-o', markersize=3, label=label)
    ax.set(xlabel='Period (s)', ylabel='Phase Velocity (km/s)')
    ax.legend()
