import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import define_models
from util import mineos
from util import inversion
from util import partial_derivatives
from util import weights
from util import constraints

def make_fig():
    plt.figure(figsize=(10,8))
    return plt.subplot(1, 1, 1)

def plot_model(model, label, ax, depth_range=(), iflegend=True):
    depth = np.cumsum(model.thickness)
    for ib in model.boundary_inds:
        ax.axhline(depth[ib], linestyle=':', color='#e0e0e0')
    line, = ax.plot(model.vsv, depth, '-o', markersize=2, label=label)
    if depth_range:
        ax.set_ylim([depth_range[1], depth_range[0]])
    else:
        ax.set_ylim([depth[-1], 0])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set(xlabel='Vsv (km/s)', ylabel='Depth (km)')
    if iflegend:
        ax.legend()

def plot_model_simple(model, label, ax, depth_range=(), iflegend=True):
    depth = np.cumsum(model.thickness)
    line, = ax.plot(model.vsv, depth, linestyle='-',
                    alpha=0.5, label=label)
    ax.plot(model.vsv[model.boundary_inds], depth[model.boundary_inds],
            'o', markersize=2,  alpha=0.5, color=line.get_color())
    if depth_range:
        ax.set_ylim([depth_range[1], depth_range[0]])
    else:
        ax.set_ylim([depth[-1], 0])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set(xlabel='Vsv (km/s)', ylabel='Depth (km)')
    if iflegend:
        ax.legend()

def plot_ph_vel(periods, c, label, ax):
    line = ax.plot(periods, c, '-o', markersize=3, label=label)
    ax.set(xlabel='Period (s)', ylabel='Phase Velocity (km/s)')
    ax.legend()

def plot_ph_vel_data_std(periods, obs_c, std_c, label, ax):
    ax.plot(periods, obs_c, 'k-', linewidth=3, label=label)
    ax.errorbar(periods, obs_c, yerr=std_c, linestyle='None', ecolor='k')
    ax.set(xlabel='Period (s)', ylabel='Phase Velocity (km/s)')
    ax.legend()

def plot_ph_vel_simple(periods, c, ax):
    ax.plot(periods, c, '-', alpha=0.5) #color='#706F6F',
    ax.set(xlabel='Period (s)', ylabel='Phase Velocity (km/s)')

def plot_dc(periods, dc, ax):
    line = ax.plot(periods, dc, '-o', markersize=3)
    ax.set(xlabel='Period (s)', ylabel='c misfit (km/s)')

def plot_rf_data_std(rf_data, std_rf_data, label, ax):
    tt = rf_data[:len(rf_data) // 2].flatten()
    dv = rf_data[len(rf_data) // 2:].flatten() * 100
    std_tt = std_rf_data[:len(rf_data) // 2].flatten()
    std_dv = std_rf_data[len(rf_data) // 2:].flatten() * 100

    ax.errorbar(tt, dv, yerr=std_dv, xerr=std_tt, linestyle='None', ecolor='k')
    ax.plot(tt, dv, 'k.', markersize=5, label=label)
    ax.set(xlabel='RF Travel Time (s)', ylabel='Estimated dVs from RF (%)')
    ax.axhline(0, linestyle=':', color='#e0e0e0')
    #ax.legend()

def plot_rf_data(rf_data, label, ax):
    tt = rf_data[:len(rf_data) // 2].flatten()
    dv = rf_data[len(rf_data) // 2:].flatten() * 100
    ax.plot(tt, dv, '.', markersize=3, label=label)
    ax.set(xlabel='RF Travel Time (s)', ylabel='Estimated dVs from RF (%)')
    #ax.legend()

def make_plot_symmetric_in_y_around_zero(ax):
    yl = max(abs(np.array(ax.get_ylim())))
    ax.set_ylim([-yl, yl])

def plot_kernels(kernels, ax, field='vsv'):
    periods = kernels.period.unique()
    for p in periods:
        k = kernels[kernels.period == p]
        ax.plot(k[field], k.z, '-o', markersize=2, label=str(p) + ' s')

    ax.legend()
    ax.set_ylim([200, 0])
    ax.set(xlabel='Kernels for ' + field, ylabel='Depth (km)')
