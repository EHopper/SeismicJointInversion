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

def setup_figure_layout(location, t_LAB):
    f = plt.figure()
    f.set_size_inches((15,7))
    ax_c = f.add_axes([0.6, 0.6, 0.35, 0.3])
    ax_dc = f.add_axes([0.6, 0.375, 0.35, 0.15])
    ax_rf = f.add_axes([0.6, 0.1, 0.2, 0.2])
    ax_m150 = f.add_axes([0.35, 0.3, 0.2, 0.6])
    ax_mDeep = f.add_axes([0.35, 0.1, 0.2, 0.12])
    ax_map = f.add_axes([0.84, 0.1, 0.1, 0.2])

    lat, lon = location
    ax_c.set_title(
        '{:.1f}N, {:.1f}W:  {:.0f} km LAB'.format(lat, -lon, t_LAB)
    )

    return f, ax_c, ax_dc, ax_rf, ax_m150, ax_mDeep, ax_map

def plot_area_map(location, ax_map):
    vs_SL14 = pd.read_csv('data/earth_models/CP_SL14.csv',
                           header=None).values
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values

    im = ax_map.contourf(np.arange(-119, -100.9, 0.2),
        np.arange(30, 45.1, 0.2), vs_SL14, levels=20,
        cmap=plt.cm.RdBu, vmin=4, vmax=4.7)
    ax_map.plot(location[1], location[0], 'k*')
    ax_map.plot(cp_outline[:, 1], cp_outline[:, 0], 'k:')

def plot_SL14_profile(location, ax):
    # To save spave, have saved the lat, lon, depth ranges of the Schmandt & Lin
    # model just as three lines in a csv
    vs_SLall = pd.read_csv('data/earth_models/SchmandtLinVs_only.csv',
                            header=None).values.flatten()
    vs_lats = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=0,
                          nrows=1, header=None).values.flatten()
    vs_lons = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=1,
                          nrows=1, header=None).values.flatten()
    vs_deps = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=2,
                          nrows=1, header=None).values.flatten()
    vs_SLall = vs_SLall.reshape((vs_lats.size, vs_lons.size, vs_deps.size))

    lat, lon = location
    vs_ilon = np.argmin(abs(vs_lons - lon))
    vs_ilat = np.argmin(abs(vs_lats - lat))
    ax.plot(vs_SLall[vs_ilat, vs_ilon, :], vs_deps,
                 'k-', linewidth=3, label='SL14')

def plot_damping_params(model_id, f):
    save_name = 'output/{0}/{0}'.format(model_id)
    damp_s = pd.read_csv(save_name + 'damp_s.csv')
    damp_t = pd.read_csv(save_name + 'damp_t.csv')
    n = 0
    for label in ['roughness', 'to_m0', 'to_0_grad']:
        ax_d = f.add_axes([0.05 + 0.1 * n, 0.3, 0.05, 0.6])
        ax_d.plot(damp_s[label], damp_s.Depth, 'ko-', markersize=3)
        ax_d.plot(damp_t[label], damp_t.Depth, 'ro', markersize=2)
        ax_d.set(title=label)
        ax_d.set_ylim([150, 0])
        ax_d.xaxis.tick_top()
        plt.rcParams.update({'axes.titlesize': 'xx-small',
                             'axes.labelsize': 'xx-small',
                             'xtick.labelsize': 'xx-small',
                             'ytick.labelsize': 'xx-small'})
        n += 1

def plot_G(G, model, periods):

    olabs = ['{:.0f} s'.format(p) for p in periods]
    olabs += ['Moho tt', 'LAB tt', 'Moho dVs', 'LAB dVs']
    mlabs = ['Vs at {:.0f} km'.format(d)
             for d in np.cumsum(model.thickness[model.d_inds])]
    mlabs += ['Moho depth', 'LAB depth']

    # Plot all G as an image
    f = plt.figure(figsize=(10, 5))
    a = f.add_axes([0, 0, 1, 1])
    a.text(0.475, 0.95, 'G matrix image')
    ax1 = f.add_axes([0.1, 0.2, 0.65, 0.7])
    im = ax1.imshow(G, vmin=-0.5, vmax=0.5)
    ax1.set(xlabel='Model', ylabel='Observation')
    ax1.set(yticks=range(G.shape[0]), yticklabels=olabs, xticks=range(G.shape[1]))
    ax1.set_xticklabels(mlabs, rotation=90)
    cb = plt.colorbar(im, f.add_axes([0.76, 0.25, 0.01, 0.6]))

    ax2 = f.add_axes([0.875, 0.2, 0.04, 0.7])
    im = ax2.imshow(G[:, -2:], vmin=-0.0003, vmax=0.0003)
    ax2.set(xlabel='Model', ylabel='Observation')
    ax2.set(yticks=range(G.shape[0]), yticklabels=olabs, xticks=range(2))
    ax2.set_xticklabels(mlabs[-2:], rotation=90)
    cb = plt.colorbar(im, f.add_axes([0.93, 0.25, 0.01, 0.6]))

    f = plt.figure(figsize=(10, 5))
    a = f.add_axes([0, 0, 1, 1])
    a.text(0.45, 0.95, 'abs(G) as stacked bar')
    ax1 = f.add_axes([0.1, 0.2, 0.4, 0.7])
    pp = np.hstack((periods, periods[-1] + 10 + [0, 5, 10, 15]))
    for i in range(G.shape[1]):
        if i < 18 or i > G.shape[1] - 3:
            plt.bar(pp, abs(G[:, i]), bottom=np.sum(abs(G[:, :i]), 1),
                    width=3, label=mlabs[i])
        else:
            plt.bar(pp, abs(G[:, i]), bottom=np.sum(abs(G[:, :i]), 1), width=3)
    plt.legend(fontsize='xx-small')
    ax1.set(ylabel='abs(G)', xlabel='Observation', xticks=pp)
    ax1.set_xlim(pp[0] - 2, pp[-1] + 2)
    ax1.set_xticklabels(olabs, rotation=90)
    ax1.set_title('Moho at {:.0f} km; LAB at {:.0f} km'.format(
        np.sum(model.thickness[:model.boundary_inds[0] + 1]),
        np.sum(model.thickness[:model.boundary_inds[1] + 1]),
    ))

    ax2 = f.add_axes([0.55, 0.2, 0.4, 0.7])
    d = np.cumsum(model.thickness[:-1])
    d = np.hstack((d, d[-1] + [6, 12]))
    for i in range(G.shape[0]):
        plt.bar(d, abs(G[i, :]), bottom=np.sum(abs(G[:i, :]), 0),
                width=5, label=olabs[i])
    plt.legend(fontsize='xx-small')
    ax2.set(ylabel='abs(G)', xlabel='Model', xticks=d)
    ax2.set_xticklabels(mlabs, rotation=90)
