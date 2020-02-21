import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sklclust
import scipy.interpolate
import sklearn.linear_model


from util import define_models
from util import mineos
from util import inversion
from util import partial_derivatives
from util import weights
from util import constraints


def make_fig():
    plt.figure(figsize=(10,8))
    return plt.subplot(1, 1, 1)

def plot_model(model, label, ax, depth_range=(), iflegend=True, col=False):
    depth = np.cumsum(model.thickness)
    for ib in model.boundary_inds:
        ax.axhline(depth[ib], linestyle=':', color='#e0e0e0')

    if col:
        ax.plot(model.vsv, depth, '-o', markersize=2, label=label, color=col)
    else:
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
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))

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
    ax1 = f.add_axes([0.1, 0.2, 0.8, 0.7])
    # pp = np.hstack((periods, periods[-1] + 10 + [0, 5, 10, 15]))
    pp = periods
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

    f = plt.figure(figsize=(10, 5))
    ax2 = f.add_axes([0.1, 0.2, 0.8, 0.7])
    d = np.cumsum(model.thickness[:-1])
    d = np.hstack((d, d[-1] + [6, 12]))
    for i in range(G.shape[0]):
        plt.bar(d, abs(G[i, :]), bottom=np.sum(abs(G[:i, :]), 0),
                width=5, label=olabs[i])
    plt.legend(fontsize='xx-small')
    ax2.set(ylabel='abs(G)', xlabel='Model', xticks=d)
    ax2.set_xticklabels(mlabs, rotation=90)


def plot_results_map(depth, t_LAB=5., vmi=0, vma=0, ifsave=False):
    f = plt.figure(figsize=(6, 6))
    ax_map = f.add_axes([0.1, 0.2, 0.8, 0.7])

    z = np.arange(0, 300, 5)
    lats = np.arange(33, 43)
    lons = np.arange(-115, -105)
    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB)
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))
    idep = np.argmin(abs(z - depth))

    if vmi == 0:
        if z[idep] < 40:
            vmi = 3.
            vma = 4.
        else:
            vmi = 3.75
            vma = 4.5
    im = ax_map.imshow(vs[:, :, idep], cmap=plt.cm.RdBu, aspect=1.4,
                       vmin=vmi, vmax=vma)
    ax_map.set(xticks=range(len(lons)), xticklabels = abs(lons),
               yticks=range(len(lats)), yticklabels = lats,
               xlabel='Longitude (W)', ylabel='Latitude (N)')
    c = plt.colorbar(im, cax=f.add_axes([0.15, 0.1, 0.7, 0.02]),
                      orientation='horizontal')
    c.ax.set_xlabel('Vsv (km/s)')
    ax_map.set_title('Velocity slice at {} km'.format(z[idep]))
    ax_map.set_ylim((0, len(lats) - 1))
    ax_map.plot(cp_outline[:, 1] - lons[0],
                cp_outline[:, 0] - lats[0], 'k:')

    if ifsave:
        plt.savefig(
            '/media/sf_VM_Shared/rftests/map_{}kmLAB_{}kmDepth.png'.format(
            t_LAB, z[idep]
            )
        )
    return ax_map

def plot_BLs_map(t_LAB=5.):
    f = plt.figure(figsize=(14, 6))
    ax_bl1 = f.add_axes([0.1, 0.2, 0.4, 0.7])
    ax_bl2 = f.add_axes([0.55, 0.2, 0.4, 0.7])

    z = np.arange(0, 300, 5)
    lats = np.arange(33, 43)
    lons = np.arange(-115, -105)
    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB)
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))

    im = ax_bl1.imshow(bls[:, :, 0], cmap=plt.cm.RdBu, aspect=1.4)
    ax_bl1.set(xticks=range(len(lons)), xticklabels = abs(lons),
               yticks=range(len(lats)), yticklabels = lats,
               xlabel='Longitude (W)', ylabel='Latitude (N)')
    c1 = plt.colorbar(im, cax=f.add_axes([0.15, 0.1, 0.3, 0.02]),
                      orientation='horizontal')
    c1.ax.set_xlabel('Moho Depth (km)')
    ax_bl1.set_title('Moho Depth')
    ax_bl1.set_ylim((0, len(lats) - 1))
    ax_bl1.plot(cp_outline[:, 1] - lons[0],
                cp_outline[:, 0] - lats[0], 'k:')

    im2 = ax_bl2.imshow(bls[:, :, 1], cmap=plt.cm.RdBu, aspect=1.4)
    ax_bl2.set(xticks=range(len(lons)), xticklabels = abs(lons),
               yticks=range(len(lats)), yticklabels = lats,
               xlabel='Longitude (W)', ylabel='Latitude (N)')
    c2 = plt.colorbar(im2, cax=f.add_axes([0.6, 0.1, 0.3, 0.02]),
                      orientation='horizontal')
    c2.ax.set_xlabel('LAB Depth (km)')
    ax_bl2.set_title('LAB Depth')
    ax_bl2.set_ylim((0, len(lats) - 1))
    ax_bl2.plot(cp_outline[:, 1] - lons[0],
                cp_outline[:, 0] - lats[0], 'k:')

    plt.savefig(
        '/media/sf_VM_Shared/rftests/map_BLdepths.png'
    )

def plot_BLs_dVs_map(t_LAB=5.):
    f = plt.figure(figsize=(14, 6))
    ax_bl1 = f.add_axes([0.1, 0.2, 0.4, 0.7])
    ax_bl2 = f.add_axes([0.55, 0.2, 0.4, 0.7])

    z = np.arange(0, 300, 5)
    lats = np.arange(33, 43)
    lons = np.arange(-115, -105)
    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB)
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))

    im = ax_bl1.imshow(bls[:, :, 2] * 100, cmap=plt.cm.RdBu_r, aspect=1.4,
                       vmin=0, vmax=np.quantile(bls[:, :, 2], 0.95) * 100)
    ax_bl1.set(xticks=range(len(lons)), xticklabels = abs(lons),
               yticks=range(len(lats)), yticklabels = lats,
               xlabel='Longitude (W)', ylabel='Latitude (N)')
    c1 = plt.colorbar(im, cax=f.add_axes([0.15, 0.1, 0.3, 0.02]),
                      orientation='horizontal')
    c1.ax.set_xlabel('Moho dVs (%)')
    ax_bl1.set_title('Moho dVs')
    ax_bl1.set_ylim((0, len(lats) - 1))
    ax_bl1.plot(cp_outline[:, 1] - lons[0],
                cp_outline[:, 0] - lats[0], 'k:')

    im2 = ax_bl2.imshow(bls[:, :, 3] * 100, cmap=plt.cm.RdBu, aspect=1.4,
                        vmin=np.quantile(bls[:, :, 3], 0.05) * 100, vmax=0)
    ax_bl2.set(xticks=range(len(lons)), xticklabels = abs(lons),
               yticks=range(len(lats)), yticklabels = lats,
               xlabel='Longitude (W)', ylabel='Latitude (N)')
    c2 = plt.colorbar(im2, cax=f.add_axes([0.6, 0.1, 0.3, 0.02]),
                      orientation='horizontal')
    c2.ax.set_xlabel('LAB dVs (%)')
    ax_bl2.set_title('LAB dVs')
    ax_bl2.set_ylim((0, len(lats) - 1))
    ax_bl2.plot(cp_outline[:, 1] - lons[0],
                cp_outline[:, 0] - lats[0], 'k:')

    plt.savefig(
        '/media/sf_VM_Shared/rftests/map_BLdVs.png'
    )


def plot_map(vs, lats, lons, z, depth, label='', vmi=0, vma=0):
    f = plt.figure(figsize=(6, 6))
    ax_map = f.add_axes([0.1, 0.2, 0.8, 0.7])
    idep = np.argmin(abs(z - depth))
    if vmi == 0:
        if z[idep] < 40:
            vmi = 3.
            vma = 4.
        else:
            # vmi = 3.75
            # vma = 4.5
            vmi = np.mean(vs[:, :, idep] - 0.2)
            vma = np.mean(vs[:, :, idep] + 0.2)

    im = plt.imshow(vs[:, :, idep], cmap=plt.cm.RdBu, aspect=1.4,
                    vmin=vmi, vmax=vma)
    ax_map.set(xticks=range(len(lons)), xticklabels = abs(lons),
               yticks=range(len(lats)), yticklabels = lats,
               xlabel='Longitude (W)', ylabel='Latitude (N)')

    c = plt.colorbar(im, cax=f.add_axes([0.15, 0.075, 0.7, 0.02]),
                      orientation='horizontal')
    c.ax.set_xlabel('Vsv (km/s)')
    ax_map.set_title('Velocity slice at {} km'.format(z[idep]))
    ax_map.set_ylim((-0.5, len(lats) - 0.5))
    ax_map.set_xlim((-0.5, len(lons) - 0.5))

    _plot_States(ax_map, lats, lons)
    _plot_Colorado_Plateau(ax_map, lats, lons)


    plt.savefig(
        '/media/sf_VM_Shared/rftests/map_{}_{}kmDepth.png'.format(
        label, z[idep]
        )
    )


def _plot_Colorado_Plateau(ax_map, lats, lons):
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))
    ax_map.plot(convert_latlons(cp_outline[:, 1], lons),
                convert_latlons(cp_outline[:, 0], lats), 'k:')

def _plot_States(ax_map, lats, lons):
    states = ['Nevada', 'New_Mexico', 'Arizona', 'Utah', 'Oklahoma']
    # states = ['Washington', 'Oregon', 'California', 'Nevada',
    #           'Idaho', 'Utah', 'Arizona',
    #           'Montana', 'Wyoming', 'Colorado', 'New_Mexico',
    #           'North_Dakota', 'South_Dakota', 'Nebraska', 'Kansas', 'Oklahoma', 'Texas',
    #           'Minnesota', 'Iowa', 'Missouri', 'Arkansas', 'Louisiana',
    #           'Michigan_UP', 'Wisconsin', 'Illinois','Mississippi',
    #           'Michigan_LP', 'Indiana', 'Ohio', 'Kentucky', 'Tennessee', 'Alabama']
    for state in states:
        border = pd.read_csv('data/earth_models/US_States/' + state + '.csv',
                             header=None).values
        ax_map.plot(convert_latlons(border[:, 1], lons),
                    convert_latlons(border[:, 0], lats), '-',
                    linewidth=0.5, color='#d4d4d4')

def plot_map_2D(vals, lats, lons, vmi=0, vma=0):
    f = plt.figure(figsize=(6, 6))
    ax_map = f.add_axes([0.1, 0.2, 0.8, 0.7])
    im = _plot_map(vals, lats, lons, plt.cm.RdBu, vmi, vma, ax_map)
    c = plt.colorbar(im, cax=f.add_axes([0.15, 0.075, 0.7, 0.02]),
                      orientation='horizontal')

    return ax_map

def _plot_map(vals, lats, lons, cbar, vmi, vma, ax_map):
    im = plt.imshow(vals[:, :], cmap=cbar, aspect=1.4, vmin=vmi, vmax=vma)
    lon_inc = max(1, len(lons) // 10)
    lat_inc = max(1, len(lats) // 10)
    ax_map.set(xticks=range(0, len(lons), lon_inc), xticklabels = abs(lons[::lon_inc]),
               yticks=range(0,len(lats), lat_inc), yticklabels = lats[::lat_inc],
               xlabel='Longitude (W)', ylabel='Latitude (N)')

    ax_map.set_ylim((0, len(lats) - 1))

    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))
    ax_map.plot(convert_latlons(cp_outline[:, 1], lons),
                convert_latlons(cp_outline[:, 0], lats), 'k:')

    return im


def plot_map_2D_r(vals, lats, lons, vmi=0, vma=0):
    f = plt.figure(figsize=(6, 6))
    ax_map = f.add_axes([0.1, 0.2, 0.8, 0.7])
    im = _plot_map(vals, lats, lons, plt.cm.RdBu_r, vmi, vma, ax_map)
    c = plt.colorbar(im, cax=f.add_axes([0.15, 0.075, 0.7, 0.02]),
                      orientation='horizontal')

    return ax_map

def convert_latlons(map_points, loc_vector):
    return (map_points - loc_vector[0]) / np.diff(loc_vector[:2])

def plot_phase_vels_margins():
    # locs = [(39,-113), (38,-113), (37, -114), (36, -113), (35, -113),
    #         (35, -112), (34, -111), (34, -110), (34, -109), (35, -108),
    #         (35, -107), (36, -107), (37, -107), (38, -108), (39, -107)]
    locs = []
    for lat in range(34, 41):
        for lon in [-114, -113, -108, -107]:
            locs += [(lat, lon)]

    phv_all = constraints._load_observed_sw_constraints()

    i = 0
    for loc in locs:
        a = constraints._extract_phase_vels(loc, (1, phv_all))
        if i == 0:
            phv = np.zeros((len(a), len(locs)))
        phv[:, i] = a.ph_vel.values
        i += 1
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))

    f = plt.figure(figsize=(15, 6))
    ax_m = f.add_axes([0.35, 0.1, 0.3, 0.8])
    ax_m.plot(cp_outline[:, 1], cp_outline[:, 0], 'k-')

    yl = ax_m.get_ylim()
    yl = (yl[0], max(lat + 1.5, yl[1]))
    ax_m.set_ylim(yl)

    i = 0
    for loc in locs:
        p = ax_m.plot(loc[1], loc[0], '^')
        if loc[1] < -110:
            st_x = 0.02
        else:
            st_x = 0.67
        if i % 2:
            st_x += 0.125
        st_y = float((loc[0] - yl[0])/(np.diff(yl))) + 0.01

        aa = f.add_axes([st_x, st_y, 0.18, 0.15])
        _plot_c_one_colour(phv, a.period, i, p, aa)
        aa.set_axis_off()

        i += 1

    return

def _plot_c_one_colour(phv, periods, i, p, ax):

    for ip in range(phv.shape[1]):
        ax.plot(periods, phv[:, ip], color='#E4E4E4')
    ax.plot(periods, phv[:, i], '-o', color=p[0].get_color(), markersize=2)
    ax.plot(periods[10], phv[10, i], '*', color=p[0].get_color())

def plot_all_v_models_on_map():
    z = np.arange(0, 350, 0.5)
    lats = np.arange(34, 41)
    lons = np.arange(-114, -106)
    t_LAB = 5
    vs_older, bls, bis = define_models.load_all_models(z, lats, lons, 5, '_highQ')
    vs_old, bls, bis = define_models.load_all_models(z, lats, lons, 5, '_damped')
    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB, '_smoothed')
    #vs = constraints.interpolate_lit_model('P15', z, lats, lons)
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))
    f = plt.figure(figsize=(10, 12))
    a = f.add_axes([0.1, 0.1, 0.8, 0.8], aspect=1.2)
    a.plot(cp_outline[:, 1], cp_outline[:, 0], 'k-')
    a.set(xlim=[lons[0]-0.5, lons[-1]+0.5], ylim=[lats[0]-0.5, lats[-1]+0.5],
          xlabel='Longitude (W)', ylabel='Latitude (N)')

    vlim = (3.2, 4.8)
    zlim = (0, 350)

    vz = vs[:, :, (zlim[0] <= z) & (z < zlim[1])]
    #iiz =  vz.shape[2] #np.argmax(z > 120) #
    vsold = vs_old[:, :, (zlim[0] <= z) & (z < zlim[1])]
    vsolder = vs_older[:, :, (zlim[0] <= z) & (z < zlim[1])]
    # maxv = (np.quantile(np.quantile(vz, 0.75, 0), 0.75, 0) - vlim[0]) / np.diff(vlim) - 0.5
    # minv = (np.quantile(np.quantile(vz, 0.25, 0), 0.25, 0) - vlim[0]) / np.diff(vlim) - 0.5

    lit_models = ('Y14', 'P14','P15','C15','S15','SR16','F18')
    vs_lit = np.zeros((vs.shape[0], vs.shape[1], vs.shape[2], len(lit_models)))

    i = 0
    for mod in lit_models:
        vs_lit[:, :, :, i] = constraints.interpolate_lit_model(mod, z, lats, lons)
        i += 1
    i = 0
    for mod in lit_models:
        ds = constraints.load_literature_vel_model(mod)
        z_a = ds.depth.values
        if zlim[0] < z_a[0]:
            iz = np.argmax(z_a[0] < z)
            vs_lit[:, :, :iz, i] = np.median(vs_lit[:, :, :iz, :], 3)
        if z_a[-1] < zlim[1]:
            iz = np.argmax(z_a[-1] < z)
            vs_lit[:, :, iz:, i] = np.median(vs_lit[:, :, iz:, :], 3)
        i += 1
    vs_lit_z = vs_lit[:, :, (zlim[0] <= z) & (z < zlim[1]), :]
    maxv = (np.max(vs_lit_z, 3) - vlim[0]) / np.diff(vlim) - 0.5
    minv = (np.min(vs_lit_z, 3) - vlim[0]) / np.diff(vlim) - 0.5
    maxv[maxv < -0.5] = -0.5
    minv[minv < -0.5] = -0.5


    for ila in range(len(lats)):
        for ilo in range(len(lons)):
            pz = np.linspace(lats[ila] + 0.5, lats[ila] - 0.5, vz.shape[2])
            # a.fill(np.append(maxv + lons[ilo], minv[::-1] + lons[ilo]),
            #        np.append(pz, pz[::-1]), color='#8B8B8B')
            a.fill(np.append(maxv[ila, ilo, :] + lons[ilo],
                             minv[ila, ilo, ::-1] + lons[ilo]),
                   np.append(pz, pz[::-1]), color='#8B8B8B')

            v = (vsolder[ila, ilo, :] - vlim[0]) / np.diff(vlim) - 0.5
            v[v < -0.5] = -0.5
            v += lons[ilo]
            a.plot(v, pz, 'c-', linewidth=1.5)

            v = (vsold[ila, ilo, :] - vlim[0]) / np.diff(vlim) - 0.5
            v[v < -0.5] = -0.5
            v += lons[ilo]
            a.plot(v, pz, 'b-', linewidth=1)

            v = (vz[ila, ilo, :] - vlim[0]) / np.diff(vlim) - 0.5
            ilab = np.argmax(z > bls[ila, ilo, 1] + 5)
            # im = np.argmax(np.diff(v[ilab:-200]) <= 0) + ilab#np.argmin(v[ilab:]) + ilab
            v[v < -0.5] = -0.5
            v += lons[ilo]
            a.plot([lons[ilo]] * len(pz), pz, 'k-', linewidth=0.5)
            a.plot(lons[ilo] + [-0.5, 0.5], [lats[ila]] * 2, 'k-', linewidth=0.5)
            a.plot(v, pz, 'r-', linewidth=1)
            # a.plot(v[:iiz], pz[:iiz], 'r-', linewidth=1)
            # if im > ilab:
            #     a.plot(v, pz, 'b-', linewidth=1)




            a.plot(lons[ilo] + np.array([-0.5, -0.5, 0.5, 0.5, -0.5]),
                   lats[ila] + np.array([-0.5, 0.5, 0.5, -0.5, -0.5]),
                   color='#F7F7F7', linewidth=2)

def plot_v_model_comparison_on_map():
    z = np.arange(0, 300, 0.5)
    lats = np.arange(33, 43)
    lons = np.arange(-115, -105)
    t_LAB = 5.

    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB)
    lit_models = ('Y14','P14','P15','C15','S15','SR16','F18')
    vs_lit = np.zeros((vs.shape[0], vs.shape[1], vs.shape[2], len(lit_models)))

    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))
    f = plt.figure(figsize=(8, 12))
    a = f.add_axes([0.1, 0.1, 0.7, 0.8], aspect=1.2)
    a.plot(cp_outline[:, 1], cp_outline[:, 0], '-', color='#BDBDBD')
    a.set(xlim=[lons[0]-0.5, lons[-1]+0.5], ylim=[lats[0]-0.5, lats[-1]+0.5],
          xlabel='Longitude (W)', ylabel='Latitude (N)')

    a_leg = f.add_axes([0.85, 0.3, 0.1, 0.4])
    i = 0
    cols = []
    for mod in lit_models:
        vs_lit[:, :, :, i] = constraints.interpolate_lit_model(mod, z, lats, lons)
        p = a_leg.plot([0, 1], [i] * 2)
        cols += [p[0].get_color()]
        a_leg.text(1.5, i, mod, va='center')
        i += 1
    a_leg.set_axis_off()
    a_leg.set_xlim([0, 3])

    vlim = (3.5, 4.7)
    zlim = (0, 150)

    vz = vs[:, :, (zlim[0] <= z) & (z < zlim[1])]
    vs_lit_z = vs_lit[:, :, (zlim[0] <= z) & (z < zlim[1]), :]

    for ila in range(len(lats)):
        for ilo in range(len(lons)):
            pz = np.linspace(lats[ila] + 0.5, lats[ila] - 0.5, vz.shape[2])
            i = 0
            for mod in lit_models:
                a.plot(_scale_vel_profile(vs_lit_z[ila, ilo, :, i], vlim, lons[ilo]),
                       pz, linewidth=2, color=cols[i])
                i += 1
            a.plot(_scale_vel_profile(vz[ila, ilo, :], vlim, lons[ilo]),
                   pz, 'k-', linewidth=2)

            a.plot(lons[ilo] + np.array([-0.5, -0.5, 0.5, 0.5, -0.5]),
                   lats[ila] + np.array([-0.5, 0.5, 0.5, -0.5, -0.5]),
                   color='#F7F7F7', linewidth=2)

def _scale_vel_profile(v, vlim, lon):
    v_sv = (v - vlim[0]) / np.diff(vlim) - 0.5
    v_sv[v_sv < -0.5] = -0.5
    return v_sv + lon

def plot_cross_section(v, v_grid, coords):
    z, lats, lons = v_grid
    fn = scipy.interpolate.RegularGridInterpolator((lats, lons, z), v)

    lo_q = np.linspace(coords[0][1], coords[1][1], 50)
    la_q = np.linspace(coords[0][0], coords[1][0], 50)
    v_q = np.zeros((lo_q.size, z.size))

    for id in range(len(z)):
        v_q[:, id] = fn([[la_q[i], lo_q[i], z[id]] for i in range(len(lo_q))])

    # Calculate LAB depth
    lab_d = np.zeros_like(la_q)
    lab_amp = np.zeros_like(la_q)
    for i in range(len(la_q)):
        rfs = constraints._extract_rf_constraints(
            (la_q[i], lo_q[i]), '', (['LAB'], [5]), 1.75
        )

        lab_d[i] = z[
            np.argmax(np.cumsum(
                    np.diff(z)
                    / np.mean(np.vstack((v_q[i, :-1], v_q[i, 1:])), 0))
                > rfs.tt[0])
        ]
        lab_amp[i] = abs(rfs.dv[0])

    lab_amp /= min(lab_amp) / 2

    f = plt.figure(figsize=(14, 4))
    a = f.add_axes([0.05, 0.2, 0.5, 0.8], aspect=1/50)
    if np.abs(np.diff(la_q[[0, -1]])) < np.abs(np.diff(lo_q[[0, -1]])):
        x_pts = lo_q
        x_lab = 'Longitude (E)'
    else:
        x_pts = la_q
        x_lab = 'Latitude (N)'

    vmi = 3.75
    vma = 4.65
    v_q[v_q < vmi] = vmi
    v_q[v_q > vma] = vma
    im = a.contourf(x_pts, z, v_q.T, 20, cmap=plt.cm.RdBu, vmin=vmi, vmax=vma)
    #a.plot(x_pts, lab_d, 'k+')
    for i in range(len(la_q)):
        a.plot(x_pts[i], lab_d[i], 'k+', markersize=lab_amp[i])
    a.set(ylabel='Depth (km)', xlabel=x_lab, ylim=[150, 0])
    c = plt.colorbar(im, cax=f.add_axes([0.1, 0.15, 0.4, 0.02]),
                      orientation='horizontal')
    c.ax.set_xlabel('Vsv (km/s)')

    a2 = f.add_axes([0.65, 0.15, 0.3, 0.8], aspect=1.4)
    id = np.argmax(75 <= z)
    im2 = _plot_map(v[:, :, id], lats, lons, plt.cm.RdBu, vmi, vma, a2)
    c2 = plt.colorbar(im2, cax=f.add_axes([0.95, 0.2, 0.005, 0.7]),
                      orientation='vertical')
    c2.ax.set_ylabel('Vsv at 75 km depth (km/s)')
    a2.plot(convert_latlons(lo_q, lons), convert_latlons(la_q, lats), 'k-')


def plot_correlation(x, y, ax, lab):

    lm = sklearn.linear_model.LinearRegression().fit(x.ravel()[:, np.newaxis],
                                                     y.ravel()[:, np.newaxis])
    lab = lab + '   {:.2f} x + {:.2f}; R2 = {:.2f}'.format(
        lm.coef_[0, 0], lm.intercept_[0],
        lm.score(x.ravel()[:, np.newaxis], y.ravel()[:, np.newaxis])
    )
    xl = np.array([[x.min(), x.max()]]).T
    ax.plot(xl, lm.predict(xl), '--', color='#EBEBEB')
    ax.plot(x.ravel(), y.ravel(), '.', label=lab)
