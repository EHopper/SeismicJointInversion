""" Plotting cross sections with RF data

Pulling together things for VBR paper
"""

import scipy.interpolate
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt




def plot_all(coords):
    z = np.arange(0, 300, 2)
    lats = np.arange(30, 51, 0.25)
    lons = np.arange(-125, -90, 0.25)
    vs = interpolate_lit_model('SR16', z, lats, lons)
    q = interpolate_lit_model('DE08', z, lats, lons)

    fn_v = scipy.interpolate.RegularGridInterpolator((lats, lons, z), vs)
    fn_q = scipy.interpolate.RegularGridInterpolator((lats, lons, z), q)

    lon_xsect = np.linspace(coords[0][1], coords[1][1], 50)
    lat_xsect = np.linspace(coords[0][0], coords[1][0], 50)
    vs_xsect = np.zeros((50, z.size))
    q_xsect = np.zeros((50, z.size))

    for id in range(len(z)):
        vs_xsect[:, id] = fn_v([[lat, lon, z[id]]
                             for lat, lon in zip(lat_xsect, lon_xsect)])
        q_xsect[:, id] = fn_q([[lat, lon, z[id]]
                             for lat, lon in zip(lat_xsect, lon_xsect)])


    f = plt.figure(figsize=(8, 10))
    vmi = 4.0
    vma = 4.7
    ax_xsect = f.add_axes([0.075, 0.4, 0.55, 0.2], aspect=1/30)
    im, x_pts = _plot_xsect(vs_xsect, lat_xsect, lon_xsect, z,
                     plt.cm.RdBu, vmi, vma, [150, -50], ax_xsect)
    _plot_LAB_data(lat_xsect, lon_xsect, vs_xsect, x_pts, z, ax_xsect)
    ax_xsect.plot(x_pts, -_get_topo(lat_xsect, lon_xsect) * 10, 'k-')
    c = plt.colorbar(im, cax=f.add_axes([0.1, 0.36, 0.5, 0.01]), orientation='horizontal')
    c.ax.set_xlabel('Vsv (km/s)')

    ax_xsectQ = f.add_axes([0.075, 0.1, 0.55, 0.2], aspect=1/30)
    im, xpts = _plot_xsect(q_xsect, lat_xsect, lon_xsect, z,
                     plt.cm.RdBu_r, 0, 0.02, [250, 50], ax_xsectQ)
    _plot_LAB_data(lat_xsect, lon_xsect, vs_xsect, x_pts, z, ax_xsectQ)
    c = plt.colorbar(im, cax=f.add_axes([0.1, 0.06, 0.5, 0.01]), orientation='horizontal')
    c.ax.set_xlabel('1/Qmu')

    ax_map = f.add_axes([0.05, 0.65, 0.6, 0.3], aspect=1.4)
    d_slice = 125
    im2 = _plot_map(vs[:, :, np.argmax(d_slice <= z)], lats, lons,
                    plt.cm.RdBu, vmi, vma, ax_map)
    # c2 = plt.colorbar(im2, cax=f.add_axes([0.95, 0.2, 0.005, 0.7]),
    #                   orientation='vertical')
    # c2.ax.set_ylabel('Vsv at {:.0f} km depth (km/s)'.format(d_slice))
    ax_map.plot(convert_latlons(lon_xsect, lons),
                convert_latlons(lat_xsect, lats), 'k-')

    # Plot depth sections
    # vs_P14 = interpolate_lit_model('P14', z, lats, lons)
    depth_locs = ((45, -111), (40.7, -117.5), (39, -109.8), (37.2, -100.9))
    box_depths = ((75, 105), (75, 105), (120, 150), (120, 150))
    y = 0.75
    dist = 2
    for location, depth_range in zip(depth_locs, box_depths):
        iz = list(range(np.argmax(depth_range[0] <= z), np.argmax(depth_range[1] < z)))
        ax_z = f.add_axes([0.74, y, 0.075, 0.15])
        # vs_mean, vs_std = _plot_depth_section(location, dist, vs_P14,
        #                                       lats, lons, z, ax_z)
        # print('{}. Mean vel {}-{} km: {:.2f} +- {:.2f} km/s'.format(
        #     depth_locs.index(location), depth_range[0], depth_range[1],
        #     np.mean(vs_mean[iz]), np.mean(vs_std[iz])
        # ))
        vs_mean, vs_std = _plot_depth_section(location, dist, vs[:, :, z <= 150],
                            lats, lons, z[z <= 150], ax_z)
        print('{}. Mean vel {}-{} km: {:.2f} +- {:.2f} km/s'.format(
            depth_locs.index(location), depth_range[0], depth_range[1],
            np.mean(vs_mean[iz]), np.mean(vs_std[iz])
        ))
        ax_z.plot(np.mean(vs_mean[iz]) + np.array([-0.2, 0.2, 0.2, -0.2, -0.2]),
                  [depth_range[0], depth_range[0], depth_range[1], depth_range[1], depth_range[0]],
                  'k--')
        ax_z.set(xlim=[3.7, 4.8], ylim=[300, 0],
                 ylabel='Depth (km)', xlabel='Vs (km/s)')

        ax_z.plot()
        ax_z.plot()
        ax_map.text(convert_latlons(location[1], lons),
                    convert_latlons(location[0], lats),
                    '{:.0f}'.format(depth_locs.index(location) + 1))
        ax_map.plot(convert_latlons(location[1], lons),
                    convert_latlons(location[0], lats), 'k.')
        ax_z.text(2.5, 10, '{:.0f}.'.format(depth_locs.index(location) + 1))
        y -= 0.22

        if depth_locs.index(location) > 0:
            i = np.argmax(xpts > location[1])
            ax_xsect.plot(xpts[i], -50, 'k.', markersize=10)
            ax_xsect.text(xpts[i], -60,
                          '{:.0f}'.format(depth_locs.index(location) + 1),
                           horizontalalignment='center')
            ax_xsect.plot(xpts[[i-2, i+2, i+2, i-2, i-2]],
                [depth_range[0], depth_range[0], depth_range[1], depth_range[1], depth_range[0]],
                'k--')
            ax_xsectQ.plot(xpts[[i-2, i+2, i+2, i-2, i-2]],
                [depth_range[0], depth_range[0], depth_range[1], depth_range[1], depth_range[0]],
                'k--')

    y = 0.75
    for location, depth_range in zip(depth_locs, box_depths):
        z2 = z[z > 50].copy()
        iz = list(range(np.argmax(depth_range[0] <= z2), np.argmax(depth_range[1] < z2)))
        ax_z = f.add_axes([0.9, y, 0.075, 0.15])
        q_mean, q_std = _plot_depth_section(location, dist, 1 / q[:, :, z > 50],
                            lats, lons, z2, ax_z)
        print('{}. Mean Qinv {}-{} km: {:.2f} +- {:.2f}'.format(
            depth_locs.index(location), depth_range[0], depth_range[1],
            np.mean(q_mean[iz]), np.mean(q_std[iz])
        ))
        ax_z.plot(np.mean(q_mean[iz]) + np.array([-30, 30, 30, -30, -30]),
                  [depth_range[0], depth_range[0], depth_range[1], depth_range[1], depth_range[0]],
                  'k--')
        ax_z.set(xlim=[0, 250], ylim=[300, 0],
                 ylabel='Depth (km)', xlabel='Q')

        ax_z.plot()
        ax_z.plot()
        ax_map.text(convert_latlons(location[1], lons),
                    convert_latlons(location[0], lats),
                    '{:.0f}'.format(depth_locs.index(location) + 1))
        ax_map.plot(convert_latlons(location[1], lons),
                    convert_latlons(location[0], lats), 'k.')
        y -= 0.22


    return #ax_map, ax_xsect

def _plot_depth_section(location, dist, vs, lats, lons, z, ax_z):

    vs_lim = _find_lat_lon_within_distance(location, dist, vs, lats, lons)
    vs_std = np.std(vs_lim, 0)
    vs_mean = np.mean(vs_lim, 0)
    ax_z.fill(np.append(vs_mean + vs_std, vs_mean[::-1] - vs_std[::-1]),
           np.append(z, z[::-1]), color='#d4d4d4')
    ax_z.plot(vs_mean, z, '-', label=location)

    return vs_mean, vs_std



def _plot_LAB_data(lats, lons, vs, x_pts, z, ax_sect):
    # Calculate LAB depth
    nvg_tts, nvg_amps, lab_tts, lab_amps = _extract_rf_constraints(lats, lons)
    nvg_z = _convert_tt_to_z(nvg_tts, vs, z)
    lab_z = _convert_tt_to_z(lab_tts, vs, z)
    lab_amps /= max(nvg_amps) / 2
    nvg_amps /= max(nvg_amps) / 2

    # Plot NVG data
    for x, z1, z1a, z2, z2a in zip(x_pts, nvg_z, nvg_amps, lab_z, lab_amps):
        ax_sect.plot(x, z1, '+', markersize=z1a, color='#BFBFBF')
        ax_sect.plot(x, z2, 'kx', markersize=z2a)


def _plot_xsect(vals, lats, lons, z, cbar, vmi, vma, ylims, ax_sect):
    if np.abs(np.diff(lats[[0, -1]])) < np.abs(np.diff(lons[[0, -1]])):
        x_pts = lons
        x_lab = 'Longitude (E)'
    else:
        x_pts = lats
        x_lab = 'Latitude (N)'

    vals[vals < vmi] = vmi
    vals[vals > vma] = vma
    im = ax_sect.contourf(x_pts, z, vals.T, 20, cmap=cbar, vmin=vmi, vmax=vma)

    ax_sect.set(ylabel='Depth (km)', xlabel=x_lab, ylim=ylims)


    return im, x_pts


def _plot_map(vals, lats, lons, cbar, vmi, vma, ax_map):
    im = plt.imshow(vals[:, :], cmap=cbar, aspect=1.4, vmin=vmi, vmax=vma)
    lon_inc = int(np.ceil(len(lons) / 20) / np.diff(lons[:2]))
    lat_inc = int(np.ceil(len(lats) / 20) / np.diff(lons[:2]))
    ax_map.set(xticks=range(0, len(lons), lon_inc),
               xticklabels = [int(abs(lon)) for lon in lons[::lon_inc]],
               yticks=range(0,len(lats), lat_inc),
               yticklabels = [int(lat) for lat in lats[::lat_inc]],
               )
    # ax_map.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
    # ax_map.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
    #            xlabel='Longitude (W)', ylabel='Latitude (N)')
    ax_map.set_ylim((0, len(lats) - 1))
    ax_map.set_xlim((0, len(lons) - 1))
    _plot_States(ax_map, lats, lons)
    _plot_Colorado_Plateau(ax_map, lats, lons)


    return im

def _plot_Colorado_Plateau(ax_map, lats, lons):
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values
    cp_outline = np.vstack((cp_outline, cp_outline[0, :]))
    ax_map.plot(convert_latlons(cp_outline[:, 1], lons),
                convert_latlons(cp_outline[:, 0], lats), 'k:')

def _plot_States(ax_map, lats, lons):
    states = ['Washington', 'Oregon', 'California', 'Nevada',
              'Idaho', 'Utah', 'Arizona',
              'Montana', 'Wyoming', 'Colorado', 'New_Mexico',
              'North_Dakota', 'South_Dakota', 'Nebraska', 'Kansas', 'Oklahoma', 'Texas',
              'Minnesota', 'Iowa', 'Missouri', 'Arkansas', 'Louisiana',
              'Michigan_UP', 'Wisconsin', 'Illinois','Mississippi',
              'Michigan_LP', 'Indiana', 'Ohio', 'Kentucky', 'Tennessee', 'Alabama']
    for state in states:
        border = pd.read_csv('data/earth_models/US_States/' + state + '.csv',
                             header=None).values
        ax_map.plot(convert_latlons(border[:, 1], lons),
                    convert_latlons(border[:, 0], lats), '-', color='#BFBFBF')

def convert_latlons(map_points, loc_vector):
    return (map_points - loc_vector[0]) / np.diff(loc_vector[:2])

def _get_topo(lats, lons):
    all_topo = pd.read_csv('data/earth_models/US_topo.csv')
    all_topo.columns = ['lat', 'lon', 'topo']
    topo = []
    for lat, lon in zip(lats, lons):
        topo += [all_topo.loc[_find_closest_lat_lon(all_topo, (lat, lon)), 'topo']]

    return np.array(topo) / 1000 # in km


def interpolate_lit_model(ref, z, lats, lons):
    """ Load a literature Vs model on a given grid of z, latitude and longitude

    Arguments:
        ref:
            - string
            - a shorthand denoting which model to load
              (see load_literature_vel_model)
        z:
            - (n_depth_points) np.array
            - depth vector (km)
        lats:
            - (n_latitude_points) np.array
            - latitude vector
        lons:
            - (n_longitude_points) np.array
            - longitude vector

    Returns:
        vs_a:
            - (n_latitude_points, n_longitude_points, n_depth_points) np.array
            - cube of Vs data from the model denoted by 'ref'

    """
    ds = load_literature_vel_model(ref)
    z_a = ds.depth.values
    lats_a = ds.latitude.values
    lons_a = ds.longitude.values
    if lons_a[0] > 0:
        lons_a -= 360

    vs_a = np.zeros((len(lats), len(lons), len(z)))
    ila = 0
    ilo = 0
    for lat in lats:
        for lon in lons:

            i_lat = np.argmin(np.abs(lats_a - lat))
            i_lon = np.argmin(np.abs(lons_a - lon))
            if abs(lats_a[i_lat] - lat) > 1:
                print('Nearest latitude is {}'.format(lats_a[i_lat]))
            if abs(lons_a[i_lon] - lon) > 1:
                print('Nearest longitude is {}'.format(lons_a[i_lon]))

            vs_a[ila, ilo, :] = np.interp(z, z_a, ds.vs.values[:, i_lat, i_lon])

            ilo += 1
        ila += 1
        ilo = 0

    return vs_a

def load_literature_vel_model(ref:str):
    """ Load a published Vs model as a dataset.

    IRIS EMC has many models of the WUS avaiable to download.  These are models
    including surface wave data with absolute Vs values (note that some were still
    reported as dVs with a given reference model).

    Models listed:
        SR16:       Shen & Ritzwoller, 2016     - fits Moho Ps RFs
        S15:        Schmandt et al., 2015       - fits Moho Ps RFs
        P14:        Porrit et al., 2014         - reported as dVs relative to WUS
        P15:        Porter et al., 2015
        F18:        Fichtner et al., 2018
        Y14:        Yuan et al., 2014           - no Vs modelled < 50 km depth
        C15:        Chai et al., 2015           - includes fitting RFs throughout

    Arguments:
        ref:
            - string
            - one of the shorthands listed above denoting which model to load
    Returns:
        ds
            - xarray dataset with at least the following fields
                - vs:           absolute shear velocity (km/s)
                - depth:        depth (km)
                - latitude:     (degrees N)
                - longitude:    (degrees E)

    """

    if ref == 'SR16':
        nm = 'data/earth_models/US.2016.nc'
        url = 'https://doi.org/10.17611/DP/EMCUS2016'
        ref = 'Shen & Ritzwoller, 2016 (10.1002/2016JB012887)'
        v_field = 'vsv'
        ref_v = np.array([])
    elif ref == 'S15':
        nm = 'data/earth_models/US-CrustVs-2015_kmps.nc'
        url = 'https://doi.org/10.17611/DP/EMCUSCRUSTVS2015'
        ref = 'Schmandt et al., 2015 (10.1002/2015GL066593)'
        v_field = 'vs'
        ref_v = np.array([])
    elif ref == 'P14':
        nm = 'data/earth_models/DNA13_percent.nc'
        url = 'https://doi.org/10.17611/DP/9991615'
        ref = 'Porrit et al., 2014 (10.1016/j.epsl.2013.10.034)'
        v_field = 'vsvj'
        ref_v = np.array([ # WUS model, Pollitz (2008) Fig. 17: 10.1029/2007JB005556
            [2.40, 0], [2.45, 0.5], [3.18, 4.5], [3.20, 18], [3.90, 20],
            [3.90, 33], [4.30, 35], [4.30, 60], [4.26, 65], [4.26, 215],
            [4.65, 220], [4.65, 240], [4.74, 242], [4.74, 300]
            ])
    elif ref == 'P15':
        nm = 'data/earth_models/US-Crust-Upper-mantle-Vs.Porter.Liu.Holt.2015_kmps.nc'
        url = 'https://doi.org/10.17611/DP/EMCUCUMVPLH15MLROWCU'
        ref = 'Porter et al., 2015 (10.1002/2015GL066950)'
        v_field = 'vs'
        ref_v = np.array([])
    elif ref == 'F18':
        nm = 'data/earth_models/csem-north-america-2019.12.01.nc'
        url = 'https://doi.org/10.17611/dp/emccsemnamatl20191201'
        ref = 'Fichtner et al., 2018 (10.1029/2018GL077338)'
        v_field = 'vsv'
        ref_v = np.array([])
    elif ref == 'Y14':
        nm = 'data/earth_models/SEMum-NA14_kmps.nc'
        url = 'https://doi.org/10.17611/dp/EMCSEMUMNA14'
        ref = 'Yuan et al., 2014 (10.1016/j.epsl.2013.11.057)'
        v_field = 'Vs'
        ref_v = np.array([])
    elif ref == 'C15':
        nm = 'data/earth_models/WUS-CAMH-2015.nc'
        url = 'https://doi.org/10.17611/dp/EMCWUSCAMH2015'
        ref = 'Chai et al., 2015 (10.1002/2015GL063733)'
        v_field = 'vs'
        ref_v = np.array([])
    elif ref == 'DE08':
        return load_Dalton_Ekstrom()


    else:
        print('Unknown reference - try again')
        return


    try:
        ds = xr.open_dataset(nm)
    except:

        print(
            'You need to download the {} model from'.format(ref)
            + ' IRIS EMC\n\t{} \nand save to \n\t{}'.format(url, nm)
        )
        return

    if ref_v.any(): # Convert perturbations to absolute values
        imax = np.argmax(ds.depth.values > ref_v[-1, 1])
        ref_vz = np.interp(ds.depth.values[:imax], ref_v[:, 1], ref_v[:, 0])
        dvs = ds[v_field].values[:imax, :, :]
        vs = np.zeros_like(dvs)
        for ila in range(dvs.shape[1]):
            for ilo in range(dvs.shape[2]):
                vs[:, ila, ilo] = (1 + (dvs[:, ila, ilo] / 100)) * ref_vz

        ds = xr.Dataset(
            {'vs': (['depth', 'latitude', 'longitude'],  vs)},
            coords={'longitude': (['longitude'], ds.longitude.values),
                    'latitude': (['latitude'], ds.latitude.values),
                    'depth': ds.depth.values[:imax],
                    }
        )
        v_field = 'vs'

    if v_field != 'vs':
        ds['vs'] = ds[v_field]

    return ds

def load_Dalton_Ekstrom():
    depths = np.arange(50, 401, 50)
    dir = 'data/earth_models/DaltonEkstrom08/'
    url = 'http://www.geo.brown.edu/research/Dalton/assets/qrfsi12_files.tar.gz'
    q_all = np.zeros((len(depths), 180, 360))
    i = 0
    for dep in depths:
        filename = 'qrfsi12_' + '{:.0f}'.format(dep).zfill(3) + '_final'
        try:
             q = pd.read_csv(dir + filename, header=None, sep='\s+')
             q.columns = ['latitude', 'longitude', '1/Qmu']
        except:
            print('Download {:s} from \n\t{:s}'.format(filename, url))
        q_all[i, :, :] = q['1/Qmu'].values.reshape(180, 360)
        i += 1

    ds = xr.Dataset(
        {'vs': (['depth', 'latitude', 'longitude'],  q_all)},
        coords={'longitude': (['longitude'], q.longitude.unique()),
                'latitude': (['latitude'], q.latitude.unique()),
                'depth': depths,
                }
        )

    return ds





def _extract_rf_constraints(lats, lons):
    """ Pull RF observeables of Strongest NVG and LAB from Hoper & Fischer, 2018.

    """
    nvg_tt = []
    nvg_amp = []
    lab_tt = []
    lab_amp = []

    # Load in receiver function constraints
    all_rfs = pd.read_csv('data/RFconstraints/Hopper_Fischer.csv')
    for lat, lon in zip(lats, lons):
        ind = _find_closest_lat_lon(all_rfs, (lat, lon))
        nvg_tt += [all_rfs.loc[ind, 'ttNVG']]
        lab_tt += [all_rfs.loc[ind, 'ttLAB']]
        nvg_amp += [all_rfs.loc[ind, 'ampNVG']]
        lab_amp += [all_rfs.loc[ind, 'ampLAB']]


    return np.array(nvg_tt), np.array(nvg_amp), np.array(lab_tt), np.array(lab_amp)

def _convert_tt_to_z(tts, vs, z):

    zs = np.zeros_like(tts)
    vpvs_ratio = 1.75 # scale from Vp to Vs
    for i in range(len(tts)):
        if np.isnan(tts[i]):
            zs[i] = tts[i]
        else:
            tt_model = np.cumsum(
                np.diff(z) / np.mean(np.vstack((vs[i, :-1], vs[i, 1:])), 0)
            )
            zs[i] = z[np.argmax(tt_model > tts[i] * vpvs_ratio)]

    return zs

def _find_closest_lat_lon(df:pd.DataFrame, location: tuple):
    """ Find index in dataframe of closest point to lat, lon.

    Assumes that the dataframe has (at least) two columns, lat and lon, which
    are in °N and °E respectively.

    """
    lat, lon = location
    # Make sure all longitudes are in range -180 to 180
    if lon > 180:
        lon -= 360
    df.loc[df['lon'] > 180, 'lon'] -= 360

    df['distance_squared'] = (df['lon'] - lon)**2 + (df['lat'] - lat)**2
    min_ind = df['distance_squared'].idxmin()

    return min_ind

def _find_lat_lon_within_distance(location, dist, vs, lats, lons):
    """ Find all points in numpy array within dist of location.

    """
    lat, lon = location
    # Make sure all longitudes are in range -180 to 180
    if lon > 180:
        lon -= 360
    lons[lons > 180] -= 360

    lons_g, lats_g = np.meshgrid(lons, lats)
    distance_squared = (lons_g - lon)**2 + (lats_g - lat)**2

    return vs[distance_squared < dist, :]
