""" Constraints on the inversion.

This includes observed phase velocities and constraints pulled from receiver
function observations.
"""

import typing
import numpy as np
import pandas as pd
import re
import os

from util import define_models


# =============================================================================
# Set up classes for commonly used variables
# =============================================================================


# =============================================================================
#       Extract the observations of interest for a given location
# =============================================================================

def extract_observations(setup_model:define_models.SetupModel):
    """ Make an Observations object.
    """
    lat, lon = setup_model.location
    surface_waves = _extract_phase_vels(lat, lon)
    #surface_waves = surface_waves.iloc[[3, 5, 7, 11, 12, 13], :]
    rfs = _extract_rf_constraints(lat, lon, setup_model)

    # Write to file
    if not os.path.exists('output/' + setup_model.id):
        os.mkdir('output/' + setup_model.id)
    surface_waves.to_csv(
        'output/{0}/{0}_surface_wave_constraints.csv'.format(setup_model.id)
    )
    rfs.to_csv(
        'output/{0}/{0}_RF_constraints.csv'.format(setup_model.id)
    )

    d = np.vstack((surface_waves['ph_vel'][:, np.newaxis],
                   rfs['tt'][:, np.newaxis],
                   rfs['dv'][:, np.newaxis]))
    std = np.vstack((surface_waves['std'][:, np.newaxis],
                   rfs['ttstd'][:, np.newaxis],
                   rfs['dvstd'][:, np.newaxis]))
    periods = surface_waves['period'].values

    return d, std, periods

def _extract_rf_constraints(lat:float, lon:float,
                            setup_model:define_models.SetupModel):
    """

    Note that some of the reported standard deviation on values are
    unrealistically low (i.e. zero), so we will assume the minimum standard
    deviation on a value is the 10% quantile of the total data set.
    """

    # Load in receiver function constraints
    all_rfs = pd.read_csv('data/RFconstraints/a_priori_constraints.csv')
    ind = _find_closest_lat_lon(all_rfs.copy(), lat, lon)
    obs = all_rfs.loc[ind]

    rfs = pd.DataFrame(
        columns = ['lat', 'lon', 'tt', 'ttstd', 'dv', 'dvstd']
    )
    ib = 0
    for bound in setup_model.boundary_names:
        # Extract travel time information
        try:
            tt = obs['tt' + bound]
            ttstd = obs['tt' + bound + 'std']

            min_allowed_ttstd = all_rfs['tt' + bound + 'std'].quantile(0.1)
            ttstd = max((ttstd, min_allowed_ttstd))
        except:
            print('No RF constraints on travel time for ' + bound)
            return

        # If necessary, scale to Vs travel time
        try:
            rftype = obs['type' + bound]
        except:
            rftype = 'Ps'
            print('RF type unspecified for ' + bound + ' - assuming Ps')
        if rftype == 'Sp': # Scale travel time from travelling at Vp to at Vs
            tt *= setup_model.vpv_vsv_ratio
            # for constant a, variable A: sigma_aA = |a| * sigma_A
            ttstd *= setup_model.vpv_vsv_ratio

        # Extract velocity contrast information
        try:
            dv = obs['dv' + bound]
            dvstd = obs['dv' + bound + 'std']

            min_allowed_dvstd = all_rfs['dv' + bound + 'std'].quantile(0.1)
            ampstd = max((dvstd, min_allowed_dvstd))
        except:
            try:
                amp = obs['amp' + bound]
                ampstd = obs['amp' + bound + 'std']

                min_allowed_amstd = all_rfs['amp' + bound + 'std'].quantile(0.1)
                ampstd = max((ampstd, min_allowed_amstd))

                type = obs['type' + bound]
                dv, dvstd = _convert_amplitude_to_dv(
                    amp, ampstd, type, setup_model.boundary_widths[ib]
                )
            except:
                print('No RF constraints on dV for ' + bound)
                return

        rfs = rfs.append(
            pd.Series([lat, lon, tt, ttstd, dv, dvstd], index=rfs.columns),
            ignore_index=True,
        )

        ib += 1


    return rfs

def _convert_amplitude_to_dv(amp, ampstd, rftype, boundary_width):
    """

    Calculated the Sp synthetics in MATLAB for a variety of dV (where
    dV = (1 - v_bottom / v_top) * 100 given the width of the boundary
    layer (here labelled 'breadth').  These are saved as individual .csv
    files, where the breadth of the synthetic layer is in the file name,
    and each row represents a different input dV and output phase amplitude.

    synthvals = pd.DataFrame(columns = ['breadth', 'dv', 'amplitude'])
    for b in range(0, 51, 5):
        a = pd.read_csv('synthvals_' + str(b) + '.0.csv', header=None,
                        names = ['dv', 'amplitude'])
        a['breadth'] = b
        a['dv'] *= -1 / 100
        synthvals = synthvals.append(a, ignore_index=True, sort=False)
    synthvals.to_csv('data/RFconstraints/synthvals_Sp.csv', index=False)
    """

    try:
        synth = pd.read_csv('data/RFconstraints/synthvals_' + rftype +'.csv')
    except:
        print('No synthetic amplitudes calculated for ' + rftype + '!')
        return

    synth = synth[synth.breadth == boundary_width]
    # Note that numpy default for interpolation is x < x[0] returns y[0]
    dv = np.round(np.interp(amp, synth.amplitude, synth.dv), 3)
    dvstd = abs(np.round(np.interp(ampstd, synth.amplitude, synth.dv), 3))

    return dv, dvstd

def _extract_phase_vels(lat:float, lon:float):

    phv = _load_observed_sw_constraints()

    surface_waves = pd.DataFrame()
    for period in phv['period'].unique():
        ind = _find_closest_lat_lon(
            phv[phv['period'] == period].copy(), lat, lon
        )
        surface_waves = surface_waves.append(phv.loc[ind])
    surface_waves = (surface_waves.sort_values(by=['period'], ascending=True)
        .reset_index(drop=True))
    # Should actually load in some std!!!!  Will be set to 0.15 if  == 0 later
    surface_waves['std'] = 0.

    return surface_waves.reset_index(drop=True)

def _load_observed_sw_constraints():
    """ Load surface wave constraints into pandas.

    Very specific to the way the data is currently stored!  See READMEs in the
    relevent directories.
    """

    # Load in surface waves
    data_dir = 'data/obs_dispersion/'
    phvel = pd.DataFrame()
    # Have ASWMS data with filenames 'helmholtz_stack_LHZ_[period].xyz'
    # and ambient noise data with filenames 'R[period]_USANT15.txt'
    for file in os.listdir(data_dir):
        if 'USANT15' in file: # Ambient noise
            phvel = phvel.append(_load_ambient_noise(data_dir, file))
        if 'helmholtz' in file: # ASWMS data
            phvel = phvel.append(_load_earthquake_sw(data_dir, file))
    phvel.reset_index(drop=True, inplace=True)

    return phvel


def _load_ambient_noise(data_dir:str, file:str):
    """
    """
    # Find period of observations
    # Filenames are R[period]_USANT15.pix, so split on R or _
    period = float(re.split('R|_', file)[1])

    # Find reference phase velocity
    with open(data_dir + file, 'r') as fid:
        for line in fid:
            if 'PVELREF' in line:
                break
    ref_vel = float(line.split()[1])

    # Load file
    # data structure: 1. geocentric latitude, longitude, pixel size, deviation
    ambient_noise = pd.read_csv(data_dir + file, header=None,
        skiprows=11, sep='\s+')
    ambient_noise.columns = ['geocentric_lat', 'lon', 'size', 'dV']
    ambient_noise['ph_vel'] = (1 + ambient_noise['dV'] / 100) * ref_vel
    # convert to geodetic latitude,
    #       tan(geocentric_lat) = (1 - f)**2 * tan(geodesic_lat)
    # https://en.wikipedia.org/wiki/Latitude#Geocentric_latitude
    WGS84_f = 1 / 298.257223563  # flattening for WGS84 ellipsoid
    ambient_noise['lat'] = np.degrees(np.arctan(
        np.tan(np.radians(ambient_noise['geocentric_lat'])) / (1 - WGS84_f)**2
    ))
    ambient_noise['period'] = period

    return ambient_noise[['period', 'lat', 'lon', 'ph_vel']]

def _load_earthquake_sw(data_dir:str, file:str):
    """
    """
    # Find period of observations
    # Filenames are R[period]_USANT15.pix, so split on R or _
    period = float(re.split('_|\.', file)[-2])

    surface_waves = pd.read_csv(data_dir + file, sep='\s+', header=None)
    surface_waves.columns = ['lat', 'lon', 'ph_vel']
    surface_waves['period'] = period

    return surface_waves[['period', 'lat', 'lon', 'ph_vel']]

def _find_closest_lat_lon(df:pd.DataFrame, lat:float, lon:float):
    """ Find index in dataframe of closest point to lat, lon.

    Assumes that the dataframe has (at least) two columns, Lat and Lon, which
    are in °N and °E respectively.

    """
    # Make sure all longitudes are in range -180 to 180
    if lon > 180:
        lon -= 360
    df.loc[df['lon'] > 180, 'lon'] -= 360

    df['distance_squared'] = (df['lon'] - lon)**2 + (df['lat'] - lat)**2
    min_ind = df['distance_squared'].idxmin()

    if df.loc[min_ind, 'distance_squared'] > 1:
        print('!!!!!! Closest observation at {}°N, {}°E !!!!!!'.format(
            df.loc[min_ind, 'lat'], df.loc[min_ind, 'lon']))

    return min_ind

def get_vels_Crust1(lat, lon):
    """
    Crust 1.0 is given in a 1 degree x 1 degree grid (i.e. 360 lon points, 180
    lat points).  The downloads are structured as
        crust1.bnds  (360 * 180) x 9 depths to top of each layer
                             0. water (i.e. topography)
                             1. ice (i.e. bathymetry)
                             2. upper sediments (i.e. depth to rock)
                             3. middle sediments
                             4. lower sediments
                             5. upper crust (i.e. depth to bedrock)
                             6. middle crust
                             7. lower crust
                             8. mantle (i.e. Moho depth)
    Note that for places where a layer doesn't exist, the difference between
    bnds[i, n] and bnds[i, n+1] = 0; i.e. for continents, the top of the ice is
    the same as the top of the water; where there are no glaciers, the top of
    sediments is the same as the top of the ice, etc.

        crust1.[rho|vp|vs]  (360 * 180) x 9 values of density, Vp, Vs for each
                            of the layers specified in bnds

    Each row in these datasets steps first in longitude (from -179.5 to +179.5)
    then in latitude (from 89.5 to -89.5).
        i.e. index of (lat, lon) will be at (lon + 179.5) + (89.5 - lat) * 360


    """

    lons = np.arange(-179.5,180,1)
    lats = np.arange(89.5,-90,-1)
    i = int((lon - lons[0]) + ((lats[0] - lat) // 1) * len(lons))

    nm = 'data/earth_models/crust1/crust1.'
    cb = pd.read_csv(nm + 'bnds', skiprows=i, nrows=1, header=None, sep='\s+'
        ).values.flatten()
    vs = pd.read_csv(nm + 'vs', skiprows=i, nrows=1, header=None, sep='\s+'
        ).values.flatten()
    vp = pd.read_csv(nm + 'vp', skiprows=i, nrows=1, header=None, sep='\s+'
        ).values.flatten()
    rho = pd.read_csv(nm + 'rho', skiprows=i, nrows=1, header=None, sep='\s+'
        ).values.flatten()

    thickness = -np.diff(cb)
    ib = 0
    m_t = [0]
    m_vs = []
    for t in thickness:
        if t > 0:
            m_vs += [vs[ib]]
            m_t += [t]
        ib += 1
