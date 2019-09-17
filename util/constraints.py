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


class Observations(typing.NamedTuple):
    """ Observed constraints from surface wave dispersion & receiver functions

    Fields:
        - surface_waves
            - pd.DataFrame with columns:
                - period
                    - Units:    s
                    - Dominant period of dispersion measurement
                - c
                    - Units:    km/s
                    - Phase velocity
                - std
                    - Units:    km/s
                    - Standard deviation of measurement at each period
            - At time of writing, data/obs_dispersion files come from
              ASWMS estimates from IRIS (long periods) and Ekstrom et al., 2017
              (ambient noise; periods < 20s).  See README in that directory
              for more information on origin & format.
        - receiver_functions
            - pd.DataFrame with columns:
                - ttMoho
                    - Units:    s
                    - Travel time to the Moho (average)
                - ttMohostd
                    - Units:    s
                    - Standard deviation of travel time to the Moho
                - dvMoho
                    - Units:    km/s
                    - Velocity change at the Moho (average)
                - dvMohostd
                    - Units:    km/s
                    - Standard deviation of velocity change at the Moho
                - ttLAB
                    - Units:    s
                    - Travel time to the LAB (average)
                - ttLABstd
                    - Units:    s
                    - Standard deviation of travel time to the LAB
                - ampLAB
                    - Units:    none
                    - Amplitude of LAB phase
                - ampLABstd
                    - Units:    none
                    - Standard deviation of LAB amplitude
            - At time of writing, data/RFconstraints/a_prior_constraints.txt
              come from Shen et al., 2012 (Moho); Hopper & Fischer, 2018 (LAB).
              See README in that directory for more information on origin.
        - latitude
            - Units:    °N
        - longitude
            - Units:    °E
            - Assumes -180 to +180, not 0 to 360
    """
    surface_waves: pd.DataFrame
    receiver_functions: pd.DataFrame
    latitude: float
    longitude: float


# =============================================================================
#       Extract the observations of interest for a given location
# =============================================================================

def extract_observations(lat:float, lon:float,
                         setup_model:define_models.SetupModel):
    """ Make an Observations object.
    """

    surface_waves = _extract_phase_vels(lat, lon)
    rfs = _extract_rf_constraints(lat, lon, setup_model)
    # Write to file
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
    periods = surface_waves['period']

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
    cols = list(obs.index)

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
            tt = 0
            ttstd = 0
            print('No RF constraints on travel time for ' + bound)

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
                dv = 0
                dvstd = 0
                print('No RF constraints on dV for ' + bound)

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
