""" Constraints on the inversion.

This includes observed phase velocities and constraints pulled from receiver
function observations.
"""

import typing
import numpy as np
import pandas as pd
import re
import os
import xarray as xr # for loading netcdf



# =============================================================================
# Set up classes for commonly used variables
# =============================================================================


# =============================================================================
#       Extract the observations of interest for a given location
# =============================================================================

def extract_observations(location:tuple, id:str, boundaries:tuple, vpvs:float):
    """ Make an Observations object.
    """

    surface_waves = _extract_phase_vels(location)
    #surface_waves = surface_waves.iloc[[3, 5, 7, 11, 12, 13], :]
    rfs = _extract_rf_constraints(location, id, boundaries, vpvs)

    # Write to file
    if not os.path.exists('output/' + id):
        os.mkdir('output/' + id)
    savedir = 'output/{0}/{0}_'.format(id)
    surface_waves.to_csv('{}_surface_wave_constraints.csv'.format(savedir, id))
    rfs.to_csv('{}_RF_constraints.csv'.format(savedir, id))

    d = np.vstack((surface_waves['ph_vel'][:, np.newaxis],
                   rfs['tt'][:, np.newaxis],
                   rfs['dv'][:, np.newaxis]))
    std = np.vstack((surface_waves['std'][:, np.newaxis],
                   rfs['ttstd'][:, np.newaxis],
                   rfs['dvstd'][:, np.newaxis]))
    periods = surface_waves['period'].values

    return d, std, periods

def _extract_rf_constraints(location:tuple, id:str, boundaries:pd.DataFrame,
                            vpvs:float):
    """

    Note that some of the reported standard deviation on values are
    unrealistically low (i.e. zero), so we will assume the minimum standard
    deviation on a value is the 10% quantile of the total data set.
    """
    bnames, bwidths = boundaries

    # Load in receiver function constraints
    all_rfs = pd.read_csv('data/RFconstraints/a_priori_constraints.csv')
    ind = _find_closest_lat_lon(all_rfs.copy(), location)
    obs = all_rfs.loc[ind]

    rfs = pd.DataFrame(
        columns = ['lat', 'lon', 'tt', 'ttstd', 'dv', 'dvstd']
    )
    ib = 0
    for bound in bnames:
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
            tt *= vpvs
            # for constant a, variable A: sigma_aA = |a| * sigma_A
            ttstd *= vpvs

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

                dv, dvstd = _convert_amplitude_to_dv(
                    amp, ampstd, rftype, bwidths[ib]
                )
            except:
                print('No RF constraints on dV for ' + bound)
                return

        lat, lon = location
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

def _extract_phase_vels(location: tuple):

    phv = _load_observed_sw_constraints()

    surface_waves = pd.DataFrame()
    for period in phv['period'].unique():
        ind = _find_closest_lat_lon(
            phv[phv['period'] == period].copy(), location
        )
        surface_waves = surface_waves.append(phv.loc[ind])
    surface_waves = (surface_waves.sort_values(by=['period'], ascending=True)
        .reset_index(drop=True))
    # Should actually load in some std!!!!  Going to do a random estimate
    for index, row in surface_waves.iterrows():
        surface_waves.loc[index, 'std'] = (
            phv[phv.period == row.period].ph_vel.std() / 2
        )

    return surface_waves.reset_index(drop=True)

def _load_observed_sw_constraints():
    """ Load surface wave constraints into pandas.

    Very specific to the way the data is currently stored!  See READMEs in the
    relevent directories.
    """

    # Load in surface waves
    data_dir = 'data/obs_dispersion/'
    phvel = pd.DataFrame()
    # Have ASWMS data with filenames 'helmholtz_stack_LHZ_[period].xyz',
    # longer period surface wave data with filenames 'c_[period]s_BD19',
    # and ambient noise data with filenames 'R[period]_USANT15.txt'
    for file in sorted(os.listdir(data_dir), reverse=True, key=str.lower):
        if 'USANT15' in file: # Ambient noise
            phvel = phvel.append(_load_ambient_noise(data_dir, file))
        if 'helmholtz' in file: # ASWMS data
            phvel = phvel.append(_load_earthquake_sw(data_dir, file))
        if 'BD19' in file: # Longer T surface waves: Babikoff & Dalton, 2019
            c = _load_earthquake_sw_BD19(data_dir, file)
            if c.period[0] not in phvel.period.unique():
                phvel = phvel.append(c)
    phvel = phvel.sort_values(by=['period', 'lat', 'lon']).reset_index(drop=True)

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

def _load_earthquake_sw_BD19(data_dir:str, file:str):
    """
    """
    # Find period of observations
    # Filenames are c_[period]s_BD19, so split on _ and s
    period = float(re.split('_|s', file)[1])

    surface_waves = pd.read_csv(data_dir + file, sep='\s+', header=None)
    surface_waves.columns = ['lat', 'lon', 'ph_vel']
    surface_waves['period'] = period

    # Phase velocities are reported in m/s - convert to km/s
    surface_waves['ph_vel'] /= 1000

    return surface_waves[['period', 'lat', 'lon', 'ph_vel']]

def _find_closest_lat_lon(df:pd.DataFrame, location: tuple):
    """ Find index in dataframe of closest point to lat, lon.

    Assumes that the dataframe has (at least) two columns, Lat and Lon, which
    are in °N and °E respectively.

    """
    lat, lon = location
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

def get_vels_Crust1(location):
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
    lat, lon = location

    all_lons = np.arange(-179.5,180,1)
    all_lats = np.arange(89.5,-90,-1)
    i = int((lon - all_lons[0]) + ((all_lats[0] - lat) // 1) * len(all_lons))

    nm = 'data/earth_models/crust1/crust1.'
    try:
        cb = pd.read_csv(nm + 'bnds', skiprows=i, nrows=1, header=None, sep='\s+'
            ).values.flatten()
        vs = pd.read_csv(nm + 'vs', skiprows=i, nrows=1, header=None, sep='\s+'
            ).values.flatten()
    except:
        crust1url = 'http://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz'
        print(
            'You need to download (and extract) the Crust1.0 model'
            + ' from \n\t{} \nand save to \n\t{}'.format(crust1url, nm[:-7])
        )
        return
    # vp = pd.read_csv(nm + 'vp', skiprows=i, nrows=1, header=None, sep='\s+'
    #     ).values.flatten()
    # rho = pd.read_csv(nm + 'rho', skiprows=i, nrows=1, header=None, sep='\s+'
    #     ).values.flatten()

    thickness = -np.diff(cb)
    ib = 0
    m_t = [0]
    m_vs = []
    for t in thickness:
        if t > 0:
            m_vs += [vs[ib]]
            m_t += [t]
        ib += 1
    m_vs += [vs[ib]]#[m_vs[-1]]

    return m_t, m_vs


def get_vels_ShenRitzwoller2016(location):
    """
    To get a more accurate starting model for the crust in the US, load in
    data from Shen & Ritzwoller, 2016.  Note that this is the updated model
    of Shen et al., 2013 - the model used (at time of writing) for Moho depth
    and velocity contrast constraints.  (Note we used an older model for that
    as the download included explicit Moho depth, dVs and error information).

    Download is available from https://doi.org/10.17611/DP/EMCUS2016

    The downloaded file - 'US.2016.nc' - is a NetCDF file with coordinates
        'depth'         0 - 150 km, spacing 0.5 km
        'latitude'      20 - 50 °N, spacing 0.25°
        'longitude'     235 - 295 °E, spacing 0.25°
    and fields
        'vsv'           (depth, latitude, longitude)
        'vp'            (depth, latitude, longitude)
        'rho'           (depth, latitude, longitude)

    """
    lat, lon = location
    if lon < 0: # Convert to °E if in °W
        lon += 360

    nm = 'data/earth_models/US.2016.nc'
    try:
        ds = xr.open_dataset(nm)
    except:
        crusturl = 'https://doi.org/10.17611/DP/EMCUS2016'
        print(
            'You need to download the Shen & Ritzwoller (2016) model from'
            + ' IRIS EMC\n\t{} \nand save to \n\t{}'.format(crusturl, nm)
        )
        return

    i_lat = np.argmin(np.abs(ds.latitude.values - lat))
    i_lon = np.argmin(np.abs(ds.latitude.values - lon))

    thickness = np.hstack((np.array([0]), np.diff(ds.depth.values)))
    vsv = ds.vsv.values[:, i_lat, i_lon]

    return thickness, vsv
