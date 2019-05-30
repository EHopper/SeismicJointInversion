""" Generate synthetic surface waves.

This is an all-Python code to calculate surface wave phase velocities
from a given starting velocity model.  Ultimately, this should perhaps
be replaced with MINEOS.

Classes:
    PhaseVelocity - period, phase velocity, and error for surface waves

Functions:
    synthesise_surface_wave(model, swd_in) -> PhaseVelocity:
        - calculate surface wave phase velocities given model and period
    _Rayleigh_phase_velocity_in_half_space(vp, vs) -> float:
        - given Vp and Vs of a half space, calculate phase velocity
    _min_value_secular_function(omega, k_lims, n_ksteps,
                                thick, rho, vp, vs, mu) -> float:
        - search over k_lims to find the minimum value of _secular()
    _secular(k, om, thick, mu, rho, vp, vs) -> float:
        - calculate Green's function for velocity structure
"""


#import collections
import typing
import os
import numpy as np
import urllib.request as urlrequest
import xarray as xr

import matlab

# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class EarthModel(typing.NamedTuple):
    """ Layered Earth model (Vs, Vp, rho, layer thickness, boundary depths)

    Vs, Vp, and rho include values for the underlying half space.

    Fields:
        - vs: layer shear velocity (km/s)
        - vp: phase velocity (km/s)
        - rho: density (Mg/m^3)
        - thickness: layer thickness (km), half space not included
        - depth: depth of base of layer (km), half space not included
    """
    vs: np.array
    vp: np.array
    rho: np.array
    thickness: np.array
    depth: np.array

# =============================================================================
#       Initialise and update Earth Model.
# =============================================================================

def download_velocity_model(save_name, server_name):
    """ Import an IRIS hosted Earth model and extract model for a lat/lon point

    This is based on fetch_velo_models.py from the VBR package (Holtzman
    and Havlin, 2019), written by Chris Havlin.
    """

    # EMC website
    url_base = 'https://ds.iris.edu/files/products/emc/emc-files/'

    full_url = url_base + server_name
    filename = './data/earth_models/' + save_name + '.nc'
    if os.path.isfile(filename):
        print(save_name + ' already downloaded.')
    else:
        print('attempting to fetch ' + full_url)
        urlrequest.urlretrieve(full_url, filename)
        print('file downloaded as ' + filename)

    return filename

def load_velocity_model(save_name, vs_fieldname) -> EarthModel:
    """ Load in relevant parts of .nc files

    """

    ds = xr.open_dataset(save_name)
    earth_model = EarthModel(
        vs = ds[]
    )
    save_dict={'Latitude':ds[iris_files[fi]['lat_field']].values,
                 'Longitude':ds[iris_files[fi]['lon_field']].values,
                   'Depth':ds[iris_files[fi]['z_field']].values,
                   'Vs':ds[iris_files[fi]['Vs_field']].values.transpose(1,2,0)}

    return
