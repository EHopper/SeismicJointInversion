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
import matplotlib.pyplot as plt

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

def load_velocity_model(save_name, vs_fieldname, lat, lon) -> EarthModel:
    """ Load in relevant parts of .nc files

    """

    ds = xr.open_dataset(save_name)
    # Find the location in the model closest to the input lat, lon
    i_lat = np.argmin(np.abs(ds['latitude'] - lat))
    # For longitude, make sure everything is 0-360 degree range
    if np.any(ds['longitude'] < 0):
        ds['longitude'] += 360
    if lon < 0:
        lon += 360
    i_lon = np.argmin(np.abs(ds['longitude'] - lon))
    model_vs = ds[vs_fieldname].values[:, i_lat, i_lon]
    model_depth = ds['depth'].values

    # Fit the Vs profile with a smaller number of layers
    tol_vs = 0.1 # dVs (km/s) that should be noticeably different
    min_layer_thickness = 5 # (km)
    # Find the indices for the uppermost possible layer as a flattened np array
    d_inds = np.flatnonzero(min_layer_thickness+model_depth[0] >= model_depth)
    layered_vs, layered_depth = convert_to_coarser_layers(
        model_vs, model_depth, d_inds, tol_vs, min_layer_thickness)

    # Convert from Vs and depth to Vp, rho, and thickness
    vs = np.array(layered_vs)
    depth = np.array(layered_depth)
    max_crustal_vs = 4. # Vs assumed to mark transition from crust to mantle
    vp, rho, thickness = calculate_vp_rho(vs, depth, max_crustal_vs)

    starting_model = EarthModel(
        vs = vs,
        vp = vp,
        rho = rho,
        depth = depth,
        thickness = thickness,
    )

    return starting_model

def plot_earth_model(earth_model, fieldname):
    """ Convert from layered model into something to plot. """
    field = getattr(earth_model, fieldname)
    depth = earth_model.depth

    plot_field = np.tile(field, (2, 1)).T.reshape(2 * field.size)
    plot_depth = np.append(0., np.tile(depth, (2, 1)).T.reshape(2 * depth.size))
    plot_depth = np.append(plot_depth, depth[-1] + earth_model.thickness[-1])

    plt.plot(plot_field, plot_depth)
    plt.xlabel(fieldname)
    plt.ylabel('Depth (km)')
    ax = plt.gca().set_ylim(plot_depth[[-1, 0]])
    plt.show()

def convert_to_coarser_layers(
        input_vs:np.array, input_depth:np.array, d_inds:np.array,
        tol_vs=0.1, min_layer_thickness=5,
        layered_vs=None, layered_depth=None) -> (list, list):
    """ Redefine the velocity model in terms of layers.

    The surface wave forward model runs slower and slower the more layers
    there are, so simplify the velocity model to speed things up.

    The new layered model requires layers to be at least min_layer_thickness km
    thick (5 km default).  It allows thicker layers than this, as long as the
    Vs change within the layer stays within tol_vs (0.1 km/s default).

    The function works by guessing a depth range of a possible layer (d_inds).
    If this layer has more than tol_s velocity contrast in it, the guessed layer
    is appended to the output layered_vs, layered_depth.  The function is then
    called again guessing the adjacent deeper layer depth range.
    If there is less velocity contrast than tol_s, the guessed depth range is
    increased, and the function is called again using this expanded d_inds.
    If d_inds reaches the maximum input depth, the median velocity of the
    half-space is appended to layered_vs, and layered_vs and layered_depth are
    returned.

    Note that layered_vs will be 1 longer than layered_depth, as it includes
    a velocity value for the half-space.  These are both lists.

    args section here!
    returns section afterwards!
    """

    # This is the recommended way of starting with an empty argument, as
    # default values are not re-evaluated on multiple calls of the function,
    # so if define it as something mutable (e.g. []), then future calls to
    # the function use the default values with any subsequent changes applied:
    # https://docs.python.org/3/reference/compound_stmts.html#function-definitions
    if layered_vs is None:
        layered_vs = []
        layered_depth = []

    # If it has reached the bottom of the input model, add on the half space
    # and return the layered model.
    if d_inds[-1] >= input_depth.size - 1:
        d_inds = np.arange(d_inds[0], input_depth.size)
        layered_vs += [np.round(np.median(input_vs[d_inds]), 3)]
        return layered_vs, layered_depth

    if np.ptp(input_vs[d_inds]) > tol_vs:
        # If dVs across depth range is above tol_vs, record the new layer.
        deepest_point = input_depth[d_inds[-1]]
        layered_depth += [deepest_point]
        layered_vs += [np.round(np.median(input_vs[d_inds]), 3)]
        # Reset d_inds to span new minimum thickness layer, and
        # recursively call this function.
        d_inds = np.flatnonzero(
            (min_layer_thickness >= input_depth-deepest_point)
            & (input_depth > deepest_point)
        )
        return convert_to_coarser_layers(
            input_vs, input_depth, d_inds, tol_vs, min_layer_thickness,
            layered_vs, layered_depth)

    else:
        # If dVs across depth range is below tol_vs, increase the depth range
        # and try re-calling the function.
        d_inds = np.append(d_inds, d_inds[-1] + 1)
        return convert_to_coarser_layers(
            input_vs, input_depth, d_inds, tol_vs, min_layer_thickness,
            layered_vs, layered_depth)

def calculate_vp_rho(vs:np.array, depth:np.array, max_crustal_vs=4.):
    """ Convert from Vs and depth to Vp, rho, and thickness.

    Different scalings are required for the crust and the mantle, so
    need to define the Moho - here, as where Vs goes above max_crustal_vs
    (default 4 km/s).
    """

    # Initialise Vp, rho
    vp = np.zeros_like(vs)
    rho = np.zeros_like(vs)

    # Calculate the thickness and average depth of the layers
    half_space_thickness = 100 # arbitrary
    layer_tops = np.append(0., depth)
    if not depth.size: # half-space only
        thickness = half_space_thickness
    else:
        thickness = np.append(np.diff(layer_tops), half_space_thickness)
    av_layer_depth = layer_tops + thickness/2

    # Assume that Vs under max_crustal_vs (default 4 km/s) indicates crust.
    mantle_inds, = np.where(vs > max_crustal_vs)
    if mantle_inds.size:  # if this array isn't empty
        Moho_ind = mantle_inds[0]
    else:
        Moho_ind = vs.size

    crust_inds = np.arange(Moho_ind)

    # Scale from Vs to Vp and rho in the crust
    vp[crust_inds], rho[crust_inds] = Brocher05_scaling(vs[crust_inds])

    # Any deeper values (sub-Moho) are mantle, so have different scaling.
    if mantle_inds.size:
        vp[mantle_inds] = vs[mantle_inds] * 1.75 # Fix mantle Vp/Vs ratio
        rho[mantle_inds] = Forte07_scaling(vs[mantle_inds],
                                           av_layer_depth[mantle_inds])

    return np.round(vp, 5), np.round(rho, 5), thickness

def Brocher05_scaling(vs):
    """ Scale from Vs to Vp and rho - appropriate for the crust only.

    These equations are from Brocher et al., 2005, BSSA
    (doi: 10.1785/0120050077).
    """

    vp = (0.9409
          + 2.0947 * vs
          - 0.8206 * vs**2
          + 0.2683 * vs**3
          - 0.0251 * vs**4)
    rho = (1.6612 * vp
           - 0.4721 * vp**2
           + 0.0671 * vp**3
           - 0.0043 * vp**4
           + 0.000106 * vp**5)

    return vp, rho

def Forte07_scaling(vs, depth):
    """ Scale from Vs to rho - appropriate for the mantle only.

    These equations are from Forte et al., 2007, GRL
    (doi:10.1029/2006GL027895)

    They relates d(ln(rho))/d(ln(vs)) to z and comes from fitting
    linear/quadratics to values picked from their Fig. 1b.

    We will assume an oceanic setting (suitable for orogenic zones too)
    although they give a different scaling for shields if necessary.

    Note that depth here is the depth to the MIDDLE of the layer, not the base.
    """

    rho_scaling = np.zeros(vs.size)
    del_rho = np.zeros(vs.size)
    i_uppermost_mantle, = np.where(135 > depth)
    i_upper_mantle, = np.where(depth > 135)
    # Fitting the linear part of the graph
    rho_scaling[i_uppermost_mantle] = (
        2.4e-4 * depth[i_uppermost_mantle]
        + 5.6e-2)
    # Fitting the quadratic part of the graph
    rho_scaling[i_upper_mantle] = (
        2.21e-6 * depth[i_upper_mantle]**2
        - 6.6e-4 * depth[i_upper_mantle]
        + 0.075)
    # For shields, there is a linear relationship only:
    #    rho_scaling = 3.3e-4 * depth - 3.3e-2

    # rho_scaling is d(ln(rho))/d(ln(vs)), i.e. d(rho)/rho / d(vs)/vs
    # We'll take as our reference value the 120km value of AK135
    ref_vs = 4.5
    ref_rho = 3.4268
    del_rho = (rho_scaling  # d(ln(rho))/d(ln(vs))
               * (vs-ref_vs) / ref_vs # d(vs)/vs
               * ref_rho)

    return ref_rho + del_rho
