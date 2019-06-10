""" Generate Earth models to work with surface_waves

Classes:
    EarthModel - Vs, Vp, density, layer thickness, boundary depths

Functions:
    download_velocity_model(save_name:str, server_name:str) -> str:
        - Download an IRIS EMC hosted velocity model as a .nc file
    load_velocity_model(save_name:str, vs_fieldname:str,
                        lat:float, lon:float) -> EarthModel:
        - Starting from a .nc file, calculate a full model (inc. Vp, rho)
    _make_layered_earth_model(depth:np.array, vs:np.array, tol_vs:float=0.1,
                              min_layer_thickness:int=2,
                              max_crustal_vs:float=4.) -> EarthModel:
        - Starting from Vs and Depth, make a layered EarthModel
    _make_layered_param_model(depth:np.array, param:np.array, tol:float,
                              min_layer_thickness:int) -> (np.array, np.array):
        - Starting from any parameter and depth, return arrays of layered depth
          and the median value of that parameter for each layer
    _find_all_splits(y:np.array, min_layers_in:int, tol:float,
                        splits=None, y_first_ind=None) -> list:
        - Find the best points to split input y (i.e. Vs) based on largest jumps
    _find_best_split(y:np.array, min_layers_in:int, tol:float) -> int:
        - Find the single best point to split input y (Vs) based on largest jump
    _moving_window(y:np.array, window_length:int) -> list:
        - Define moving windows (offset by 1 element) for y
    _get_new_layers(x:np.array, y:np.array, inds:list) -> (np.array, np.array):
        - Convert x and y to downsampled model, with new layers defined by inds
    _calculate_vp_rho(vs:np.array, depth:np.array,
                      max_crustal_vs:float=4.) -> (np.array, np.array,
                                                   np.array):
        - Calculate Vp and density from Vs and depth
    _Brocher05_scaling(vs: np.array) -> (np.array, np.array):
        - Scale from Vs to Vp, rho for crustal rocks.
    _Forte07_scaling(vs:np.array, depth:np.array) -> np.array:
        - Scale from Vs & depth to rho for mantle rocks.
    _model(earth_model:EarthModel, fieldname:str):
        - plot EarthModel field (i.e. 'vs', 'vp', 'rho')


"""


#import collections
import typing
import os
import itertools
import numpy as np
import urllib.request as urlrequest
import xarray as xr
import matplotlib.pyplot as plt

from util import matlab

# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class EarthModel(typing.NamedTuple):
    """ Layered Earth model (Vs, Vp, rho, layer thickness, boundary depths)

    Vs, Vp, rho, and thickness include values for the underlying half space.

    Fields:
        - vs: layer shear velocity (km/s)
        - vp: phase velocity (km/s)
        - rho: density (Mg/m^3)
        - thickness: layer thickness (km)
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


def plot_earth_model(earth_model:EarthModel, fieldname:str):
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

def download_velocity_model(save_name:str, server_name:str) -> str:
    """ Download an IRIS hosted Earth model (.nc file).

    This is based on fetch_velo_models.py from the VBR package (Holtzman
    and Havlin, 2019), written by Chris Havlin.

    Arguments:
        - save_name (str):
            Name for saving the model (in ./data/earth_models/)
        - server_name (str):
            Name that model is saved as on the IRIS EMC

    Returns:
        - filename (str):
            Saved location of downloaded file.

    Saves:
        - ./data/earth_models/[save_name].nc:
            File from IRIS server
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

def load_velocity_model(save_name:str, vs_fieldname:str,
                        lat:float, lon:float) -> EarthModel:
    """ Load in velocity model (from .nc file) for a given (lat, lon).

    This function opens the .nc file, finds the closest point in latitude
    and longitude to the input location (lat, lon arguments), and pulls out
    the Vs structure there.  It then converts the depth/Vs model into a
    model with coarser layers (using _make_layered_model).

    Arguments:
        - save_name (str):
            Name of .nc file (inc. path from working dir)
        - vs_fieldname (str):
            This is 'vs' if input model is isotropic, 'vsv' if anisotropic.
            Need to check on the IRIS EMC website to make sure of the name
            so will load in the correct field from the .nc file.
        - lat (float):
            Latitude of interest (+ve is north, -ve is south)
        - lon (float):
            Longitude of interest (+ve is east, can optionally use -ve as west)
            Note that the convention for the IRIS EMC seems to be all positive.

    Returns:
        - starting_model (EarthModel):
            The EarthModel is returned as a coarse layers version of the input
            Vs model (using _make_layered_model).

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
    print(i_lat, i_lon)
    model_depth_orig = ds['depth'].values

    # Interpolate to make sure all velocity models are evenly spaced in depth
    model_depth = np.arange(0, model_depth_orig[-1],
                            np.min(np.append(np.diff(model_depth_orig), 1.)))
    model_vs = np.interp(model_depth, model_depth_orig, model_vs)

    # Fit the Vs profile with a smaller number of layers
    tol_vs = 0.1 # dVs (km/s) that should be noticeably different
    min_layer_thickness = 2 # (km)
    max_crustal_vs = 4. # Vs assumed to mark transition from crust to mantle
    starting_model = _make_layered_earth_model(model_depth, model_vs,
                                               tol_vs, min_layer_thickness,
                                               max_crustal_vs)

    return starting_model

def _make_layered_earth_model(depth:np.array, vs:np.array, tol_vs:float=0.1,
                              min_layer_thickness:int=2,
                              max_crustal_vs:float=4.) -> EarthModel:
    """ Generate a coarser Earth Model from input depth and Vs.

    From a Vs, depth profile, this simplifies the model into fewer
    coarse layers (using_find_all_splits()), and then calculates
    the expected Vp and density structure (using _calculate_vp_rho()).

    Arguments:
        - depth (np.array):
            Depth (km) of model to be divided into coarser layers.
        - vs (np.array):
            Vs profile (km/s) to be divided into coarser layers.
        - tol_vs (float):
            The maximum contrast in Vs (km/s) acceptable within each layer.
        - min_layer_thickness (int):
            The minimum thickness (km) of each output layer.

    Returns:
        - starting_model (EarthModel):
            Layered model, including Vs, Vp, and density.
    """
    layered_depth, layered_vs = _make_layered_param_model(
        depth, vs, tol_vs, min_layer_thickness)

    # Convert from Vs and depth to Vp, rho, and thickness
    vp, rho, thickness = _calculate_vp_rho(layered_vs, layered_depth,
                                          max_crustal_vs)

    starting_model = EarthModel(
        vs = layered_vs,
        vp = vp,
        rho = rho,
        depth = layered_depth,
        thickness = thickness,
    )

    return starting_model

def _make_layered_param_model(depth:np.array, param:np.array, tol:float,
                              min_layer_thickness:int) -> (np.array, np.array):
    """ Generate a coarser Earth Model from input depth and Vs.

    From a Vs, depth profile, this simplifies the model into fewer
    coarse layers (using_find_all_splits()), and then calculates
    the expected Vp and density structure (using _calculate_vp_rho()).

    Arguments:
        - depth (np.array):
            Depth (km) of model to be divided into coarser layers.
            Assumed to be uniformly sampled!
        - param (np.array):
            Parameter of model to be divided into coarser layers.
        - tol (float):
            The maximum contrast in the parameter acceptable within each layer.
        - min_layer_thickness (int):
            The minimum thickness (km) of each output layer.

    Returns:
        layered_depth, layered_param (np.arrays):
            The new maximum depth and median value of param for each layer
            (output of _get_new_layers()).
    """

    min_layers_in = np.where(depth >= min_layer_thickness)[0][0]
    print(min_layer_thickness, min_layers_in)
    # Find the indices for the uppermost possible layer as a flattened np array
    split_inds = _find_all_splits(param, min_layers_in, tol)
    return _get_new_layers(depth, param, sorted(split_inds))

def _find_all_splits(y:np.array, min_layers_in:int, tol:float,
                    splits=None, y_first_ind=None) -> list:
    """ Recursively find the best places to split a vector, y, into layers.

    Find all splits in the input, 'y', that result in coarser layers that are
    made up of at least 'min_layers_in' original layers.  The coarser layers
    must also have contrasts between them of at least 'tol'.  For the initial
    call, set splits and y_first_ind to be empty.

    This function searches first for the largest contrasts in y and puts splits
    in there, then works down to smaller and smaller contrasts in y until the
    threshold values (min_layers_in, tol) are met.

    Arguments:
        - y (np.array):
            Array to be divided into coarser layers.
        - min_layers_in (int):
            When downsampling the original array, y, this is the minimum number
            of original layers that are to be included in each layer.  Note that
            this is number of layers, not some variable with units!
        - tol (float):
            This is the minimum contrast in y that is acceptable within each
            layer.  If y varies less than 2 * tol, no new splits will be made.
        - splits (list):
            When the function is called recursively, this is the list of split
            points on which it has identified the biggest changes.
        - y_first_ind (int):
            When the function is called recursively, this is the index in y
            that the first point in the subset of y being searched over
            corresponds to such that the output splits correspond to indices
            in the original y.

    Returns:
        - splits (list):
            List of the indices of the major jumps in y, for use in making
            a coarse version of y.  Note that this list is unsorted - instead,
            the first index corresponds to the first identified (largest) jump.
            Note that the order does not strictly correspond to the size of the
            change in y identified, as the code will search for earlier changes
            before it searches for later changes.

    """

    # This is the recommended way of starting with an empty argument, as
    # default values are not re-evaluated on multiple calls of the function,
    # so if define it as something mutable (e.g. []), then future calls to
    # the function use the default values with any subsequent changes applied:
    # https://docs.python.org/3/reference/compound_stmts.html#function-definitions
    if splits is None:
        splits = []
        y_first_ind = 0

    split = _find_best_split(y, min_layers_in, tol)
    if not split:
        # No splits were found for this y given the search thresholds
        return splits

    splits += [split + y_first_ind]

    # Find additional splits by looking at the subsets of y before and after
    # the identified best split.
    if split >= min_layers_in * 2:
        # If there are possible splits earlier in y than the identified split
        y_subset = y[:split]
        splits = _find_all_splits(y_subset, min_layers_in, tol,
                                splits, y_first_ind)
    if y.size - min_layers_in*2 >= split:
        # If there are possible splits later in y than the identified split
        y_first_ind += split
        y_subset = y[split:]
        splits = _find_all_splits(y_subset, min_layers_in, tol,
                                splits, y_first_ind)

    return splits

def _find_best_split(y:np.array, min_layers_in:int, tol:float) -> int:
    """ Find the best place to split y.

    This is a slightly more complicated version of just finding the maximum
    difference in y.  It first identifies the window (min_layers_in large) that
    has the largest change in y, as this should be more significant than a
    single, slightly larger change in y surrounded by relatively constant y.
    It then finds the index of the largest change in y within that window.

    It further checks that this best split location conforms to the threshold
    requirements (min_layers_in and tol).  No splits are returned if the
    resulting layers would not conform.  If the identified split would not
    conform to these requirements, the split is adjusted until it does fit.

    Arguments:
        - y (np.array):
            Array to be divided into coarser layers.
        - min_layers_in (int):
            When downsampling the original array, y, this is the minimum number
            of original layers that are to be included in each layer.  Note that
            this is number of layers, not some variable with units!
        - tol (float):
            This is the minimum contrast in y that is acceptable within each
            layer.  If y varies less than 2 * tol, no new splits will be made.

    Returns:
        - split (int):
            Best index in y for capturing large changes in y.
    """
    if (2 * tol > np.ptp(y)) | (2 * min_layers_in > y.size):
        return False

    y_windowed = _moving_window(y, min_layers_in)
    split_window = np.argmax(
        np.fromiter(map(np.ptp, y_windowed), dtype=np.float)
    )
    split = (split_window
            + np.argmax(np.diff(y[split_window : split_window+min_layers_in]))
            + 1)

    # Make sure that split point gives large enough change in resultant layers
    while tol > np.ptp(y[:split]):
        split += 1
    while tol > np.ptp(y[split:]):
        split -= 1
    # Make sure that split point leaves the resultant layers thick enough
    split = max(split, min_layers_in)
    split = min(split, y.size - (min_layers_in))

    return split

def _moving_window(y:np.array, window_length:int) -> list:
    """ Return a list of the windowed inputs.

    For the array y, return windows of length window_length.  The moving
    window shifts by 1 index between windows.  The windowed y is returned
    as a list of tuples, where each tuple is an adjacent window.

    Itertools help from https://realpython.com/python-itertools/

    Arguments:
        - y (np.array):
            Array to be divided into windows.
        - window_length (int):
            Number of elements in each window

    Returns:
        - List of tuples, where each tuple contains the values of y that are
          in each window.  The list therefore has (y.size - window_length)
          elements in it.

    """
    iters = []
    for i in range(window_length):
        iters += [iter(y[i::])]
    return list(zip(*iters))

def _get_new_layers(x:np.array, y:np.array, inds:list) -> (np.array, np.array):
    """ Convert x, y arrays to downsampled arrays (coarser layers).

    Arguments:
        - x (np.array):
            Vector assumed to be monotonically increasing.
        - y (np.array):
            Vector of the same size as x.
        - inds (list):
            List of indices in the x vector to be used for downsampling.  Each
            index is of the shallowest point in the new layer, but the first
            index is not included.

    Returns:
        - layered_x (np.array):
            Maximum x in each new layer, of the same length as inds.
        - layered_y (np.array):
            Median y in each new layer, including for the half-space (layer
            starting at inds[-1] assumed to extend to infinity).  Layered_y
            will therefore have one more element than layered_x.
    """

    layered_x = x[np.array(inds) - 1]

    inds = [0] + inds + [x.size]
    layers = []
    for i in range(len(inds) - 1):
        layers += [y[inds[i]:inds[i + 1]]]
    layered_y = np.fromiter(map(np.median, layers), np.float)

    return layered_x, layered_y

def _calculate_vp_rho(vs:np.array, depth:np.array,
                     max_crustal_vs:float=4.) -> (np.array, np.array, np.array):
    """ Convert from Vs and depth to Vp, rho, and thickness.

    Different scalings are required for the crust and the mantle, so
    need to define the Moho - here, as where Vs goes above max_crustal_vs
    (default 4 km/s).

    Arguments:
        - vs (np.array):
            Vector of Vs (km/s)
        - depth (np.array):
            Vector of depth (km), one element shorter than vs.
            Note that this is the depth at the base of the layer (not including
            the half space).
        - max_crustal_vs (int):
            Scaling for Vp and rho is different for crust and mantle -
            define the cutoff between crust and mantle as when input Vs gets
            above this value (km/s).

    Returns:
        - vp (np.array):
            Vector of vp (km/s), same size as vs, calculated from Vs and depth.
        - rho (np.array):
            Vector of density (g.cm^-3), same size as vs, from Vs and depth.
        - thickness (np.array):
            Vector of layer thicknesses (km), same size as vs, including some
            arbitrary half_space_thickness appended onto the end.
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
    vp[crust_inds], rho[crust_inds] = _Brocher05_scaling(vs[crust_inds])

    # Any deeper values (sub-Moho) are mantle, so have different scaling.
    if mantle_inds.size:
        vp[mantle_inds] = vs[mantle_inds] * 1.75 # Fix mantle Vp/Vs ratio
        rho[mantle_inds] = _Forte07_scaling(vs[mantle_inds],
                                           av_layer_depth[mantle_inds])

    return vp, rho, thickness

def _Brocher05_scaling(vs: np.array) -> (np.array, np.array):
    """ Scale from Vs to Vp and rho - appropriate for the crust only.

    These equations are from Brocher et al., 2005, BSSA
    (doi: 10.1785/0120050077).

    Arguments:
        - vs (np.array):
            Vector of Vs (km/s)

    Returns:
        - vp (np.array):
            Vector of calculated Vp (km/s), same size as vs.
        - rho (np.array):
            Vector of calculated density (g.cm^-3), same size as vs.
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

def _Forte07_scaling(vs:np.array, depth:np.array) -> np.array:
    """ Scale from Vs to rho - appropriate for the mantle only.

    These equations are from Forte et al., 2007, GRL
    (doi:10.1029/2006GL027895)

    They relates d(ln(rho))/d(ln(vs)) to z and comes from fitting
    linear/quadratics to values picked from their Fig. 1b.

    We will assume an oceanic setting (suitable for orogenic zones too)
    although they give a different scaling for shields if necessary.

    Note that depth here is the depth to the MIDDLE of the layer, not the base.

    Arguments:
        - vs (np.array):
            Vector of Vs (km/s)
        - depth (np.array):
            Vector of depth (km), same size as vs.  Note that this is the depth
            at which the conversion calculation is done, so should be the
            average depth of a layer.

    Returns:
        - rho (np.array):
            Vector of calculated density (g.cm^-3), same size as vs.

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


def _fill_with_PREM():
    """ If need to fill in the gaps, then do so with PREM.

    prem = pd.read_csv('mineos_inputs/PREM500.csv')
        columns of prem are: radius,density,Vpv,Vsv,Q-kappa,Q-mu,Vph,Vsh,eta

    http://ds.iris.edu/ds/products/emc-prem/
    """
    pass
