""" Inversion from phase velocities to velocity model.

This is based on MATLAB code from Josh Russell and Natalie Accardo (2019),
and formulated after Geophysical Data Analysis: Discrete Inverse Theory.
(DOI: 10.1016/B978-0-12-397160-9.00003-5) by Bill Menke (2012).

"""

#import collections
import typing
import numpy as np
import pandas as pd

from util import matlab
from util import define_earth_model
from util import surface_waves


# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class ModelLayerIndices(typing.NamedTuple):
    """ Parameters used for smoothing.

    Fields:
        upper_crust:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for upper crust.
        lower_crust:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for lower crust.
        lithospheric_mantle:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for lithospheric mantle.
        asthenosphere:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for asthenosphere.

        discontinuities:
            - (n_different_layers + 1, ) np.array
            - Depth indices of model discontinuties, including the surface
              and the base of the model.
            - i.e. [0, mid-crustal boundary, Moho, LAB, base of mode]

        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - Depth vector for the whole model space.
        layer_names:
            - list
            - All layer names in order of increasing depth.

    """

    upper_crust: np.array
    lower_crust: np.array
    lithospheric_mantle: np.array
    asthenosphere: np.array
    discontinuties: np.array
    depth: np.array
    layer_names: list = ['upper_crust', 'lower_crust',
                         'lithospheric_mantle', 'asthenosphere']

class ModelLayerValues(typing.NamedTuple):
    """ Values that are layer-specific.

    All fields should be an (n_values_in_layer x n_unique_vals_by_model_param)
    array, where
        n_values_in_layer:
            Can be 1 or ModelLayerIndices.[field_name].size.
                i.e. constant within a layer, or specified for each depth point
                     in that layer.
        n_vals_by_model_param:
            Can be 1 or the number of model parameters (5).
                i.e. single value across all model parameters, or the value
                     is dependent on which model parameter it is being
                     applied to.


    Fields:
        upper_crust:
            - (n_values_in_layer, n_vals_by_model_param) np.array
        lower_crust:
            - (n_values_in_layer, n_vals_by_model_param) np.array
        lithospheric_mantle:
            - (n_values_in_layer, n_vals_by_model_param) np.array
        asthenosphere:
            - (n_values_in_layer, n_vals_by_model_param) np.array

    """

    upper_crust: np.array
    lower_crust: np.array
    lithospheric_mantle: np.array
    asthenosphere: np.array


class LoveKernels(typing.NamedTuple):
    """ Frechet kernels for Love (toroidal) waves.

    Kernels for different periods are appended to each other.

    Fields:
        period:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    seconds
            - Period of Love wave of interest.
        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - Depth vector for kernel.
        vsv:
            - (n_depth_points * n_love_periods, ) np.array
            - Units:    assumes velocities in km/s
            - Vertically polarised S wave kernel.
        vsh:
            - (n_depth_points * n_love_periods, ) np.array
            - Units:    assumes velocities in km/s
            - Horizontally polarised S wave kernel.
        type: str = 'love'
            - Default shouldn't be changed
            - This is because was having issues with isinstance() for locally
              defined classes.


    """

    period: np.array
    depth: np.array
    vsv: np.array
    vsh: np.array
    type: str = 'love'

class RayleighKernels(typing.NamedTuple):
    """ Frechet kernels for Rayleigh (spheroidal) waves.

    Kernels for different periods are stacked on top of each other.

    Fields:
        period:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    seconds
            - Period of Rayleigh wave of interest.
        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - Depth vector for kernel.
        vsv:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Vertically polarised S wave kernel.
        vpv:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Vertically polarised P wave kernel.
        vph:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Horizontally polarised P wave kernel.
        eta:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Anellipticity (eta = F/(A-2L)) kernel.
        type: str = 'rayleigh'
            - Default shouldn't be changed.
            - This is because was having issues with isinstance() for locally
              defined classes.

    """

    period: np.array
    depth: np.array
    vsv: np.array
    vpv: np.array
    vph: np.array
    eta: np.array
    type: str = 'rayleigh'


# =============================================================================
#       Run the Damped Least Squares Inversion
# =============================================================================

def run_inversion(model:define_earth_model.EarthModel,
                  data:surface_waves.ObsPhaseVelocity,
                  n_iterations:int=5) -> (define_earth_model.EarthModel):
    """ Set the inversion running over some number of iterations.

    """


    for i in range(n_iterations):
        model = _inversion_iteration(model, data)

    return model

def _inversion_iteration(model, data):
    """ Run a single iteration of the least squares
    """

    # Build all of the inputs to the damped least squares
    rayleigh_kernels, love_kernels, depth = _calculate_Frechet_kernels(model)
    m0 = _build_model_vector(model, depth)
    G = _build_partial_derivatives_matrix(rayleigh_kernels, love_kernels)
    d = _build_data_misfit_matrix(data, model, m0, G)

    # Build all of the weighting functions for damped least squares
    W = _build_error_weighting_matrix(data)
    layer_indices = _set_layer_indices(m0)
    D_mat, d_vec, H_mat, h_vec = _build_weighting_matrices(data, layer_indices)

    # Perform inversion
    model_new = _damped_least_squares(m0, G, d, W, D_mat, d_vec, H_mat, h_vec)

    return _build_earth_model_from_vector(model_new)

def _calculate_Frechet_kernels(model:define_earth_model.EarthModel,
                               max_depth=None) -> (RayleighKernels, LoveKernels,
                                                   np.array):
    """ Load in pre-calculated, constant Frechet kernels from csv.

    Ultimately, this will be a call to MINEOS, updating Frechet kernels
    at every model iteration.

    Note that MINEOS works in SI units, so need to convert depth from
    metres to kilometres.

    Arguments
        model:
            - This will ultimately be used to make a card for MINEOS.  Now it's
              a placeholder for when I do things properly.
        max_depth:
            - float
            - Units:    kilometres
            - Depth below which to crop off the Frechet kernels
            - i.e. do not invert for Earth structure below this depth.

    Returns
        rayleigh_kernels:
            - RayleighKernels
        love_kernels: LoveKernels
            - LoveKernels
        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - Vector of depth (km) that is the same as in the loaded kernels.
    """

    if max_depth is None:
        max_depth = model.depth.max() + model.thickness[-1]

    file_name = './mineos_inputs/frechet_kernels.csv'
    frechet = pd.read_csv(file_name, ',\s+', engine='python')

    frechet['D_km'] = frechet['Depth'] / 1e3 # metres to km

    rayleigh_kernels = RayleighKernels(
        period = np.array(_crop(frechet['Period'], frechet['D_km'], max_depth)),
        depth = np.array(_crop(frechet['D_km'], frechet['D_km'], max_depth)),
        vsv = np.array(_crop(frechet['Vsv'], frechet['D_km'], max_depth)),
        vpv = np.array(_crop(frechet['Vpv'], frechet['D_km'], max_depth)),
        vph = np.array(_crop(frechet['Vph'], frechet['D_km'], max_depth)),
        eta = np.array(_crop(frechet['Eta'], frechet['D_km'], max_depth)),
    )


    love_kernels = LoveKernels(
        period = np.array(_crop(frechet['Period'], frechet['D_km'], max_depth)),
        depth = np.array(_crop(frechet['D_km'], frechet['D_km'], max_depth)),
        vsv = np.array(_crop(frechet['Vsv'], frechet['D_km'], max_depth)),
        vsh = np.array(_crop(frechet['Vsh'], frechet['D_km'], max_depth)),
    )

    depth = (rayleigh_kernels.depth[rayleigh_kernels.period
                                    == rayleigh_kernels.period[0]])

    return rayleigh_kernels, love_kernels, depth

def _crop(field, field_to_crop_by, max_value):
    """ Crop one field according to the maximum value of a second field """

    return field[field_to_crop_by < max_value]


def _build_model_vector(model:define_earth_model.EarthModel,
                        depth:np.array) -> (np.array):
    """ Make model into column vector n_depth_points*n_model_parameters x 1.

    Arguments:
        model:
            - define_earth_model.EarthModel
            - Units:    seismological, so km/s for velocities
            - Input Vs, Vp model.
        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - This is a non-MINEOS workaround to make sure the depth vectors
              are the same

    Returns:
        m0:
            - (n_model_points, 1) np.array
            - Units:    seismological, so km/s for velocities,
                        dimensionless for eta
            - Vsv, Vsh, Vpv, Vph, eta model stacked on top of each other.
            - N.B. n_model_points = n_depth_points * n_model_parameters
    """

    # Find depth vector from kernels and interpolate model over it
    depth = _make_2D(depth)

    vsv = _interpolate_earth_model_parameter(depth, model, 'vs')
    vsh = _interpolate_earth_model_parameter(depth, model, 'vs')
    vpv = _interpolate_earth_model_parameter(depth, model, 'vp')
    vph = _interpolate_earth_model_parameter(depth, model, 'vp')
    eta = np.ones_like(depth)

    return np.vstack((vsv, vsh, vpv, vph, eta))

def _make_2D(x:np.array) -> np.array:
    """ Turn a 1D numpy array into a column vector. """

    return x[np.newaxis].T

def _interpolate_earth_model_parameter(new_depth:np.array,
                                       model:define_earth_model.EarthModel,
                                       field_str:str) -> np.array:
    """ Convert from EarthModel layout and interpolate.

    Arguments:
        new_depth:
            - (n_depth_points, 1) np.array
            - Units:    kilometres
            - Depth vector over which to interpolate EarthModel parameter.
            - Shape of this vector, i.e. a column, is very important!
              np.interp will generate vectors of the same shape.
        model:
            - define_earth_model.EarthModel
            - Units:    seismological (e.g. km not m)
            - Input EarthModel.
        field_str:
            - str
            - Name of the parameter (field in model) to be interpolated.

    Returns:
        new_field:
            - (n_depth_points, 1) np.array
            - Units:    as units of field in input model.
            - Vector of model parameter interpolated onto new_depth and ready
              to be vertically stacked into m0.


    """

    depth, field = define_earth_model._convert_earth_model(model, field_str)
    new_field = np.interp(new_depth, depth, field)

    return new_field


def _build_partial_derivatives_matrix(rayleigh, love):
    """ Make partial derivative matrix, G, by stacking the Frechet kernels.

    Build the G matrix - this is a n_Love_periods+n_Rayleigh_periods by
    n_depth_points*5 (SV, SH, PV, PH, ETA) matrix.  This is filled in by
    the frechet kernels for each period - first the vsv and vsh T frechet
    kernels (sensitivities from Love) rows, then the vsv, (vsh set to 0), vpv,
    vph, eta S frechet kernels (sensitivities from Rayleigh) rows:

    G matrix:
              T_Vsv_p1  T_Vsh_p1      0         0         0
              T_Vsv_p2  T_Vsh_p2      0         0         0
        [     S_Vsv_p1     0      S_Vpv_p1  S_Vph_p1  S_eta_p1    ]
              S_Vsv_p2     0      S_Vpv_p2  S_Vph_p2  S_eta_p2
              S_Vsv_p2     0      S_Vpv_p2  S_Vph_p2  S_eta_p2

    where, e.g. T_Vsv_p1 is the Frechet kernel for Toroidal Vsv sensitivity
    for the first (Love) period. Frechet kernels are depth dependent, so each
    entry in the matrix above (including the 0) is actually a row vector
    n_depth_points long.

    Arguments:
        rayleigh:
            - RayleighKernels
        love:
            - LoveKernels

    Returns:
        G:
            - (n_Love_periods + n_Rayleigh_periods, n_model_points) np.array
            - Units:    assumes velocities in km/s
            - Partial derivative matrix is of the format described earlier in
              the docstring, stacking the Love and Rayleigh kernels vertically
              on top of each other with rows corresponding to different periods.
            - For now, I have set this to onlt the Rayleigh wave kernels -
              part of the non-MINEOS workaround.

    """

    periods = np.unique(rayleigh.period)

    G_Rayleigh = _hstack_frechet_kernels(rayleigh, periods[0])
    G_Love = _hstack_frechet_kernels(love, periods[0])
    for i_p in range(1,len(periods)):
        G_Rayleigh = np.vstack((G_Rayleigh,
                               _hstack_frechet_kernels(rayleigh, periods[i_p])))
        G_Love = np.vstack((G_Love,
                            _hstack_frechet_kernels(love, periods[i_p])))

    # surface_waves code only calculates Rayleigh phv

    return G_Rayleigh #np.vstack((G_Love, G_Rayleigh))


def _hstack_frechet_kernels(kernel, period:float):
    """ Append all of the relevent Frechet kernels into a row of the G matrix.

    Different parameters are of interest for Rayleigh (Vsv, Vpv, Vph, Eta)
    and Love (Vsv, Vsh) waves.

    Arguments:
        kernel:
            - RayleighKernels or LoveKernels
            - Frechet kernels across all periods
        period:
            - float
            - Units:    seconds
            - Period of interest.
            - Should match a period in kernel.period.

    Returns:
        Row vector:
            - (n_model_points, ) np.array
            - Units:     assumes velocities in km/s
            - Vector contains the Vsv, Vsh, Vpv, Vph, Eta kernels for the
              requested period.
            - Note that some of these are filled with zeros, depending on if
              the kernel is a Love or Rayleigh kernel.
    """

    # Note: if want to have kernels scaled for changes in velocity in SI
    #       units (i.e. m/s not km/s), multiply all kernels (including eta)
    #       by 1e3.
    vsv = kernel.vsv[kernel.period == period]

    if kernel.type == 'rayleigh':#isinstance(kernel, RayleighKernels):
        vsh = np.zeros_like(vsv)
        vpv = kernel.vpv[kernel.period == period]
        vph = kernel.vph[kernel.period == period]
        eta = kernel.eta[kernel.period == period]

    if kernel.type == 'love':#isinstance(kernel, LoveKernels):
        vsh = kernel.vsh[kernel.period == period]
        vpv = np.zeros_like(vsv)
        vph = np.zeros_like(vsv)
        eta = np.zeros_like(vsv)

    return np.hstack((vsv, vsh, vpv, vph, eta))


def _build_data_misfit_matrix(data, model, m0, G):
    """ Calculate data misfit.

    This is the difference between the observed and calculated phase velocities.
    To allow the inversion to invert for the model directly (as opposed to
    the model perturbation), we add this to the forward model prediction, Gm.

    Gm = d              Here, G = d(phase vel)/d(model),
                              m = change in model, m - m0
                              d = change in phase vel (misfit from observations)

    To allow us to use m = new model, such that we can later add linear
    constraints based on model parameters rather than model perturbation
    parameters,
        G (m - m0) = d          (as above)
        Gm - G*m0 = d
        Gm = d + G*m0

    d is still the misfit between observed (data.c) and predicted phase
    velocities (predictions.c), so we find the full RHS of this equation
    as (data.c - predictions.c) + G * m0 (aka forward_problem_predictions).

    See Russell et al., 2019 (10.1029/2018JB016598), eq. 7 to 8.

    Arguments:
        data:
            - surface_waves.ObsPhaseVelocity, data.c is (n_periods, ) np.array
            - Units:    data.c is in km/s
            - Observed phase velocity data
        model:
            - define_earth_model.EarthModel
            - Units:    seismology units, i.e. km/s for velocities
            - Earth model used to predict the phase velocity data.
        m0:
            - (n_model_points, 1) np.array
            - Units:    seismology units, i.e. km/s for velocities
            - Earth model ([Vsv; Vsh; Vpv; Vph; Eta]) in format for
              calculating the forward problem, G * m0.
        G:
            - (n_periods, n_model_points) np.array
            - Units:    assumes velocities in km/s
            - Matrix of partial derivatives of phase velocity wrt the
              Earth model (i.e. stacked Frechet kernels).

    Returns:
        data_misfit:
            - (n_periods, ) np.array
            - Units:    km/s
            - The misfit of the data to the predictions, altered to account
              for the inclusion of G * m0.


    """

    predictions = surface_waves.synthesise_surface_wave(model, data.period)
    forward_problem_predictions = np.matmul(G, m0).flatten()

    data_misfit = forward_problem_predictions + (data.c - predictions.c)

    return data_misfit

def _build_weighting_matrices(data:ObsPhaseVelocity, m0:np.array,
                              layers:ModelLayerIndices):
    """ Build all of the weighting matrices.

    Arguments:
        data:
            - surface_waves.ObsPhaseVelocity
            - Observed phase velocity data
            - data.c is (n_periods, ) np.array
        m0:
            - (n_model_points, 1) np.array
            - Units: SI units, i.e. m/s for velocities
            - Earth model ([Vsv; Vsh; Vpv; Vph; Eta]) in format for
              calculating the forward problem, G * m0.
        layers:
            - ModelLayerIndices

    Returns:
        data_errors:
            - (n_periods, n_periods) np.array
            - Units:    km/s
        roughness_mat:
            - (n_depth_points, n_model_points) np.array
        roughness_vec:
            - (n_depth_points, 1) np.array
        a_priori_mat:
            - (n_constraint_equations, n_model_points) np.array
        a_priori_vec:
            - (n_constraint_equations, 1) np.array


    """

    model_order = {'vsv': 0, 'vsh': 1, 'vpv': 2, 'vph': 3, 'eta': 4}


    # Minimise second derivative within layers
    damping = ModelLayerValues()
    roughness_mat, roughness_vec = _build_model_weighting_matrices(
        _build_smoothing_constraints, (layers), ['vsv', 'vsh'],
        model_order, layers, damping,
    )

    # Linear constraint equations
    # Damp towards starting model
    damping = ModelLayerValues()
    damp_to_m0_mat, damp_to_m0_vec = _build_model_weighting_matrices(
        _build_constraint_damp_to_m0, (n_depth_points, start_ind, m0),
        ['vsv', 'vsh', 'eta'], model_order, layers, damping,
    )
    # Damp towards starting Vp/Vs
    damping = ModelLayerValues()
    damp_vpvs_mat, damp_vpvs_vec = _build_constraint_damp_vpvs(
        layers, damping, model_order, m0,
    )
    # Damp towards input radial anisotropy
    xi_vals = ModelLayerValues()
    damping = ModelLayerValues()
    damp_isotropic_mat, damp_isotropic_vec = _build_constraint_damp_anisotropy(
        layers, damping, model_order, xi_vals,
    )
    # Damp towards starting model gradients in Vs (i.e. maintain layer Xi)
    damping = ModelLayerValues()
    damp_to_m0_grad_mat, damp_to_m0_grad_vec = _build_model_weighting_matrices(
        _build_constraint_damp_original_gradient,
        (n_depth_points, start_ind, m0),
        ['vsv', 'vsh'], model_order, layers, damping,
    )

    # Put all a priori constraints together
    a_priori_mat = np.vstack((
        damp_to_m0_mat,
        damp_vpvs_mat,
        damp_isotropic_mat,
        damp_to_m0_grad_mat,
    ))
    a_priori_vec = np.vstack((
        damp_to_m0_vec,
        damp_vpvs_vec,
        damp_isotropic_vec,
        damp_to_m0_grad_vec,
    ))


    return data_errors, roughness_mat, roughness_vec, a_priori_mat, a_priori_vec

def _build_layer_value_vector(values, layers):
    """

    Note: This assumes that the arrays in ModelLayerValues are a single column.
          That is, we are not inputting different values corresponding to
          different model parameters, but either a constant value for the layer
          or values that vary as a function of depth.

    Arguments:
        values:
            - ModelLayerValues
            - Assumes that the arrays here are a single column,
              (n_values_in_layer, 1) np.arrays
            - That is, for any given layer, we either have a single constant
              value, or a single value for each depth point in that layer.
        layers:
            - ModelLayerIndices
            - Description of the distinct model layers and their indices.

    Returns:
        value_vec:
            - (n_depth_points, 1) np.array
            - Single column vector of these values assembled by depth.
    """

    value_vec = np.zeros((layers.depth.size, 1))

    for layer_name in layers.layer_names:
        value_vec[getattr(layers, layer_name)] = getattr(values, layer_name)

    return value_vec

def _build_model_weighting_matrices(function_name, function_arguments,
                                    model_params, model_order,
                                    layers, damping):
    """ Helper function for constraints that are applied to one parameter.

    This function will run through the list of model_params (i.e. vsv, vsh, etc)
    that is input, calculate H and h using the function_name function, and then
    assemble these into H_all and h_all matrices of the correct shape.

    Arguments:
        function_name:
            - Callable function, with no () at the end, e.g. max
            - Note that this is not in quotes!  Not a string!
            - This is the function that will be used to actually set up the
              component parts of H and h.
        function_arguments:
            - tuple
            - All of the arguments used by the function_name function, which
              will be unpacked within that function.
        model_params:
            - list of strings, e.g. ['vsv', 'vsh']
            - The different model parameters that this constraint is going
              to be applied to.
            - Parameter names (list elements) must correspond to keys in
              model_order.
        model_order:
            - dict
            - This lays out the order of m0, i.e. which model parameters are
              stacked and in which order.
            - The keys are the names of the model parameters, e.g. 'vsv', 'vsh'
            - The values are the order, such that the index of the first
              element of that parameter in m0 is dict[key] * n_depth_points.
        layers:
            - ModelLayerIndices
        damping:
            - ModelLayerValues
            - This has the damping coefficients for this constraint.
            - Component arrays are (n_values_in_layer, n_vals_by_model_param),
              so can vary with depth across a layer and be different depending
              on the model_param.

    Returns:
        H_all:
            - (n_depth_points * len(model_params), n_model_points) np.array
            - Constraints matrix, with constraint equations for every depth
              point for each input model parameter (listed in model_params).
        h_all:
            - (n_depth_points * len(model_params), 1) np.array
            - Constraints vector, with solutions to H_all * m0 that we are
              damping towards.
    """

    n_depth_points = layers.depth.size
    n_model_points = n_depth_points * (max(model_order.values()) + 1)

    H_all = None

    for param in model_params:
        param_ind = model_order[param]
        start_ind = n_depth_points * param_ind
        H, h = function_name(function_arguments)
        H, h = _damp_constraints(H, h, layers, damping, param_ind)
        H = _pad_constraints(H, n_depth_points, n_model_points, start_ind)

        if not H_all:
            H_all = H
            h_all = h
        else:
            H_all = np.vstack((H_all, H))
            h_all = np.vstack((h_all, h))

    return H_all, h_all


def _damp_constraints(H, h, layers, damping, param_ind:int=0):
    """ Apply layer specific damping to H and h.

    This is done one model parameter at a time - i.e. assumes that H is only
    (n_constraints x n_depth_points), NOT n_model_points wide!!

    Damping parameters are set for individual layers.

    Arguments:
        H:
            - (n_constraint_equations, n_depth_points) np.array
            - Constraints matrix in H * m = h.
            - Note that this is only n_depth_points NOT n_model_points wide!
              It is assumed that H is input one model parameter at a time.
        h:
            - (n_constraint_equations, 1) np.array
            - Constraints vector in H * m = h.
        layers:
            - ModelLayerIndices
        damping:
            - ModelLayerValues
            - This has the damping coefficients for this constraint.
            - Component arrays are (n_values_in_layer, n_vals_by_model_param),
              so can vary with depth across a layer and be different depending
              on the model_param.
        param_ind:
            - int = 0
            - This tells us which column in the damping arrays to apply,
              i.e. which model parameter we are damping.
            - Default value set to 0 for when input damping array does not
              have model parameter-specific values.

        Returns:
            H:
                - (n_constraint_equations, n_depth_points) np.array
                - Damped constraints matrix in H * m = h.
            h:
                - (n_constraint_equations, 1) np.array
                - Damped constraints vector in H * m = h.
    """

    # A note on multiplication in numpy:
    #   numpy does everything element-wise unless you explicitly tell it not
    #   to, e.g. by using np.matmul() to do matrix multiplication.
    #   This means that it will broadcast arrays by repeating rows or columns
    #   so they are the same shape as the matrix they are being multiplied by.
    #   This broadcasting is ONLY done for matrices where one of the dimensions
    #   is equal to 1 (i.e. row (1, n) or column (n, 1) vectors) OR for 1D
    #   numpy arrays (i.e. (n,) arrays), which are treated as row vectors.
    #   If the arrays cannot be broadcast to the same shape, the * operator
    #   will return an error. Note that this element-wise behaviour means all
    #   numpy * operations are commutative!
    #   As such, for any c = a * b...
    #       a.shape | b.shape | d.shape      | Result
    # ______________|_________|______________|____________________________
    #       (n,)    | (m,)    | (n==m,)      |  [a_1*b_1, ..., a_n*b_m]
    #       (1, n)  | (m,)*   | (n==m, 1)    |  [[a_1*b_1, ..., a_n*b_m]]
    #       (n, 1)  | (m,)*   | (n, m)       |  [[a_1*b_1, ..., a_1*b_m],
    #               |         |              |   [a_n*b_1, ..., a_n*b_m]]
    #       (n,)* **| (m, 1)  | (n, m)       |  [[a_1*b_1, ..., a_n*b_1],
    #               |         |              |   [a_1*b_m, ..., a_n*b_m]]
    #       (n, m)  | (p,)*   | (n, m==p)    |  [[a_11*b_1, ..., a_1m*b_p],
    #               |         |              |   [a_n1*b_1, ..., a_nm*b_p]]
    #       (n, m)  | (p, 1)  | (n==p, m)    |  [[a_11*b_1, ..., a_1m*b_1],
    #               |         |              |   [a_n1*b_n, ..., a_nm*b_p]]
    #       (n, m)  | (p, q)  | (n==p, m==q) |  [[a_11*b_11, ..., a_1m*b_1p],
    #               |         |              |   [a_n1*b_p1, ..., a_nm*b_pq]]
    #
    #       * Same result for array.shape = (1, len) as (len,)
    #               i.e. row vector acts the same as 1D array
    #       ** This is to show commutative behaviour with example in row above.


    # Loop through all layers
    for layer_name in layers.layer_names:
        layer_inds = getattr(layers, layer_name)
        layer_damping = getattr(damping, layer_name)

        # (n_constraint_equations, n_depth_points_in_layer) slice of H
        #   * (n_depth_points_in_layer, 1) slice of layer_damping
        # As depth increases by column in H, need to transpose the slice of
        # layer_damping for the broadcasting to work properly.
        # For h, depth increases by row, so layer_damping slice is the
        # correct orientation.
        H[:, layer_inds] *= layer_damping[:, param_ind].T
        h[layer_inds] *= layer_damping[:, param_ind]

    return H, h

def _pad_constraints(H, n_model_points, start_ind):
    """ Pad input H for one model parameter so it is n_model_points wide.

    Arguments:
        H:
            - (n_constraint_equations, n_depth_points) np.array
            - Constraints matrix in H * m = h.
            - Note that this only have values for a single model parameter.
            - Damping should already have been applied.
        n_model_points:
            - int
            - Desired size of output H matrix - should be the same as len(m0).
        start_ind:
            - int
            - First index in m0 that the H matrix corresponds to.

    Returns:
        H_pad:
            - (n_constraint_equations, n_model_points) np.array
            - Input H matrix padded with columns of zeros such that it will
              be correctly multiplied with m0.

    """

    H_pad = np.zeros((H.shape[0], n_model_points))
    H_pad[:, start_ind:start_ind+n_depth_points] = H

    return H_pad

def _build_error_weighting_matrix(data):
    """ Build the error weighting matrix from data standard deviation.

    This is just a diagonal matrix of the inverse of standard deviation.

    Arguments:
        data:
            - surface_waves.ObsPhaseVelocity
            - Observed surface wave information.
            - Here, we are interested in the standard deviation of the data.
            - data.std is a (n_periods, ) np.array
            - If no standard deviation is given, we assume 0.15.

    Returns:
        error weighting matrix:
            - (n_periods, n_periods) np.array
            - This is equivalent to We in the syntax of Menke (2012)
              (and the docstring for _damped_least_squares())
              m = (G' * We * G  + ε^2 * Wm )^-1  (G' * We * d + ε^2 * Wm * <m>)
            - Diagonal matrix of 1 / standard deviation for weighted least
              squares inversion.
            - Measurements with smaller standard deviations will be weighted
              more strongly.
    """

    # If standard deviation of data is missing (= 0), fill in with guess
    obs_std = data.std
    obs_std[obs_std == 0] = 0.15


    return np.diag(1 / obs_std)

def _build_smoothing_constraints(arguments:tuple) -> (np.array, np.array):
    """ Build the matrices needed to minimise second derivative of the model.

    That is, for each layer, we want to minimise the second derivative
    (i.e. ([layer+1 val] - [layer val]) - ([layer val] - [layer-1 val]) )
    except for where we expect there to be discontinuties.  At expected
    discontinuities, set the roughness matrix to zero.

    Note that Josh says this can be better captured by a linear constraint
    preserving layer gradients (and therefore stopping the model from getting
    any rougher).  So perhaps this can just be removed.

    The smoothing parameters will be different for crust vs. upper mantle etc,
    and we want to allow larger jumps between layers, so should be zero
    at the boundaries between layers.

    Arguments:
        arguments:
            - tuple of (layers,)

        arguments[0], or layers:
            - ModelLayerIndices

    Returns:
        roughness_mat
            - (n_depth_points, n_depth_points) np.array
            - Roughness matrix, D, in D * m = d.
            - In Menke (2012)'s notation, this is D and Wm is D^T * D.
            - This is the matrix that we multiply by the model to find the
              roughness of the model (i.e. the second derivative of the model).
        - roughness_vec
            - (n_depth_points, 1) np.array
            - Roughness vector, d, in D * m = d.
            - This is the permitted roughness of the model.
            - Actual roughness is calculated as roughness_matrix * model, and
              by setting this equal to the permitted roughness in the damped
              least squares inversion, we will tend towards the solution of
              this permitted roughness (i.e. smoothness)
            - Because we want as smooth a model as possible, permitted
              roughness is set to zero always.

    """
    layers, = arguments
    n_depth_points = layers.depth.size

    # Make and edit the banded matrix (roughness equations)
    banded_matrix = _make_banded_matrix(n_depth_points, (1, -2, 1))
    # At all discontinuities (and the first and last layers of the model),
    # set the roughness_matrix to zero
    for d in layers.discontinuities:
        banded_matrix[d,:] *= 0

    roughness_vector = np.zeros((n_depth_points, 1))

    return roughness_matrix, roughness_vector


def _make_banded_matrix(matrix_size:int, central_values:tuple):
    """ Make a banded matrix with central_values along the diagonal band.

    Arguments:
        matrix_size:
            - int
            - Desired dimension of output banded matrix.
        central_values:
            - tuple (though I think a list would work the same)
            - Values to put along the diagonal - must be an odd number of
              elements in this!
            - Note that the middle element of central_values will be put
              in along the diagonal, and it will be neighboured by however
              many elements in central_values there are space for in the matrix.

    Returns:
        banded_matrix:
            - (matrix_size, matrix_size) np.array
            - Banded matrix, with central_values inserted along the diagonal
              band.
    """
    banded_matrix = np.zeros((matrix_size, matrix_size))

    diag_ind = len(central_values) // 2 # // is floor

    for n_row in range(diag_ind):
        vals = central_values[(diag_ind - n_row):]
        banded_matrix[n_row, :len(vals)] = vals

    for n_row in range(diag_ind, matrix_size-diag_ind):
        start_i = n_row - diag_ind
        vals = central_values
        banded_matrix[n_row, start_i:start_i + len(vals)] = vals

    for n_row in range(matrix_size-diag_ind, matrix_size):
        vals = central_values[:(diag_ind + matrix_size - n_row)]
        banded_matrix[n_row, -len(vals):] = vals


    return banded_matrix


def _build_constraint_damp_to_m0(arguments:tuple) -> (np.array, np.array):
    """ Damp towards starting model.

    In H * m = h, we want to end up with
                    m_val_new = m_val_old
    As such, H is a diagonal matrix of ones, and h is made up of the starting
    model values.

    Arguments:
        arguments:
            - tuple of (n_depth_points, start_ind, m0)

        arguments[0], or n_depth_points:
            - int
            - Number of depth points in the model.
        arguments[1], or start_ind:
            - int
            - Index at which the parameter of interest (e.g. Vsv, Vsh) block
              starts in the input model, m0.
        arguments[2], or m0:
            - (n_model_points, 1) np.array
            - Starting model, with different parameters stacked on top of
              one another.

    Returns:
        H:
            - (n_depth_points, n_depth_points) np.array
            - Constraints matrix, H, in H * m = h.
            - As we are damping towards the starting model, this is a diagonal
              matrix of ones.
        h:
            - (n_depth_points, 1) np.array
            - Constraints vector, h, in H * m = h.
            - As we are damping towards the starting model, this is the
              relevant slice of the starting model for the parameter specified
              by start_ind.
    """

    n_depth_points, start_ind, m0 = arguments

    H = np.diag(np.ones(n_depth_points))
    # m0 is already a column vector, so can just pull relevant indices from it
    h = m0[start_ind : start_ind+n_depth_points]

    return H, h

def _build_constraint_damp_vpvs(layers:ModelLayerIndices,
                                damping:ModelLayerValues, model_order:dict,
                                m0:np.array) -> (np.array, np.array):
    """ Damp Vp/Vs towards starting model values.

    In H * m = h, we want to end up with
                    Vp_new - ((Vp_old/Vs_old) * Vs_new) = 0
    As such, H is set to -vp/vs in the Vs relevant column and to 1 in the Vp
    column; h is zero always.

    We also set up the damping (and padding) in this function.

    Arguments:
        layers:
            - ModelLayerIndices
        damping:
            - ModelLayerValues
            - This has the damping coefficients for this constraint.
        model_order:
            - dict
            - This lays out the order of m0.
        m0:
            - (n_model_points, 1) np.array
            - Starting model.

    Returns:
        H:
            - (n_depth_points * 2, n_model_points) np.array
            - Constraints matrix, H, in H * m = h.
            - As we are damping towards the starting Vp/Vs, this is set to
              -vp/vs in the Vs relevant column and to 1 in the Vp column.
            - Note this is applied for horizontally and vertically polarised
              Vp/Vs as separate constraints.
        h:
            - (n_depth_points * 2, 1) np.array
            - Constraints vector, h, in H * m = h.
            - As we are damping towards the starting Vp/Vs, this is zero always.
    """
    n_depth_points = layers.depth.size

    # Vsv and Vpv
    H_v, h_v = _make_damp_vpvs_matrix('v', model_order, m0, layers, damping)
    # Vsh and Vph
    H_h, h_h = _make_damp_vpvs_matrix('h', model_order, m0, layers, damping)

    return np.vstack((H_v, H_h)), np.vstack((h_v, h_h))

def _make_damp_vpvs_matrix(polarity:str, model_order:dict, m0:np.array,
                           layers:ModelLayerIndices, damping:ModelLayerValues
                           ) -> (np.array, np.array):
    """ Damp Vp/Vs towards starting model values.

    In H * m = h, we want to end up with
                    Vp_new - ((Vp_old/Vs_old) * Vs_new) = 0
    As such, H is set to -vp/vs in the Vs relevant column and to 1 in the Vp
    column; h is zero always.

    We also set up the damping (and padding) in this function.

    Arguments:
        polarity:
            - str
            - Must be set to 'h' or 'v'.
            - Tells the function to look for horizontally polarised or
              vertically polarised Vp and Vs in m0.
        model_order:
            - dict
            - This lays out the order of m0, i.e. which model parameters are
              stacked and in which order.
            - The keys are the names of the model parameters, e.g. 'vsv', 'vsh'
            - The values are the order, such that the index of the first
              element of that parameter in m0 is dict[key] * n_depth_points.
        m0:
            - (n_model_points, 1) np.array
            - Starting model, with different parameters stacked on top of
              one another.
        layers:
            - ModelLayerIndices
        damping:
            - ModelLayerValues
            - This has the damping coefficients for this constraint.
            - Component arrays are (n_values_in_layer, n_vals_by_model_param),
              so can vary with depth across a layer and be different depending
              on the model_param.

    Returns:
        H:
            - (n_depth_points, n_model_points) np.array
            - Constraints matrix, H, in H * m = h.
            - As we are damping towards the starting Vp/Vs, this is set to
              -vp/vs in the Vs relevant column and to 1 in the Vp column.
            - This is for either horizontally and vertically polarised
              Vp/Vs.
        h:
            - (n_depth_points, 1) np.array
            - Constraints vector, h, in H * m = h.
            - As we are damping towards the starting Vp/Vs, this is zero always.
    """

    n_depth_points = layers.depth.size
    n_model_points = n_depth_points * (max(model_order.values()) + 1)

    # Find correct indices and values for Vp/Vs polarity
    vs_start_ind = model_order['vs' + polarity] * n_depth_points
    vp_start_ind = model_order['vp' + polarity] * n_depth_points
    vs_inds = np.arange(vs_start_ind, vs_start_ind + n_depth_points)
    vp_inds = np.arange(vp_start_ind, vp_start_ind + n_depth_points)
    vp_vs_diag = np.diag(m0[vp_inds] / m0[vs_inds])
    ones_diag = np.diag(np.ones(n_depth_points))

    # Construct H matrix (with padding)
    H = np.zeros(n_depth_points, n_model_points)
    H[:, vs_inds] = vp_vs_diag
    H[:, vp_inds] = ones_diag

    h = np.zeros((n_depth_points, 1))

    # Apply damping to H and h - need to do this one model parameter at a time
    # Apply to Vs columns
    H[:, vs_inds], h = _damp_constraints(H[:, vs_inds], h, layers, damping)
    # Apply to Vp columns
    H[:, vp_inds], h = _damp_constraints(H[:, vp_inds], h, layers, damping)


    return H, h

def _build_constraint_damp_anisotropy(layers:ModelLayerIndices,
                                      damping:ModelLayerValues,
                                      xi_vals:ModelLayerValues,
                                      model_order:dict,
                                      ) -> (np.array, np.array):
    """ Damp (Vsh/Vsv)^2 towards set value of Xi.

    (Vsh/Vsv)^2, or xi, is the radial anisotropy parameter (squiggly e).
    Xi = 1 means the layer is radially isotropic.  We want to damp towards
    some pre-set values, given in xi_vals.

    In H * m = h, we want to end up with
                    Vsh - sqrt(xi) * Vsv = 0
            (Via...  (Vsh/Vsv)^2 = Xi; Vsh = sqrt(Xi) * Vsv)
    As such, H is set to -sqrt(xi) in the Vsv relevant column and to 1 in the
    Vsh column; h is zero always.

    We also set up the damping (and padding) in this function.

    Arguments:
        layers:
            - ModelLayerIndices
        damping:
            - ModelLayerValues
            - This has the damping coefficients for this constraint.
            - Component arrays are (n_values_in_layer, n_vals_by_model_param),
              so can vary with depth across a layer and be different depending
              on the model_param.
        xi_vals:
            - ModelLayerValues
            - This has the values of Xi ((Vsh/Vsv)^2) that we are damping
              towards.
            - Component arrays must be (n_values_in_layer, 1), so can be
              constant or vary with depth across a layer, but it is assumed
              that there is only one value for all model parameters.
            - This will be made into an array using _build_layer_value_vector().
        model_order:
            - dict
            - This lays out the order of parameters in m0.

    Returns:
        H:
            - (n_depth_points, n_model_points) np.array
            - Constraints matrix, H, in H * m = h.
            - As we are damping towards the set Xi, this is set to
              -sqrt(Xi) in the Vsv relevant column and to 1 in the Vsh column.
        h:
            - (n_depth_points, 1) np.array
            - Constraints vector, h, in H * m = h.
            - As we are damping towards the starting Vp/Vs, this is zero always.
    """
    n_depth_points = layers.depth.size
    n_model_points = n_depth_points * (max(model_order.values()) + 1)

    # Find correct indices for Vsh and Vsv
    vsv_start_ind = model_order['vsv'] * n_depth_points
    vsh_start_ind = model_order['vsh'] * n_depth_points
    vsv_inds = np.arange(vsv_start_ind, vsv_start_ind + n_depth_points)
    vsh_inds = np.arange(vsh_start_ind, vsh_start_ind + n_depth_points)

    # Build Xi vector across the whole model space
    xi = _build_layer_value_vector(xi_vals, layers)

    # Construct H matrix (with padding)
    H = np.zeros(n_depth_points, n_model_points)
    ones_diag = np.diag(np.ones(n_depth_points))
    H[:, vsv_inds] = -ones_diag * np.sqrt(xi)
    H[:, vsh_inds] = ones_diag

    h = np.zeros((n_depth_points, 1))

    # Apply damping to H and h - need to do this one model parameter at a time
    # Apply to Vs columns
    H[:, vsv_inds], h = _damp_constraints(H[:, vsv_inds], h, layers, damping)
    # Apply to Vp columns
    H[:, vsh_inds], h = _damp_constraints(H[:, vsh_inds], h, layers, damping)

    return H, h

def _build_constraint_damp_original_gradient(
        arguments:tuple) -> (np.array, np.array):
    """ Damp Vsv and Vsh gradients between layers to starting model values.

    In H * m = h, we want to end up with
            V_new_layer1/V_old_layer1 - V_new_layer2/V_old_layer2 = 0
        (Via... V_new_layer1/V_new_layer2 = V_old_layer1/V_old_layer2)
    As such, H is set to 1/V_layer1 in the V, layer 1 column, and to
    1/V_layer2 in the V, layer 2 column; h is zero always.

    Arguments:
        arguments:
            - tuple of (n_depth_points, start_ind, m0)

        arguments[0], or n_depth_points:
            - int
            - Number of depth points in the model.
        arguments[1], or start_ind:
            - int
            - Index at which the parameter of interest (e.g. Vsv, Vsh) block
              starts in the input model, m0.
        arguments[2], or m0:
            - (n_model_points, 1) np.array
            - Starting model, with different parameters stacked on top of
              one another.
        arguments[3], or layers:
            - ModelLayerIndices

    Returns:
        H:
            - (n_depth_points, n_depth_points) np.array
            - Constraints matrix, H, in H * m = h.
            - Set to 1/V_layer1 in the V, layer 1 column, and to
              1/V_layer2 in the V, layer 2 column.
        h:
            - (n_depth_points, 1) np.array
            - Constraints vector, h, in H * m = h.
            - Set to zero always.
    """

    n_depth_points, start_ind, m0, layers = arguments

    H = np.zeros((n_depth_points, n_depth_points))
    for i_d in range(n_depth_point - 1):
        # Need to transpose slice of m0 (2, 1) to fit in H (1, 2) gap,
        # because in H, depth increases with column not row.
        H[i_d, [i_d, i_d+1]] = m0[[start_ind+i_d+1, start_ind+i_d]].T


    # Remove constraints around discontinuities between layers.
    for d in layers.discontinuities:
        H[d,:] *= 0
        if d > 0:
            H[d-1, :] *= 0

    h = np.zeros((n_depth_points, 1))

    return H, h



def _damped_least_squares(m0, G, d, W, D_mat, d_vec, H_mat, h_vec):
    """ Calculate the damped least squares, after Menke (2012).

    Least squares (Gauss-Newton solution):
    m = (G'*G)^-1 * G'* d       (i.e. d = G * m)
        - m:    new model (n_model_params*n_depth_points x 1)
        - G:    partial derivative matrix (n_data_points x n_model_points)
        - d:    data misfit (n_data_points x 1)

        where n_model_points = n_model_params * n_depth_points
            (model parameters are Vsv, Vsh, Vpv, Vph, eta (= F/(A-2L), anellipticity))

    Damped least squares in the Menke (2012; eq. 345) notation:
    m = (G' * We * G  + ε^2 * Wm )^-1  (G' * We * d + ε^2 * Wm * <m>)
        - m:    new model
                    (n_model_points x 1) == (n_model_params*n_depth_points x 1)
        - G:    partial derivatives matrix
                    (n_data_points x n_model_points)
        - We:   data error weighting matrix (our input W)
                    (n_data_points x n_data_points)
        - ε:    damping parameters
        - Wm:   smoothing matrix (in terms of our input: D' * D)
                    (n_model_points x n_model_points)
        - d:    data (e.g. phase velocity at each period)
                    (n_data_points x 1)
        - <m>:  old (a priori) model
                    (n_model_points x 1)
    Where
        - D:    roughness matrix
                    (n_smoothing_equations x n_model_points)

    This is formulated in Menke (2012; eq. 3.46) as
        F * m_est  = f
        F  	=   [    sqrt(We) * G ;    εD   ]
        f 	=   [    sqrt(We) * d ;    εD<m>   ]

        i.e. m_est = (F' * F)^-1 * F' * f
                   = [(G' * We * G) + (ε^2 * D' * D)]^-1
                     * [(G' * We * d) + (ε^2 * D' * D * <m>) ]

    Then add in any a priori constraints, formulated by Menke (2012; eq. 3.55)
    as linear constraint equations:
        H * m = h

    e.g. to damp to the starting model, H would be a diagonal matrix of ones,
    and h would be the original model.

    These are combined as vertical stacks:
        all damping matrices = [εD; H]
        all damping results = [εD<m>; h]


    """

    H = np.vstack((D_mat, H_mat))
    h = np.vstack((d_vec, h_vec))

    F = np.vstack((np.matmul(np.sqrt(W), G), H))
    f = np.vstack((np.matmul(np.sqrt(W), d), h))

    Finv_denominator = np.matmul(F.T, F)
    # x = np.linalg.lstsq(a, b) solves for x: ax = b, i.e. x = a \ b in MATLAB
    Finv = np.linalg.lstsq(Finv_denominator, F.T, rcond=None)[0]

    new_model = np.matmul(Finv, f)
    #
    # H = [D2; H1; H2; H3; H4; H6; H7; H8; H9]; % where H = D
    # h = [d2; h1; h2; h3; h4; h6; h7; h8; h9]; % h = D*mhat
    # epsilon_Gmd_vec = ones(length(Ddobs),1) * epsilon_Gmd;
    # epsilon_Gmd_vec(Ilongper) = epsilon_Gmd_longper; % assign different damping to long period data
    # F = [epsilon_Gmd_vec.*We.^(1/2)*GG; epsilon_HhD*H];
    # % f = [We.^(1/2)*dobs; epsilon2*h];
    # f = [epsilon_Gmd_vec.*We.^(1/2)*Ddobs; epsilon_HhD*h];
    # [MF, NF] = size(F);
    # Finv = (F'*F+epsilon_0norm*eye(NF,NF))\F'; % least squares
    # mest_all = Finv*f;

    return new_model

def _build_earth_model_from_vector(model_vector):

    return earth_model
