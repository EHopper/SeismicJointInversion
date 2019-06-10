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

class DampingParameters(typing.NamedTuple):
    """ Parameters used for smoothing.

    Fields:
        -

    """

    #phase_or_group_velocity: str = 'ph'
    #l_min: int = 0


class ModelLayerIndices(typing.NamedTuple):
    """ Parameters used for smoothing.

    Fields:
        -

    """

    #phase_or_group_velocity: str = 'ph'
    #l_min: int = 0

class LoveKernels(typing.NamedTuple):
    """ Frechet kernels for Love (toroidal) waves.

    Kernels for different periods are stacked on top of each other.

    Fields:
        - period: period (s) of Love wave of interest
        - depth: depth (km) vector for kernel
        - vsv: vertically polarised S wave kernel
        - vsh: horizontally polarised S wave kernel

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
        - period: period (s) of Rayleigh wave of interest
        - depth: depth (km) vector for kernel
        - vsv: vertically polarised S wave kernel
        - vpv: vertically polarised P wave kernel
        - vph: horizontally polarised P wave kernel
        - eta: anellipticity (eta = F/(A-2L)) kernel
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
        model = _inversion_iteration(model, data, layer_indices)

    return model

def _inversion_iteration(model, data, layer_indices):
    """ Run a single iteration of the least squares
    """

    # Build all of the inputs to the damped least squares
    rayleigh_kernels, love_kernels, depth = _calculate_Frechet_kernels(model)
    m0 = _build_model_vector(model, depth)
    G = _build_partial_derivatives_matrix(rayleigh_kernels, love_kernels)
    d = _build_data_misfit_matrix(data, model, m0, G)

    # Build all of the weighting functions for damped least squares
    layer_indices = _set_layer_indices(m0)
    W, D, H = _build_weighting_matrices(data, layer_indices)

    # Perform inversion
    model_new = _damped_least_squares(m0, G, d, W, D, H)

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
        - model:
            This will ultimately be used to make a card for MINEOS.  Now it's
            a placeholder for when I do things properly.
        - max_depth (float):
            Depth (km) below which to crop off the Frechet kernels (i.e. do
            not invert for Earth structure below this depth)

    Returns
        - rayleigh_kernels, love_kernels:
            Rayleigh wave (spherical) kernels and Love wave (toroidal) kernels.
        - depth:
            Vector of depth (km) that is the same as in the loaded kernels.
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
        - model (define_earth_model.EarthModel):
            Input Vs, Vp model (km/s)
        - kernels (RayleighKernels, LoveKernels):
            This is a non-MINEOS workaround to make sure the depth vectors are
            the same

    Returns:
        - m0 (np.array):
            Vsv, Vsh, Vpv, Vph, eta model stacked on top of each other
            in SI units (i.e. m/s for velocities).
    """

    # Find depth vector from kernels and interpolate model over it
    depth = _make_2D(depth)

    vsv = _interpolate_earth_model_parameter(depth, model, 'vs') * 1e3
    vsh = _interpolate_earth_model_parameter(depth, model, 'vs') * 1e3
    vpv = _interpolate_earth_model_parameter(depth, model, 'vp') * 1e3
    vph = _interpolate_earth_model_parameter(depth, model, 'vp') * 1e3
    eta = np.ones_like(depth)

    return np.vstack((vsv, vsh, vpv, vph, eta))

def _make_2D(x:np.array) -> np.array:
    """ Turn a 1D numpy array into a column vector. """

    return x[np.newaxis].T

def _interpolate_earth_model_parameter(new_depth:np.array,
                                       model:define_earth_model.EarthModel,
                                       old_field:str) -> np.array:
    """ Convert from EarthModel layout and interpolate.

    """

    depth, field = define_earth_model._convert_earth_model(model, old_field)
    field = np.interp(new_depth, depth, field)

    return field


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


def _hstack_frechet_kernels(kernel, period):
    """


    """

    vsv = kernel.vsv[kernel.period == period] * 1e3

    if kernel.type == 'rayleigh':#isinstance(kernel, RayleighKernels):
        vsh = np.zeros_like(vsv)
        vpv = kernel.vpv[kernel.period == period] * 1e3
        vph = kernel.vph[kernel.period == period] * 1e3
        eta = kernel.eta[kernel.period == period] * 1e3

    if kernel.type == 'love':#isinstance(kernel, LoveKernels):
        vsh = kernel.vsh[kernel.period == period] * 1e3
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
        data (surface_waves.ObsPhaseVelocity):
            - observed phase velocity data (data.c is shape (n_periods,))
        model (define_earth_model.EarthModel):
            - Earth model used to predict the phase velocity data
        m0 (np.array):
            - Earth model ([Vsv; Vsh; Vpv; Vph; Eta]) in format for
              calculating the forward problem, G * m0.
              This is the same as 'model', though reformatted into a
              5 * n_depth_points x 1 array.
        G (np.array):
            - Matrix of partial derivatives of phase velocity wrt the
              Earth model (i.e. stacked Frechet kernels).  This is a
              n_periods x 5 * n_depth_points array.

    Returns:
        data_misfit (np.array):
            - The misfit of the data to the predictions, altered to account
              for the inclusion of G * m0.
              This is a flattened (n_periods,) array.


    """

    predictions = surface_waves.synthesise_surface_wave(model, data.period)
    forward_problem_predictions = np.matmul(G, m0).flatten()

    data_misfit = forward_problem_predictions + (data.c - predictions.c)

    return data_misfit

def _build_weighting_matrices(data, model_vector, layer_indices):
    """ Build all of the weighting matrices.
    """
    data_errors = _build_error_weighting_matrix(data)
    roughness_matrix, roughness = _build_smoothing_matrix(
        model_vector, layer_indices)
    a_priori_constraints = _build_constraint_eq_matrix(model_vector)

    return data_errors, roughness_matrix, roughness, a_priori_constraints

def _build_error_weighting_matrix(data):
    """ Build the error weighting matrix from data standard deviation.

    This is just a diagonal matrix of the inverse of standard deviation.

    Arguments:
        data (surface_waves.ObsPhaseVelocity):
            - Observed surface wave information.  Here, we are interested
              in the standard deviation of the data (data.std).  If no
              standard deviation is given, we assume 0.15.
              We have n_periods values in our data structure.

    Returns:
        error weighting matrix (We in the syntax of Menke (2012) and docstring
        for _damped_least_squares()) (np.array):
            - Diagonal matrix of 1 / standard deviation, i.e. measurements
              with smaller standard deviations will be weighted more strongly.
              This is an n_periods x n_periods array.
    """

    # If standard deviation of data is missing (= 0), fill in with guess
    obs_std = data.std
    obs_std[obs_std == 0] = 0.15


    return np.diag(1 / obs_std)

def _build_smoothing_matrices(m0):
    """ Build the matrices needed to minimise second derivative of the model.

    Note that Josh says this can be better captured by a linear constraint
    preserving layer gradients (and therefore stopping the model from getting
    any rougher).  So perhaps this can just be removed.

    The smoothing parameters will be different for crust vs. upper mantle etc

    Returns:
        - roughness_matrix (np.array):
            This is the matrix that we multiply by the model to find the
            roughness of the model (i.e. the second derivative of the model).
            In Menke (2012)'s notation, this is D and Wm is D^T * D.
            This will be a n_smoothing_equations x n_model_points array, where
            n_model_points is 5 * n_depth_points (i.e. Vsv, Vsh, Vpv, Vph, Eta
            for all depths).
        - roughness (np.array):
            This is the permitted roughness of the model.  Actual roughness
            is calculated as roughness_matrix * model, and by setting this
            equal to the permitted roughness in the damped least squares
            inversion, we will tend towards the solution of this permitted
            roughness (i.e. smoothness)
                roughness_matrix * model = roughness
            Because we want as smooth a model as possible, permitted roughness
            is always set to zero.
            This is a n_smoothing_equations x 1 array.

    """

    ######## FILL THIS IN #############

    return roughness_matrix, roughness

def _make_banded_matrix(matrix_size, central_values):
    """ Make a banded matrix with central_values along the diagonal band.

    This will return a square matrix.
    """
    diag_matrix = np.zeros((matrix_size, matrix_size))

    diag_ind = len(central_values) // 2 # // is floor

    for n_row in range(diag_ind):
        vals = central_values[(diag_ind - n_row):]
        diag_matrix[n_row, :len(vals)] = vals

    for n_row in range(diag_ind, matrix_size-diag_ind):
        start_i = n_row - diag_ind
        vals = central_values
        diag_matrix[n_row, start_i:start_i + len(vals)] = vals

    for n_row in range(matrix_size-diag_ind, matrix_size):
        vals = central_values[:(diag_ind + matrix_size - n_row)]
        diag_matrix[n_row, -len(vals):] = vals


    return diag_matrix

def _build_constraint_eq_matrix(m0):
    """ Build the matrix of constraint equations.

    e.g. damp towards the starting model, damp towards a Vp/Vs value
    The damping will also probably be different for crust vs. upper mantle etc
    """

    ######## FILL THIS IN #############

    return constraints

def _damped_least_squares(m0, G, d, W, D, H):
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
        - m:    new model (n_model_params*n_depth_points x 1)
        - G:    partial derivatives matrix
        - We:   data error weighting matrix (our input W)
        - ε:    damping parameters
        - Wm:   smoothing matrix (in terms of our input: D' * D)
        - d:    data
        - <m>:  old (a priori) model

    This is formulated in the Menke (2012; eq. 3.46) as
        F * m_est  = f
        F  	=   [    sqrt(We) * G ;    εD   ]
        f 	=   [    sqrt(We) * d ;    εD<m>   ]

        i.e. m_est = (F' * F)^-1 * F' * f
                   = [(G' * We * G) + (ε^2 * D' * D)]^-1
                     * [(G' * We * d) + (ε^2 * D' * D * <m>) ]

    Then add in any a priori constraints, formulated by Menke (2012; eq. 3.55)
    as linear constrain equations:
        H * m = h

    e.g. to damp to the starting model, H would be a diagonal matrix of ones,
    and h would be the original model.

    These are combined as vertical stacks:
        all damping matrices = [εD; H]
        all damping results = [εD<m>; h]


    """
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
