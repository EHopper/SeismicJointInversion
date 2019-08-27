""" Inversion from phase velocities to velocity model.

This is based on MATLAB code from Josh Russell and Natalie Accardo (2019),
and formulated after Geophysical Data Analysis: Discrete Inverse Theory.
(DOI: 10.1016/B978-0-12-397160-9.00003-5) by Bill Menke (2012).

For simplicity, I am only doing this for Rayleigh waves at the moment,
although there are some bits scattered around for Love waves that are commented
out.  This isn't complete, though, and for now I am not going to test those.
In any comments talking about stacking Love and Rayleigh kernels/data etc,
this indicates how it should be done, not that it is being done here.

"""

#import collections
import typing
import numpy as np
import pandas as pd

from util import define_models
from util import mineos
from util import constraints
from util import partial_derivatives
from util import weights


# =============================================================================
#       Run the Damped Least Squares Inversion
# =============================================================================
def run_with_no_inputs():

    setup_model = define_models.SetupModel(
        depth_limits=np.array([0., 200.]),
        boundary_vsv=np.array([3.5, 4.0, 4.2, 4.1]),
        boundary_widths=np.array([5., 10.]),
        boundary_depth_uncertainty=np.array([3., 10.,]),
        boundary_depths=np.array([35., 90]),
        id='test')

    data = constraints.extract_observations(35, -104)

    return run_inversion(setup_model, data)


def run_inversion(setup_model:define_models.SetupModel,
                  data:constraints.Observations,
                  n_iterations:int=5) -> (define_models.InversionModel):
    """ Set the inversion running over some number of iterations.

    """

    model = define_models.setup_starting_model(setup_model)

    for i in range(n_iterations):
        # Still need to pass setup_model as it has info on e.g. vp/vs ratio
        # needed to convert from InversionModel to MINEOS card
        model = _inversion_iteration(setup_model, model, data)

    return model

def _inversion_iteration(setup_model:define_models.SetupModel,
                         model:define_models.InversionModel,
                         data:constraints.Observations
                         ) -> define_models.InversionModel:
    """ Run a single iteration of the least squares
    """

    # Build all of the inputs to the damped least squares
    # Run MINEOS to get phase velocities and kernels
    mineos_model = define_models.convert_inversion_model_to_mineos_model(
        model, setup_model
    )
    # Can vary other parameters in MINEOS by putting them as inputs to this call
    # e.g. defaults include l_min, l_max; qmod_path; phase_or_group_velocity
    params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
    ph_vel_pred, kernels = mineos.run_mineos_and_kernels(
        params, data.surface_waves.Period.values, setup_model.id
    )
    kernels = kernels[kernels['z'] <= setup_model.depth_limits[1]]

    G = partial_derivatives._build_partial_derivatives_matrix(
        kernels, model, setup_model
    )
    p = _build_model_vector(model)
    d = _build_data_misfit_matrix(data, ph_vel_pred, p, G)

    # Build all of the weighting functions for damped least squares
    layer_indices = define_models._set_model_indices(setup_model, model)
    W, D_mat, d_vec, H_mat, h_vec = (
        weights.build_weighting_damping(data, layer_indices, p)
    )

    # Perform inversion
    model_new = _damped_least_squares(p, G, d, W, D_mat, d_vec, H_mat, h_vec)

    return _build_inversion_model_from_model_vector(model_new)


def _build_model_vector(model:define_models.InversionModel,
                        ) -> (np.array):
    """ Make model into column vector [s; t].

    Arguments:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s, km)
            - Input Vs model

    Returns:
        p:
            - (n_depth points + n_boundary_layers, 1) np.array
            - Units:    seismological, so km/s for velocities (s),
                        km for layer thicknesses (t)
            - Inversion model, p, is made up of velocities defined at various
              depths (s) and thicknesses for the layers above boundary layers
              of interest (t), i.e. controlling the depth of e.g. Moho, LAB
            - All other thicknesses are either fixed, or are dependent only
              on the variation of thicknesses, t
            - Note that the deepest velocity point is fixed too.
            - This model, p, ONLY includes the parameters that we are inverting
              for - it is not a complete description of vs(z)!
    """

    return np.vstack((model.vsv[:-1], model.thickness[model.boundary_inds]))

def _build_inversion_model_from_model_vector(
        p:np.array, model:define_models.InversionModel):
    """ Make column vector, [s; t] into InversionModel format.

    Arguments:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s, km)
            - Input Vs model
        p:
            - (n_depth points + n_boundary_layers, 1) np.array
            - Units:    seismological, so km/s for velocities (s),
                        km for layer thicknesses (t)

    Returns:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s, km)
            - Vs model with values updated from p.
    """
    new_thickness = model.thickness
    new_thickness[model.boundary_inds] = p[-2:]

    return define_models.InversionModel(
                vsv = np.vstack((p[:-2], model.vsv[-1])),
                thickness = new_thickness,
                boundary_inds = model.boundary_inds,
    )


def _build_data_misfit_matrix(data:np.array, prediction:np.array,
        m0:np.array, G:np.array):
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
            - (n_periods, ) np.array
            - Units:    data.surface_waves['Phase_vel'] is in km/s
            - Observed phase velocity data
            - Extracted from a constraints.Observations object,
              all_data.surface_waves['Phase_vel']
        prediction:
            - (n_periods, ) np.array
            - Units:    km/s
            - Previously calculated phase velocities for the input model.
        m0:
            - (n_model_points, 1) np.array
            - Units:    seismology units, i.e. km/s for velocities
            - Earth model in format for calculating the forward problem, G * m0.
        G:
            - (n_periods, n_model_points) np.array
            - Units:    assumes velocities in km/s
            - Matrix of partial derivatives of phase velocity wrt the
              Earth model (i.e. stacked Frechet kernels converted for use
              with m0).

    Returns:
        data_misfit:
            - (n_periods, ) np.array
            - Units:    km/s
            - The misfit of the data to the predictions, altered to account
              for the inclusion of G * m0.


    """

    forward_problem_predictions = np.matmul(G, m0).flatten()

    data_misfit = forward_problem_predictions + (data - predictions)

    return data_misfit


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
