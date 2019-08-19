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

from util import matlab
from util import define_models
from util import mineos
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



# =============================================================================
#       Run the Damped Least Squares Inversion
# =============================================================================

def run_inversion(setup_model:define_models.SetupModel,
                  data:surface_waves.ObsPhaseVelocity,
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
                         data:surface_waves.ObsPhaseVelocity
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
    ph_vel, kernels = mineos.run_mineos_and_kernels(params, periods,
                                                    setup_mode.id)
    kernels = kernels[kernels['z'] <= setup_model.depth_limits[1]]

    p = _build_model_vector(model)
    G = _build_partial_derivatives_matrix(kernels, model)
    d = _build_data_misfit_matrix(data, model, m0, G)

    # Build all of the weighting functions for damped least squares
    W = _build_error_weighting_matrix(data)
    layer_indices = _set_layer_indices(m0)
    D_mat, d_vec, H_mat, h_vec = _build_weighting_matrices(data, layer_indices)

    # Perform inversion
    model_new = _damped_least_squares(m0, G, d, W, D_mat, d_vec, H_mat, h_vec)

    return _build_earth_model_from_vector(model_new)


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
            - This model, p, ONLY includes the parameters that we are inverting
              for - it is not a complete description of vs(z)!
    """

    return np.vstack((model.vsv, model.thickness[model.boundary_inds]))


def _build_partial_derivatives_matrix(kernels:pd.DataFrame,
                                      model:define_models.InversionModel):
    """ Make partial derivative matrix, G, by stacking the Frechet kernels.

    Build the G matrix.
    Ultimately, this will be a n_Love_periods+n_Rayleigh_periods by
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

    For now, we are only using Rayleigh waves, so in the above explanation,
    n_Love_periods = 0, i.e. there are no rows in G for the T_*** kernels.

    Arguments:
        rayleigh:
            - mineos.RayleighKernels
        (love:
            - mineos.LoveKernels
            - Not currently passed, but if want to use both Rayleigh and Love,
              easier to pass this in as a separate argument)

    Returns:
        G_inversion_model:
            - (, ) np.array
            - Units:
            -

    """

    periods = np.unique(kernels['period'])

    G_Rayleigh = _hstack_frechet_kernels(kernels, periods[0])
    # G_Love = _hstack_frechet_kernels(love, periods[0])
    for i_p in range(1,len(periods)):
        G_Rayleigh = np.vstack((G_Rayleigh,
                               _hstack_frechet_kernels(kernels, periods[i_p])))
        # G_Love = np.vstack((G_Love,
        #                     _hstack_frechet_kernels(love, periods[i_p])))

    # G_MINEOS is dc/dm matrix
    G_MINEOS = G_Rayleigh #np.vstack((G_Love, G_Rayleigh))

    # Convert to kernels for the model parameters we are inverting for
    dm_dp_mat = _convert_to_model_kernels(kernels['z'].unique(), model)
    G_inversion_model = np.matmul(G_MINEOS, dm_dp_mat)

    return G_inversion_model


def _hstack_frechet_kernels(kernels, period:float):
    """ Append all of the relevent Frechet kernels into a row of the G matrix.

    Different parameters are of interest for Rayleigh (Vsv, Vpv, Vph, Eta)
    and Love (Vsv, Vsh) waves.

    Arguments:
        kernels:
            - pd.DataFrame
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

    # Note: To have kernels scaled for changes in velocity in SI
    #       units (i.e. m/s not km/s), multiply all kernels (including eta)
    #       by 1e3.  Even though MINEOS requires SI input, the kernel output
    #       assumes seismological (km/s) units!
    vsv = kernels.vsv[kernels.period == period]

    if kernels['type'].iloc[0] == 'Rayleigh':#isinstance(kernel, mineos.RayleighKernels):
        vsh = np.zeros_like(vsv)
        vpv = kernels.vpv[kernels.period == period]
        vph = kernels.vph[kernels.period == period]
        eta = kernels.eta[kernels.period == period]

    if kernels['type'].iloc[0]  == 'Love':#isinstance(kernel, mineos.LoveKernels):
        vsh = kernels.vsh[kernels.period == period]
        vpv = np.zeros_like(vsv)
        vph = np.zeros_like(vsv)
        eta = np.zeros_like(vsv)

    return np.hstack((vsv, vsh, vpv, vph, eta))

def _convert_to_model_kernels(depth, model):
    """ Convert from Frechet kernels as function of v(z) to function of m0.

    The Frechet kernels from MINEOS are given at the same depths as those in
    the MINEOS model card, as a function of the model card v(z).  However, we
    want to invert for a model in a different format, so we need to adjust
    the kernels accordingly.

    Let   m:    old model, as a function of depth, from MINEOS card
                multiple parameters - [vsv, vsh, vpv, vph, eta]
          p:    new model
                multiple parameters - [s, t]
          c:    observed phase velocities, Love stacked on top of Rayleigh

    The MINEOS kernels are a matrix of dc/dm
        i.e.   dc_0/dvsv_0, ..., dc_0/dvsv_M, dc_0/dvsh_0, ..., dc_0/deta_M
            [       :     ,  : ,      :     ,      :     ,  : ,      :      ]
                dc_N/dvsv_0, ..., dc_N/dvsv_M, dc_N/dvsh_0, ..., dcN_deta_M

    We want a matrix of dc/dp
        i.e.   dc_0/ds_0, ..., dc_0/ds_P, dc_0/dt_0, ..., dc_0/dt_D
            [       :   ,  : ,    :     ,    :     ,  : ,    :      ]
               dc_N/ds_0, ..., dc_N/ds_P, dc_N/dt_0, ..., dc_N/dt_D


    As dx/dy = dx/da * da/dy, we need to find the matrix dm/dp
        i.e.   dvsv_0/ds_0, ..., dvsv_0/ds_P, dvsv_0/dt_0, ..., dvsv_0/dt_D
                    :     ,  : ,    :       ,    :       ,  : ,    :
            [  dvsv_M/ds_0, ..., dvsv_M/ds_P, dvsv_M/dt_0, ..., dvsv_M/dt_D  ]
               dvsh_0/ds_0, ..., dvsh_0/ds_P, dvsh_0/dt_0, ..., dvsh_0/dt_D
                    :     ,  : ,    :       ,    :       ,  : ,    :
               deta_M/ds_0, ..., deta_M/ds_P, deta_M/dt_0, ..., deta_M/dt_D

    By matrix multiplication, this works out as
        e.g. dc_0/ds_0 = sum(dc_0/dm_a * dm_a/ds_0)
    where the sum is from a = 0 to a = N.

    A lot of these partial derivatives (dm_a/dp_b) will be zero,
    e.g. for values of vsv, vsh, vpv, vph that are far from the value of s
         or t that is being varied; eta is held constant and never dependent
         on our model parameters [s, t].
    The ones that are non-zero are calculated differently depending on where
    the depth point at which m_a is defined (z_a) is compared to the depth
    of the model parameter, p_b, that is being varied.  So we will call
    different functions to build the partial derivatives in the layer
    above p_b, the layer below p_b (and, when we are varying t, the layer two
    layers below p_b, i.e. the layer below the boundary layer).

    Given that s is just Vsv defined at a number of points in depth, we find
    the partial derivatives of the other velocities (vsh, vpv, vph) by
    scaling between them.

    Arguments:
        depth:
            - (n_card_depths, ) np.array
            - Units:    kilometres
            - Depth vector for MINEOS kernel.
        model:
            - define_models.InversionModel
            - Units:    seismological (i.e. vsv in km/s, thickness in km)
            - Remember that the field boundary_inds contains the indices of
              the boundaries that we are inverting the depth/width of.  So if
              i_b is the first of these boundary_inds,
                model.vsv[i_b]: Vsv at the top of the layer
                model.vsv[i_b + 1]: Vsv at the bottom of the layer
                model.thickness[i_b]: thickness of the layer above the boundary
                    (controls the depth of the boundary layer)
                np.sum(model.thickness[:i_b + 1]): depth of the top of the
                    boundary layer (i.e. up to and including thickness[i_b])
                model.thickness[i_b + 1]: width of the boundary layer itself
            - Remember that, for a given inversion, we are fixing the width
              of the boundary layer and only varying the depth to the top of
              of the boundary layer.
                i.e.    t = model.thickness[model.boundary_inds]

    Returns:
        dm_dp_mat:
            - (n_MINEOS_model_points,
               n_inversion_model_depths + n_boundary_layers) np.array
            - Units:    seismological (i.e. vsv in km/s, thickness in km)
            - This is dm/dp, where p = [s; t], s = vsv defined at a series of
              depth points and t = the thickness of the layer overlying a
              boundary layer (and thus controlling its depth), and m is the
              model that corresponds to the MINEOS kernels.


    """

    # Build dm/dp matrix up column by column
    n_layers = model.vsv.size - 1 # last value of s is pinned (not inverted for)
    dm_ds_mat = np.zeros((depth.size, n_layers))
    dm_dt_mat = np.zeros((depth.size, model.boundary_inds.size))

    # First, do dm/ds column by column
    # Build first column, dm/ds_0 - only affects layers deeper than s_0
    dm_ds_mat = _convert_kernels_d_deeperm_by_d_s(model, 0, depth, dm_ds_mat)
    # Build other columns, dc/ds_i
    for i in range(1, n_layers):
        dm_ds_mat = _convert_kernels_d_shallowerm_by_d_s(
            model, i, depth, dm_ds_mat
        )
        dm_ds_mat = _convert_kernels_d_deeperm_by_d_s(
            model, i, depth, dm_ds_mat
        )

    # Now, do dm/dt column by column
    for i in range(model.boundary_inds.size):
        dm_dt_mat = _convert_kernels_d_shallowerm_by_d_t(
            model, i, depth, dm_dt_mat
        )
        dm_dt_mat = _convert_kernels_d_withinboundarym_by_d_t(
            model, i, depth, dm_dt_mat
        )
        dm_dt_mat = _convert_kernels_d_deeperm_by_d_t(
            model, i, depth, dm_dt_mat
        )

    dm_dp_mat = np.hstack((dm_ds_mat, dm_d_mat))

    return dm_dp_mat


def _convert_kernels_d_shallowerm_by_d_s(model:define_models.InversionModel,
                                         i:int, depth:np.array,
                                         dm_ds_mat:np.array):
    """ Find dm/ds for the model card points above the boundary layer.

    To convert the MINEOS kernels (dc/dm) to inversion model kernels (dc/dp),
    we need to define the matrix dm/dp (dc/dp = dc/dm * dm/dp).  We divide
    our inversion model, p, into two parts: the defined velocities at various
    depths (s) and the depth and thickness of certain boundary layers of
    interest (t).

    Here, we calculate dm/ds for the model card points above (shallower than)
    the boundary layer.  In the following description, I'm replacing subscripts
    with underscores - everything before the next space should be subscripted.
    e.g. y_i+1 is the 'i+1'th value of y.

    We can define the model card Vsv (m = v(z)) in terms of s as follows:
      - For every point s_i, we define the depth of that point, y_i, as the
        sum of the thicknesses above it: np.sum(thickness[:i + 1])
            (Remember model.thickness[i] is the thickness of the layer ABOVE
             the point where model.vsv[i] is defined)
      - For any depth, z_a, find b s.t. y_b <= z_a < y_b+1
            (Remember that v(z) is just linearly interpolated between s values)
      - Can then define v_a as
            v_a = s_b + (s_b+1 - s_b)/(y_b+1 - y_b) * (z_a - y_b)
      - Note that if z_a == y_b, then v_a == s_b

    Here, we are looking specifically at the values of m that are shallower than
    the s point in question, s_i, that will still be affected by a change to
    s_i - that is, the values of z between y_i-1 and y_i.  So i = b+1 in the
    equation above.
            v_a = s_i-1 + (s_i - s_i-1)/(y_i - y_i-1) * (z_a - y_i-1)

    In terms of the partial derivative:
            d(v_a)/d(s_i) = (z_a - y_i-1)/(y_i - y_i-1)

    Note that (y_i - y_i-1) is equivalent to thickness[i], the thickness of
    the layer above the point i.

    Here, we are calculating this one layer at a time, with the loop in
    the calling function - passing the index, i, in as an argument.  This fills
    some of a single column of dm_ds_mat ([dm_0/ds_i; ...; dm_N/ds_i]),
    although most of these values will be zero.

    Arguments:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s and km)
            - Model in layout ready for easy conversion to column vector
              to be used in least squares inversion.
        i:
            - int
            - Units:    n/a
            - Index in InversionModel for which we are calculating the column
              of partial derivatives.
        depth:
            - (n_card_depths, ) np.array
            - Units:    kilometres
            - Depth vector for MINEOS kernel.
        dm_ds_mat:
            - (n_card_depths, n_inversion_model_depths) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/ds that we are filling in a bit
              at a time.

    Returns:
        dm_ds_mat:
            - (n_card_depths, n_inversion_model_depths) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/ds with a few more values filled
              in - specifically, those in the 'i'th column (for model parameter
              s_i) in rows corresponding to depths between y_i-1 and y_i.

    """
    # Find the layers in card depth, z, shallower than the depth where s_i is
    # defined, y_i, and deeper than the depth where s_i-1 is specified, y_i-1.
    # These are the points in z that will be affected by varying s_i
    y_i_minus_1 = np.sum(model.thickness[:i])
    y_i = np.sum(model.thickness[:i+1])

    d_inds, = np.where(np.logical_and(y_i_minus_1 < depth, depth <= y_i))

    for i_d in d_inds:
        dm_ds_mat[i_d, i] = ((depth[i_d] - y_i_minus_1)
                             /model.thickness[i])

    return dm_ds_mat

def _convert_kernels_d_deeperm_by_d_s(model, i, depth, dm_ds_mat):
    """ Find dm/ds for the model card points above the boundary layer.

    To convert the MINEOS kernels (dc/dm) to inversion model kernels (dc/dp),
    we need to define the matrix dm/dp (dc/dp = dc/dm * dm/dp).  We divide
    our inversion model, p, into two parts: the defined velocities at various
    depths (s) and the depth and thickness of certain boundary layers of
    interest (t).

    Here, we calculate dm/ds for the model card points above (shallower than)
    the boundary layer.  In the following description, I'm replacing subscripts
    with underscores - everything before the next space should be subscripted.
    e.g. y_i+1 is the 'i+1'th value of y.

    We can define the model card Vsv (m = v(z)) in terms of s as follows:
      - For every point s_i, we define the depth of that point, y_i, as the
        sum of the thicknesses above it: np.sum(thickness[:i + 1])
            (Remember model.thickness[i] is the thickness of the layer ABOVE
             the point where model.vsv[i] is defined)
      - For any depth, z_a, find b s.t. y_b <= z_a < y_b+1
            (Remember that v(z) is just linearly interpolated between s values)
      - Can then define v_a as
            v_a = s_b + (s_b+1 - s_b)/(y_b+1 - y_b) * (z_a - y_b)
      - Note that if z_a == y_b, then v_a == s_b

    Here, we are looking specifically at the values of m that are deeper than
    the s point in question, s_i, that will still be affected by a change to
    s_i - that is, the values of z between y_i and y_i+1.  So i = b in the
    equation above.
            v_a = s_i + (s_i+1 - s_i)/(y_i+1 - y_i) * (z_a - y_i)

    In terms of the partial derivative:
            d(v_a)/d(s_i) = 1 - (z_a - y_i)/(y_i+1 - y_i)

    Note that (y_i+1 - y_i) is equivalent to thickness[i+1], the thickness of
    the layer below the point i.

    Here, we are calculating this one layer at a time, with the loop in
    the calling function - passing the index, i, in as an argument.  This fills
    some of a single column of dm_ds_mat ([dm_0/ds_i; ...; dm_N/ds_i]),
    although most of these values will be zero.

    Arguments:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s and km)
            - Model in layout ready for easy conversion to column vector
              to be used in least squares inversion.
        i:
            - int
            - Units:    n/a
            - Index in InversionModel for which we are calculating the column
              of partial derivatives.
        depth:
            - (n_card_depths, ) np.array
            - Units:    kilometres
            - Depth vector for MINEOS kernel.
        dm_ds_mat:
            - (n_card_depths, n_inversion_model_depths) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/ds that we are filling in one
              bit at a time.

    Returns:
        dm_ds_mat:
            - (n_card_depths, n_inversion_model_depths) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/ds with a few more values filled
              in - specifically, those in the 'i'th column (for model parameter
              s_i) in rows corresponding to depths between y_i and y_i+1.
    """
    # Find h, the number of layers in card depth deeper than defined s point
    # that will be affected by varying that s
    y_i = np.sum(model.thickness[:i+1])
    y_i_plus_1 = np.sum(model.thickness[:i+2])

    d_inds, = np.where(np.logical_and(y_i <= depth, depth < y_i_plus_1))

    for i_d in d_inds:
        dm_ds_mat[i_d, i] = 1 - ((depth[i_d] - y_i)
                                 /model.thickness[i+1])

    return dm_ds_mat

def _convert_kernels_d_shallowerm_by_d_t(model:define_models.InversionModel,
                                         i:int, depth:np.array,
                                         dm_dt_mat:np.array) -> np.array:
    """ Find dm/dt for the model card points above the boundary layer.

    To convert the MINEOS kernels (dc/dm) to inversion model kernels (dc/dp),
    we need to define the matrix dm/dp (dc/dp = dc/dm * dm/dp).  We divide
    our inversion model, p, into two parts: the defined velocities at various
    depths (s) and the depth and thickness of certain boundary layers of
    interest (t).

    Here, we calculate dm/dt for the model card points above (shallower than)
    the boundary layer.  In the following description, I'm replacing subscripts
    with underscores - everything before the next space should be subscripted.
    e.g. y_i+1 is the 'i+1'th value of y.

    The values, t_i, in the model, p, have slightly confusing effects on v(z)
    because we have to go via the effect on s, the velocities defined in p.
    Let's define ib, the index in s that corresponds to the index i in t.
        ib = model.boundary_inds[i]
    Note that ib-1, ib+1, etc are adding to the indices in s, not in t.
    Let's also define y_ib, the depth of s_ib; d_i, the depth of t_i;
    w_i, the width of the boundary layer.
        y_ib = d_i = y_ib-1 + t_i
        y_ib+1 = d_i + w_i = y_ib-1 + t_i + w_i
    We want to keep the width of the boundary layer, w_i, constant throughout
    a single inversion.  Otherwise, we want to pin the absolute depths, y, of
    all other points in s.  That is, other than y_ib and y_ib+1 for each t_i,
    the depths, y, are immutable.  However, changing these depth points changes
    the velocity gradient on either side of them.  Therefore, changing t_i will
    affect velocities between y_ib-1 < z_a < y_ib+2 (note not inclusive!).


    We can define the model card Vsv (m = v(z)) in terms of t as follows:
      - Each point t_i refers to the thickness of the layer above a boundary
        layer, model.thickness[model.boundary_inds[i]]
      - For every point t_i, we define the depth of that point, d_i, as the
        sum of the thicknesses above it:
            np.sum(model.thickness[:model.boundary_inds[i] + 1])
            = np.sum(model.thickness[:model.boundary_inds[i]]) + t_i
      - As above, we've defined y_ib, y_ib+1, etc
      - For any depth, z_a, try to find ib s.t. y_b < z_a < y_b+1
        CONDITIONAL ON this being in the depth range y_ib-1 < z_a < y_ib+2
            (There may be no such z_a if these z points fall outside of the
            depth range of interest for this boundary layer, i)
      - Can then define v_a as
            v_a = s_b + ((s_b+1 - s_b) / (y_b+1 - y_b) * (z_a - y_b))

    Here, we are looking specifically at the values of m that are shallower than
    the s point in question, s_ib, that will still be affected by a change to
    t_i - that is, the values of z between y_ib-1 and y_ib.  So b = ib-1
            v_a = s_ib-1 + ((s_ib - s_ib-1) / (y_ib - y_ib-1) * (z_a - y_ib-1))
            v_a = s_ib-1 + ((s_ib - s_ib-1) / t_i * (z_a - y_ib-1))

    In terms of the partial derivative:
            d(v_a)/d(t_i) = -(z_a - y_ib-1) * (s_ib - s_ib-1) / t_i^2

    Here, we are calculating this one layer at a time, with the loop in
    the calling function - passing the index, i, in as an argument.  This fills
    some of a single column of dm_dt_mat ([dm_0/dt_i; ...; dm_N/dt_i]),
    although most of these values will be zero.

    Arguments:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s and km)
            - Model in layout ready for easy conversion to column vector
              to be used in least squares inversion.
        i:
            - int
            - Units:    n/a
            - Index in InversionModel for which we are calculating the column
              of partial derivatives.
        depth:
            - (n_card_depths, ) np.array
            - Units:    kilometres
            - Depth vector for MINEOS kernel.
        dm_dt_mat:
            - (n_card_depths, n_boundary_layers) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/dt that we are filling in a bit
              at a time.

    Returns:
        dm_dt_mat:
            - (n_card_depths, n_boundary_layers) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/dt with a few more values filled
              in - specifically, those in the 'i'th column (for model parameter
              t_i) in rows corresponding to depths between y_ib-1 and y_ib.

    """
    # s_ib is the velocity at the top of the boundary, model.boundary_inds[i]
    # s_ib_minus_1 is the velocity at the top of the model layer above this

    # model.thickness[i_b] is the thickness of the layer above the boundary
    # i.e. what we are inverting for; model.thickness[i_b + 1] is the thickness
    # of the boundary layer itself

    ib = model.boundary_inds[i]
    t_i = model.thickness[ib]

    y_ib_minus_1 = np.sum(model.thickness[:ib])
    y_ib = np.sum(model.thickness[:ib + 1])

    d_inds, = np.where(np.logical_and(y_ib_minus_1 < depth, depth < y_ib))

    for i_d in d_inds:
        dm_dt_mat[i_d, i] = -(
            ((model.vsv[ib] - model.vsv[ib - 1]) * (depth[i_d] - y_ib_minus_1))
            / t_i^2
        )

    return dm_dt_mat

def _convert_kernels_d_withinboundarym_by_d_t(
        model:define_models.InversionModel, i:int, depth:np.array,
        dm_dt_mat:np.array) -> np.array:
    """ Find dm/dt for the model card points within the boundary layer.

    To convert the MINEOS kernels (dc/dm) to inversion model kernels (dc/dp),
    we need to define the matrix dm/dp (dc/dp = dc/dm * dm/dp).  We divide
    our inversion model, p, into two parts: the defined velocities at various
    depths (s) and the depth and thickness of certain boundary layers of
    interest (t).

    Here, we calculate dm/dt for the model card points within the boundary
    layer.  In the following description, I'm replacing subscripts with
    underscores - everything before the next space should be subscripted.
    e.g. y_i+1 is the 'i+1'th value of y.

    The values, t_i, in the model, p, have slightly confusing effects on v(z)
    because we have to go via the effect on s, the velocities defined in p.
    Let's define ib, the index in s that corresponds to the index i in t.
        ib = model.boundary_inds[i]
    Note that ib-1, ib+1, etc are adding to the indices in s, not in t.
    Let's also define y_ib, the depth of s_ib; d_i, the depth of t_i;
    w_i, the width of the boundary layer.
        y_ib = d_i = y_ib-1 + t_i
        y_ib+1 = d_i + w_i = y_ib-1 + t_i + w_i
    We want to keep the width of the boundary layer, w_i, constant throughout
    a single inversion.  Otherwise, we want to pin the absolute depths, y, of
    all other points in s.  That is, other than y_ib and y_ib+1 for each t_i,
    the depths, y, are immutable.  However, changing these depth points changes
    the velocity gradient on either side of them.  Therefore, changing t_i will
    affect velocities between y_ib-1 < z_a < y_ib+2 (note not inclusive!).


    We can define the model card Vsv (m = v(z)) in terms of t as follows:
      - Each point t_i refers to the thickness of the layer above a boundary
        layer, model.thickness[model.boundary_inds[i]]
      - For every point t_i, we define the depth of that point, d_i, as the
        sum of the thicknesses above it:
            np.sum(model.thickness[:model.boundary_inds[i] + 1])
            = np.sum(model.thickness[:model.boundary_inds[i]]) + t_i
      - As above, we've defined y_ib, y_ib+1, etc
      - For any depth, z_a, try to find ib s.t. y_b < z_a < y_b+1
        CONDITIONAL ON this being in the depth range y_ib-1 < z_a < y_ib+2
            (There may be no such z_a if these z points fall outside of the
            depth range of interest for this boundary layer, i)
      - Can then define v_a as
            v_a = s_b + ((s_b+1 - s_b) / (y_b+1 - y_b) * (z_a - y_b))

    Here, we are looking specifically at the values of m that are within the
    boundary layer in question - that is, the values of z between y_ib and
    y_ib+1.  So ib = b in the equation above.
            v_a = s_ib + ((s_ib+1 - s_ib) / (y_ib+1 - y_ib) * (z_a - y_ib))
            v_a = s_ib + ((s_ib+1 - s_ib) / w_i * (z_a - (y_ib-1 + t_i)))

    In terms of the partial derivative:
            d(v_a)/d(t_i) = -(s_ib+1 - s_ib) / w_i

    Here, we are calculating this one layer at a time, with the loop in
    the calling function - passing the index, i, in as an argument.  This fills
    some of a single column of dm_dt_mat ([dm_0/dt_i; ...; dm_N/dt_i]),
    although most of these values will be zero.

    Arguments:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s and km)
            - Model in layout ready for easy conversion to column vector
              to be used in least squares inversion.
        i:
            - int
            - Units:    n/a
            - Index in InversionModel for which we are calculating the column
              of partial derivatives.
        depth:
            - (n_card_depths, ) np.array
            - Units:    kilometres
            - Depth vector for MINEOS kernel.
        dm_dt_mat:
            - (n_card_depths, n_boundary_layers) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/dt that we are filling in a bit
              at a time.

    Returns:
        dm_dt_mat:
            - (n_card_depths, n_boundary_layers) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/dt with a few more values filled
              in - specifically, those in the 'i'th column (for model parameter
              t_i) in rows corresponding to depths between y_ib-1 and y_ib.

    """

    ib = model.boundary_inds[i]
    w_i = model.thickness[ib + 1]

    y_ib = np.sum(model.thickness[:i_b + 1])
    y_ib_plus_1 = np.sum(model.thickness[:i_b + 2])

    d_inds, = np.where(np.logical_and(y_ib <= depth, depth < y_ib_plus_1))

    for i_d in d_inds:
        dm_dt_mat[i_d, i] = -(
            (model.vsv[i_b + 1] - model.vsv[i_b])
            / w_i
        )

    return dm_dt_mat

def _convert_kernels_d_deeperm_by_d_t(model:define_models.InversionModel,
                                      i:int, depth:np.array,
                                      dm_dt_mat:np.array) -> np.array:
    """ Find dm/dt for the model card points below the boundary layer.

    To convert the MINEOS kernels (dc/dm) to inversion model kernels (dc/dp),
    we need to define the matrix dm/dp (dc/dp = dc/dm * dm/dp).  We divide
    our inversion model, p, into two parts: the defined velocities at various
    depths (s) and the depth and thickness of certain boundary layers of
    interest (t).

    Here, we calculate dm/dt for the model card points below (deeper than)
    the boundary layer.  In the following description, I'm replacing subscripts
    with underscores - everything before the next space should be subscripted.
    e.g. y_i+1 is the 'i+1'th value of y.

    The values, t_i, in the model, p, have slightly confusing effects on v(z)
    because we have to go via the effect on s, the velocities defined in p.
    Let's define ib, the index in s that corresponds to the index i in t.
        ib = model.boundary_inds[i]
    Note that ib-1, ib+1, etc are adding to the indices in s, not in t.
    Let's also define y_ib, the depth of s_ib; d_i, the depth of t_i;
    w_i, the width of the boundary layer.
        y_ib = d_i = y_ib-1 + t_i
        y_ib+1 = d_i + w_i = y_ib-1 + t_i + w_i
    We want to keep the width of the boundary layer, w_i, constant throughout
    a single inversion.  Otherwise, we want to pin the absolute depths, y, of
    all other points in s.  That is, other than y_ib and y_ib+1 for each t_i,
    the depths, y, are immutable.  However, changing these depth points changes
    the velocity gradient on either side of them.  Therefore, changing t_i will
    affect velocities between y_ib-1 < z_a < y_ib+2 (note not inclusive!).


    We can define the model card Vsv (m = v(z)) in terms of t as follows:
      - Each point t_i refers to the thickness of the layer above a boundary
        layer, model.thickness[model.boundary_inds[i]]
      - For every point t_i, we define the depth of that point, d_i, as the
        sum of the thicknesses above it:
            np.sum(model.thickness[:model.boundary_inds[i] + 1])
            = np.sum(model.thickness[:model.boundary_inds[i]]) + t_i
      - As above, we've defined y_ib, y_ib+1, etc
      - For any depth, z_a, try to find ib s.t. y_b < z_a < y_b+1
        CONDITIONAL ON this being in the depth range y_ib-1 < z_a < y_ib+2
            (There may be no such z_a if these z points fall outside of the
            depth range of interest for this boundary layer, i)
      - Can then define v_a as
            v_a = s_b + ((s_b+1 - s_b) / (y_b+1 - y_b) * (z_a - y_b))

    Here, we are looking specifically at the values of m that are deeper than
    the boundary layer in question, s_ib, that will still be affected by a
    change to t_i - that is, the values of z between y_ib+1 and y_ib+2.
    So, from the equation above, sub in b = ib+1
            v_a = s_ib+1
                  + ((s_ib+2 - s_ib+1) / (y_ib+2 - y_ib+1) * (z_a - y_ib+1))
            v_a = s_ib+1
                  + ((s_ib+2 - s_ib+1) / (y_ib+2 - (y_ib-1 + t_i + w_i)
                     * (z_a - (y_ib-1 + t_i + w_i)))

    In terms of the partial derivative (via the chain rule & product rule):
            d(v_a)/d(t_i) = ((s_ib+2 - s_ib+1) * (z_a - y_ib+2))
                            / (t_i - y_ib+2 + y_ib-1 + w_i)^2

    DERIVATION BREAK
    For the purposes of deriving this, let's simplify terms a little bit.
        a = s_ib+2 - s_ib+1
        b = z_a - y_ib-1 - w_i
        c = y_ib+2 - y_ib-1 - w_i
        x = t_i

    Our equation is now
        d(v_a)/dx = d/dx (s_ib+1 + a * (b-x) / (c-x))
                  = d/dx (a/(c-x) * (b-x)/(c-x))
                  = d/dx (ab/(c-x) - ax/(c-x))

    The chain rule is d/dx (f(g(x)) = f'(g(x))g'(x))
        d/dx (ab/(c-x)):     g(x) = c-x      g'(x) = -1
                             f(y) = ab/y     f'(y) = -ab/y^2
        d/dx (ab/(c-x)) = ab/(c-x)^2

    The product rule is d/dx (h(x)j(x)) = h'(x)j(x) + h(x)j'(x)
        d/dx (-ax/(c-x)):   h(x) = -ax      h'(x) = -a
                            j(x) = 1/(c-x)  j'(x) = -1/(c-x)^2
        d/dx (-ax/(c-x)) = -a/(c-x) + ax/(c-x)^2
                         = (-a(c-x) + ax)/(c-x)^2
                         = -ac/(c-x)^2

    The total derivative is therefore
        d(v_a)/dx = ab/(c-x)^2 - ac/(c-x)^2
                  = a(b-c)/(c-x)^2
                  = a(b-c)/(x-c)^2

                  = (s_ib+2 - s_ib+1) * (z_a - y_ib+2)
                    / (t_i - y_ib+2 + y_ib-1 + w_i)^2

    BACK TO THE REAL DOCSTRING

    Here, we are calculating this one layer at a time, with the loop in
    the calling function - passing the index, i, in as an argument.  This fills
    some of a single column of dm_dt_mat ([dm_0/dt_i; ...; dm_N/dt_i]),
    although most of these values will be zero.

    Arguments:
        model:
            - define_models.InversionModel
            - Units:    seismological (km/s and km)
            - Model in layout ready for easy conversion to column vector
              to be used in least squares inversion.
        i:
            - int
            - Units:    n/a
            - Index in InversionModel for which we are calculating the column
              of partial derivatives.
        depth:
            - (n_card_depths, ) np.array
            - Units:    kilometres
            - Depth vector for MINEOS kernel.
        dm_dt_mat:
            - (n_card_depths, n_boundary_layers) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/dt that we are filling in a bit
              at a time.

    Returns:
        dm_dt_mat:
            - (n_card_depths, n_boundary_layers) np.array
            - Units:    assumes seismological (km/s, km)
            - Partial derivative matrix of dm/dt with a few more values filled
              in - specifically, those in the 'i'th column (for model parameter
              t_i) in rows corresponding to depths between y_ib-1 and y_ib.

    """
    # s_ib_minus_1 is the velocity at the top of the model layer above this
    # s_ib is the velocity at the top of the boundary, model.boundary_inds[i]
    # s_ib_plus_1 is the velocity at the bottom of the boundary
    # s_ib_plus_2 is the velocity at the bottom of the layer below the boundary
    # ((s_ib+2 - s_ib+1) * (z_a - y_ib+2))
    #                 / (t_i - y_ib+2 + y_ib-1 + w_i)^2

    ib = model.boundary_inds[i]
    t_i = model.thickness[ib]
    w_i = model.thickness[ib + 1]

    y_ib_minus_1 = np.sum(model.thickness[:ib])
    y_ib_plus_1 = np.sum(model.thickness[:ib + 2])
    y_i_plus_2 = np.sum(model.thickness[:i_b + 3])

    d_inds, = np.where(np.logical_and(y_i_plus_1 <= depth, depth < y_ib_plus_2))

    for i_d in d_inds:
        dm_dt_mat[i_d, i] = (
            (model.vsv[ib + 2] - model.vsv[ib + 1]) * (depth[i_d] - y_ib_plus_2)
            / (t_i - (y_ib_plus_2 - y_ib_minus_1 - w_i))^2
        )

    return dm_dt_mat


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

def _build_weighting_matrices(data:surface_waves.ObsPhaseVelocity, m0:np.array,
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
