""" Make all of the weighting and damping parameters.

HOO BOY, this is going to need a huge overhaul!!  Currently not useful,
all formatted for the original (MINEOS) format of the inversion model.
"""

#import collections
import typing
import numpy as np
import pandas as pd

from util import constraints
from util import define_models

# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class ModelLayerValues(typing.NamedTuple):
    """ Values that are layer-specific.

    All fields should be an (n_values_in_layer, 1)
    array, where
        n_values_in_layer:
            Can be 1 or ModelLayerIndices.[field_name].size.
                i.e. constant within a layer, or specified for each depth point
                     in that layer.


    Fields:
        sediment:
            - (n_values_in_layer, ) np.array
        crust:
            - (n_values_in_layer, ) np.array
        lithospheric_mantle:
            - (n_values_in_layer, ) np.array
        asthenosphere:
            - (n_values_in_layer, ) np.array

    """

    sediment: np.array
    crust: np.array
    lithospheric_mantle: np.array
    asthenosphere: np.array


def build_weighting_damping(std_obs:np.array, p:np.array,
                            model:define_models.InversionModel,
                            setup_model:define_models.SetupModel):
    """
    """

    # Calculate data error matrix
    W = _build_error_weighting_matrix(std_obs)

    # Record level of damping on the Vsv model parameters
    layers = define_models._set_model_indices(setup_model, model)
    damp_s = pd.DataFrame({'Depth': layers.depth})
    # Record level of damping on the boundary layer depth parameters
    damp_t = pd.DataFrame({
        'Depth': np.cumsum(model.thickness)[list(model.boundary_inds)],
    })


    # Minimise second derivative within layers - roughness
    # Note that because we are smoothing across upper-crust to lower-crust,
    # having different values for those two layers messes with things, as does
    # having variable damping within a layer
    # Already built into roughness_mat is that we do not smooth around BLs
    sc = 0.
    _set_layer_values((sc / 5., sc / 5., sc, 2, sc), layers, damp_s, damp_t, 'roughness')
    roughness_mat, roughness_vec = _damp_constraints(
        _build_smoothing_constraints(model, setup_model), damp_s, damp_t
    )

    # Linear constraint equations
    # Damp towards starting model
    # sc = 0.
    # _set_layer_values(
    #     (
    #         [sc] * (len(layers.sediment) - 1) + [sc],
    #         [sc] * (len(layers.crust) - 1) + [sc],
    #         [sc] * (len(layers.lithospheric_mantle) - 1) + [sc],
    #         [sc] * len(layers.asthenosphere),
    #         [sc * 0.01] * len(model.boundary_inds)
    #     ),
    #     layers, damp_s, damp_t, 'to_m0'
    # )
    # damp_to_m0_mat, damp_to_m0_vec = _damp_constraints(
    #     _build_constraint_damp_to_m0(p), damp_s, damp_t
    # )

    # Damp towards starting model gradients in Vs
    # sc = 0.
    # _set_layer_values((sc, sc, sc, sc, sc), layers, damp_s, damp_t, 'to_m0_grad')
    # damp_to_m0_grad_mat, damp_to_m0_grad_vec = _damp_constraints(
    #     _build_constraint_damp_original_gradient(model), damp_s, damp_t
    # )

    # Damp towards model gradient = 0
    sc = 1
    _set_layer_values((sc, sc, sc, 0, 0), layers, damp_s, damp_t, 'to_0_grad')
    damp_to_0_grad_mat, damp_to_0_grad_vec = _damp_constraints(
        _build_constraint_damp_zero_gradient(model), damp_s, damp_t
    )


    # Record damping parameters
    save_name = 'output/{0}/{0}'.format(setup_model.id)
    damp_s.to_csv(save_name + 'damp_s.csv', index=False)
    damp_t.to_csv(save_name + 'damp_t.csv', index=False)

    # Put all a priori constraints together
    a_priori_mat = np.vstack((
        roughness_mat,
        # damp_to_m0_mat,
        damp_to_0_grad_mat,
    ))
    a_priori_vec = np.vstack((
        roughness_vec,
        # damp_to_m0_vec,
        damp_to_0_grad_vec,
    ))

    return W, a_priori_mat, a_priori_vec


def _set_layer_values(damping, layers, damp_s, damp_t, label):

    damp_s[label] = 0
    damp_t[label] = damping[-1]

    # Loop through all layers in depth
    n = 0
    for layer_name in layers.layer_names:
        if layer_name == 'boundary_layers':
            n += 1
            continue
        layer_inds = getattr(layers, layer_name).astype(int)
        damp_s.loc[layer_inds, label] =  damping[n]
        n += 1

    return


def _damp_constraints(
        H_h_label:tuple, damping_s:pd.DataFrame, damping_t:pd.DataFrame
    ):
    """ Apply layer specific damping to H and h.

    This assumes that in H, every row gives a constraint on the corresponding
    model point.  That is, if there is no constraint e.g. on gradient in a
    boundary layer, H is required to have a row of zeros corresponding to that
    particular model point so that the size is right for this scaling.

    Arguments:
        H_h_label:
            - tuple of (H, h, label)
            H:
                - (n_model_points, n_model_points) np.array
                - Constraints matrix in H * m = h.
                - MUST be a square matrix for this layer-specific scaling
                  to work!  Every row corresponds to a constraint on the
                  corresponding model parameter.
            h:
                - (n_constraint_equations, 1) np.array
                - Constraints vector in H * m = h.
            label:
                - str
                - Name for this constraint, used as column name in DataFrame.
        damping_s:
            - pd.DataFrame
                columns include Depth and [label]
                n_model_depths rows (depth corresponds to centre of layers)
            - Scaling for the damping constraint equations for model Vsv
        damping_t:
            - pd.DataFrame
                columns include Depth and [label]
                n_boundary_layers rows (depth corresponds to top of BL)
            - Scaling for the damping constraint equations for model boundary
              layer depths

        Returns:
            H:
                - (n_constraint_equations, n_depth_points) np.array
                - Damped constraints matrix in H * m = h.
            h:
                - (n_constraint_equations, 1) np.array
                - Damped constraints vector in H * m = h.
    """

    H, h, label = H_h_label

    # the values from a pd.DataFrame come out as (n_vals, ) np arrays
    all = np.concatenate((damping_s[label].values, damping_t[label].values))
    all = all[:, np.newaxis]
    H *= all
    h *= all

    return H, h

def _build_error_weighting_matrix(obs_std):
    """ Build the error weighting matrix from data standard deviation.

    This is just a diagonal matrix of the inverse of standard deviation.

    Arguments:
        obs_std:
            - (n_periods, ) np.array
            - Observed surface wave standard deviation.
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
    obs_std = obs_std.flatten().copy()
    obs_std[obs_std == 0] = 0.05


    return np.diag(1 / obs_std)

def _build_smoothing_constraints(
        model:define_models.InversionModel,
        setup_model:define_models.SetupModel) -> (np.array, np.array):
    """ Build the matrices needed to minimise second derivative of the model.

    That is, for each layer, we want to minimise the second derivative
    (i.e. ([layer+1 val] - [layer val])/(layer+1 thickness)
            - ([layer val] - [layer-1 val])/(layer thickness))
    except for where we expect there to be discontinuties.  At expected
    discontinuities, set the roughness matrix to zero.

    e.g.    (v1 - v0)/t1 - (v2 - v1)/t2 = 0
            (t2*v1 - t2*v0)/(t1*t2) - (v2*t1 - v1*t1)/(t1*t2) = 0
            -t2 * v0 + (t1 + t2) * v1 - t1 * v2 = 0

    As such, row i in H should be
                    [..., -t_i+1, t_i + t_i+1, -t_i, ...]
    (or, equivalently, [..., t_i+1, -t_i - t_i+1, t_i, ...], as the value on the other side of the equation is 0).


    Note: Josh says this can be better captured by a linear constraint
    preserving layer gradients (and therefore stopping the model from getting
    any rougher).

    Arguments:
        model:
            - define_models.InversionModel
            - model.vsv.size = n_depth_points + 1 (deepest Vs is fixed)
            - model.boundary_inds.size = n_boundary_layers
        setup_model:
            - define_models.SetupModel

    Returns:
        roughness_mat
            - (n_model_params, n_model_params) np.array
            - Roughness matrix, D, in D * m = d.
            - In Menke (2012)'s notation, this is D and Wm is D^T * D.
            - This is the matrix that we multiply by the model to find the
              roughness of the model (i.e. the second derivative of the model).
        - roughness_vec
            - (n_model_params, 1) np.array
            - Roughness vector, d, in D * m = d.
            - This is the permitted roughness of the model.
            - Actual roughness is calculated as roughness_matrix * model, and
              by setting this equal to the permitted roughness in the damped
              least squares inversion, we will tend towards the solution of
              this permitted roughness (i.e. smoothness)
            - Because we want as smooth a model as possible, permitted
              roughness is set to zero always.
        - 'roughness'
            - str
            - Label for this constraint.

    """

    n_depth_points = model.vsv.size - 1
    n_BLs = model.boundary_inds.size

    roughness_vector = np.zeros((n_depth_points + n_BLs, 1))

    # Make and edit the banded matrix (roughness equations)
    banded_matrix = np.zeros((n_depth_points, n_depth_points))
    t = model.thickness
    # NOTE: around the boundary layers, layer thickness isn't equal
    for ir in range(1, n_depth_points - 1):
        # Smooth layers above and below BL assuming that variable thickness
        # layers won't been perturbed that much
        banded_matrix[ir, ir - 1:ir + 2] = ([
            t[ir + 1], -np.sum(t[ir:ir + 2]), t[ir]
        ])

    # Add smoothing at base of the model
    base_t = list(sum(t)); base_vs = list(model.vsv[-1])
    define_models._fill_in_base_of_model(base_t, base_vs, setup_model._replace(
        depth_limits=(base_t[0], base_t[0] + 10) # Interpolate 10 km below base
    ))
    ir += 1
    banded_matrix[ir, ir - 1: ir + 1] = ([base_t[-1], -t[-1] - base_t[-1]])
    roughness_vector[ir] = -t[-1] * base_vs[-1]

    # At all discontinuities, reduce the smoothing constraints
    #smooth_less = list(model.boundary_inds) + list(model.boundary_inds + 1)
    #banded_matrix[smooth_less, :] *= 0.
    # At all discontinuties, replace the smoothing constraints with a repeat
    # of smoothing the layer above and below the discontinuity
    banded_matrix[model.boundary_inds, :] = banded_matrix[model.boundary_inds - 1, :]
    banded_matrix[model.boundary_inds + 1, :] = banded_matrix[model.boundary_inds + 2, :]


    # Add columns to roughness_matrix to get it into the right shape
    # These columns can be filled with zeros as they will be mutlipliers
    # for the parameters controlling layer size
    roughness_matrix = np.hstack(
        (banded_matrix, np.zeros((n_depth_points, n_BLs)))
    )

    # Add rows to the roughness matrix to get it into the right shape
    roughness_matrix = np.vstack(
        (roughness_matrix, np.zeros((n_BLs, n_depth_points + n_BLs)))
    )

    return roughness_matrix, roughness_vector, 'roughness'


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


def _build_constraint_damp_to_m0(m:np.array) -> (np.array, np.array):
    """ Damp towards starting model.

    In H * m = h, we want to end up with
                    m_val_new = m_val_old
    As such, H is a diagonal matrix of ones, and h is made up of the starting
    model values.

    Arguments:
        m:
            - (n_model_points, 1) np.array
            - Units:    seismological
            - Column vector of model that goes into least squares equation

    Returns:
        H:
            - (n_model_points, n_model_points) np.array
            - Constraints matrix, H, in H * m = h.
            - As we are damping towards the starting model, this is a diagonal
              matrix of ones.
        h:
            - (n_model_points, 1) np.array
            - Constraints vector, h, in H * m = h.
            - As we are damping towards the starting model, this is the
              just the starting model.
        'to_m0'
            - str
            - Label for this constraint.
    """

    H = np.diag(np.ones(len(m)))
    # m is already a column vector, so can just pull relevant indices from it
    h = m

    return H, h, 'to_m0'

def _build_constraint_damp_original_gradient(
        model:define_models.InversionModel):
    """ Damp Vsv gradients between layers to starting model values.

    In H * m = h, we want to end up with
            V_new_node1/V_old_node1 - V_new_node2/V_old_node2 = 0
        (Via... V_new_node1/V_new_node2 = V_old_node1/V_old_node2)
    As such, H is set to 1/V_node1 in the node 1 column, and to
    1/V_node2 in the node 2 column; h is zero always.

    Note that this only considers a single gradient between two adjacent Vs
    points at a time, so having adjacent layers of different thickness is
    not an issue - all of this damping is for a single layer thickness.

    However, if layer thickness can change (i.e. the layer above and the layer
    below our boundary layer, t[model.bi] and t[model.bi + 2]), there is an
    issue.  We're going to assume that the change in layer thickness is
    going to be relatively small - perhaps this is ok??

    Arguments:
        model:
            - InversionModel
            - Units:    seismological

    Returns:
        H:
            - (n_depth_points - 1, n_model_points) np.array
            - Constraints matrix, H, in H * m = h.
            - Set to to 1/V_node1 in the node 1 column, and to
              1/V_node2 in the node 2 column.
            - Note that only (n_depth_points - 1) rows in this matrix,
              as every constraint is for two adjacent depth points.
            - As last velocity point is fixed, to constrain the gradient at the
              base of the model is just to damp to the last alterable value of
              vs, i.e. last column for Vs should be
                    1/V_node_end * V_node_end_new = 1
                    H[-1, -1] * m[-1] = h[-1]
                        - note this notation assumes no BLs at the end of m
        h:
            - (n_depth_points - 1, 1) np.array
            - Constraints vector, h, in H * m = h.
            - Set to zero always.
        'to_m0_grad'
            - str
            - Label for this constraint.
    """

    n_depth_points = model.vsv.size - 1 # number of Vs nodes inverted for
    n_model_points = n_depth_points + model.boundary_inds.size
    H = np.zeros((n_model_points, n_model_points))

    for i_d in range(n_depth_points - 1):
        # Need to transpose slice of m0 (2, 1) to fit in H (1, 2) gap,
        # because in H, depth increases with column not row.
        H[i_d, [i_d, i_d+1]] = np.hstack((
            1 / model.vsv[i_d], -1 / model.vsv[i_d + 1]
        ))

    H[n_depth_points - 1, n_depth_points - 1] = 1. / model.vsv[n_depth_points - 1]

    h = np.zeros((n_model_points, 1))
    i = model.boundary_inds.size + 1
    h[-i] = 1  # to apply damping equally, need to weight last inverted Vs
               # point to m0 rather than m0 grad - deepest Vs is fixed

    return H, h, 'to_m0_grad'

def _build_constraint_damp_zero_gradient(
        model:define_models.InversionModel):
    """ Damp Vsv gradients between layers to zero.

    In H * m = h, we want to end up with
            V_new_node1 - V_new_node0 = 0
        (Via... (V_new_node1 - V_new_node0)/ t0 = 0)
    As such, H is set to 1 in the node 1 column, and to -1 in the node 2 column;
    h is zero always.

    Note that this only considers a single gradient between two adjacent Vs
    points at a time, so having adjacent layers of different thickness is
    not an issue - all of this damping is for a single layer thickness.

    However, if layer thickness can change (i.e. the layer above and the layer
    below our boundary layer, t[model.bi] and t[model.bi + 2]), there is an
    issue.  We're going to assume that the change in layer thickness is
    going to be relatively small - perhaps this is ok??

    Arguments:
        model:
            - InversionModel
            - Units:    seismological

    Returns:
        H:
            - (n_depth_points - 1, n_model_points) np.array
            - Constraints matrix, H, in H * m = h.
            - Set to to 1 in the node 1 column, and to -1 in the node 2 column.
            - Note that only (n_depth_points - 1) rows in this matrix,
              as every constraint is for two adjacent depth points.
            - As last velocity point is fixed, to constrain the gradient at the
              base of the model is just to damp to the last alterable value of
              vs, i.e. last column for Vs should be
                    V_node_end_new = V_at_base
        h:
            - (n_depth_points - 1, 1) np.array
            - Constraints vector, h, in H * m = h.
            - Set to zero except for last point.
        'to_0_grad'
            - str
            - Label for this constraint.
    """

    n_depth_points = model.vsv.size - 1 # number of Vs nodes inverted for
    n_model_points = n_depth_points + model.boundary_inds.size
    H = np.zeros((n_model_points, n_model_points))

    for i_d in range(n_depth_points - 1):
        if i_d not in model.boundary_inds:
            H[i_d, [i_d, i_d+1]] = np.array([1, -1])

    H[i_d + 1, i_d + 1] = 1

    h = np.zeros((n_model_points, 1))
    h[i_d + 1] = model.vsv[-1] # to apply damping evenly, need to weight
                               # last inverted Vs to fixed deepest Vs

    return H, h, 'to_0_grad'

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
#       a.shape | b.shape | c.shape      | Result
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
