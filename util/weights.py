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
        upper_crust:
            - (n_values_in_layer, ) np.array
        lower_crust:
            - (n_values_in_layer, ) np.array
        lithospheric_mantle:
            - (n_values_in_layer, ) np.array
        asthenosphere:
            - (n_values_in_layer, ) np.array

    """

    upper_crust: np.array
    lower_crust: np.array
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
    _set_layer_values((0.5, 0.5, 1, 2, 0), layers, damp_s, damp_t, 'roughness')
    roughness_mat, roughness_vec = _damp_constraints(
        _build_smoothing_constraints(model), damp_s, damp_t
    )

    # Linear constraint equations
    # Damp towards starting model
    _set_layer_values(
        (
            [0.5] * len(layers.upper_crust),
            [0.5] * (len(layers.lower_crust) - 1) + [0.5],
            [0.5] * (len(layers.lithospheric_mantle) - 1) + [0.5],
            [0.5] * len(layers.asthenosphere),
            [0.01, 0.01]
        ),
        layers, damp_s, damp_t, 'to_m0'
    )
    damp_to_m0_mat, damp_to_m0_vec = _damp_constraints(
        _build_constraint_damp_to_m0(p), damp_s, damp_t
    )
    # Damp towards starting model gradients in Vs
    _set_layer_values((1, 1, 1, 1, 0), layers, damp_s, damp_t, 'to_m0_grad')
    damp_to_m0_grad_mat, damp_to_m0_grad_vec = _damp_constraints(
        _build_constraint_damp_original_gradient(model), damp_s, damp_t
    )


    # Record damping parameters
    save_name = 'output/{0}/{0}'.format(setup_model.id)
    damp_s.to_csv(save_name + 'damp_s.csv', index=False)
    damp_t.to_csv(save_name + 'damp_t.csv', index=False)

    # Put all a priori constraints together
    a_priori_mat = np.vstack((
        roughness_mat,
        damp_to_m0_mat,
        damp_to_m0_grad_mat,
    ))
    a_priori_vec = np.vstack((
        roughness_vec,
        damp_to_m0_vec,
        damp_to_m0_grad_vec,
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

    This is done one model parameter at a time - i.e. assumes that H is only
    (n_constraints x n_depth_points), NOT n_model_points wide!!

    Damping parameters are set for individual layers.

    Arguments:
        H_h_label:
            - tuple of (H, h, label)
            H:
                - (n_constraint_equations, n_model_points) np.array
                - Constraints matrix in H * m = h.
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

    H, h, label = H_h_label

    # (n_constraint_equations, n_depth_points_in_layer) H
    #   * (n_depth_points_in_layer, ) layer_damping
    # As depth increases by column in H, it will do element-wise
    # multiplication with layer_damping (which is treated as a row vector)
    # For h, depth increases by row, so layer_damping is the
    # wrong orientation - have to add an axis (cannot transpose as
    # it starts off as 1D)
    all = np.concatenate((damping_s[label].values, damping_t[label].values))
    H *= all
    # h is assumed to either have n_constraints = n_model_params
    # or to be 0 always
    if h.shape[0] == H.shape[1]:
        h *= all[:, np.newaxis]

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
    obs_std[obs_std == 0] = 0.15


    return np.diag(1 / obs_std)

def _build_smoothing_constraints(
        model:define_models.InversionModel) -> (np.array, np.array):
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
        model:
            - define_models.InversionModel
            - model.vsv.size = n_depth_points
            - model.boundary_inds.size = n_boundary_layers

    Returns:
        roughness_mat
            - (n_smoothing_equations, n_model_params) np.array
            - Roughness matrix, D, in D * m = d.
            - In Menke (2012)'s notation, this is D and Wm is D^T * D.
            - This is the matrix that we multiply by the model to find the
              roughness of the model (i.e. the second derivative of the model).
        - roughness_vec
            - (n_smoothing_equations, 1) np.array
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

    # Make and edit the banded matrix (roughness equations)
    banded_matrix = _make_banded_matrix(n_depth_points, (1, -2, 1))
    t = model.thickness
    # NOTE: around the boundary layers, layer thickness isn't equal
    for ib in model.boundary_inds:
        # Smooth layers above and below BL assuming that variable thickness
        # layers won't been perturbed that much
        banded_matrix[ib - 1, ib - 1:ib + 1] = (
              [-model.thickness[ib - 1] / model.thickness[ib] - 1,
              model.thickness[ib - 1] / model.thickness[ib]])
        banded_matrix[ib + 2, ib + 1:ib + 3] = (
              [model.thickness[ib + 3] / model.thickness[ib + 2],
               -model.thickness[ib + 3] / model.thickness[ib + 2] - 1])

    # At all discontinuities (and the first and last layers of the model),
    # set the roughness_matrix to zero
    do_not_smooth = ([0, -1]
                    + list(model.boundary_inds) + list(model.boundary_inds + 1))
    banded_matrix[do_not_smooth, :] = 0
    # Add columns to roughness_matrix to get it into the right shape
    # These columns can be filled with zeros as they will be mutlipliers
    # for the parameters controlling layer size
    roughness_matrix = np.hstack(
        (banded_matrix, np.zeros((n_depth_points, model.boundary_inds.size)))
    )

    roughness_vector = np.zeros((n_depth_points, 1))

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

    This will not work around our boundary layers because the thickness of
    the layer above, the boundary layer, and the layer below can all change -
    so the gradient is dependent on more than just the values of velocity!
    However, we don't want to damp the gradient in our boundary layer that
    much, and perhaps it is ok to leave out the adjacent layers too?

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
        h:
            - (n_depth_points - 1, 1) np.array
            - Constraints vector, h, in H * m = h.
            - Set to zero always.
        'to_m0_grad'
            - str
            - Label for this constraint.
    """

    n_depth_points = model.vsv.size - 1

    H = np.zeros((n_depth_points - 1,
                  n_depth_points + model.boundary_inds.size))

    for i_d in range(n_depth_points - 1):
        # Need to transpose slice of m0 (2, 1) to fit in H (1, 2) gap,
        # because in H, depth increases with column not row.
        H[i_d, [i_d, i_d+1]] = np.hstack((
            1 / model.vsv[i_d], -1 / model.vsv[i_d + 1]
        ))
    H[-1, -1] = 1 / model.vsv[-2] - 1

    # Remove constraints around discontinuities
    do_not_damp = list(model.boundary_inds)
    H[do_not_damp, :] = 0

    h = np.zeros((n_depth_points - 1, 1))

    return H, h, 'to_m0_grad'

def _build_constraint_damp_min_thickness(setup_model, p):
    """ Damp thickness of variable layers to pre-set minimum.

    In H * m = h, we want to end up with
            exp(t_i - setup_model.boundary_depth_uncertainty[i]) = 1
            exp(setup_model.boundary_depth_uncertainty[i] - t_i) = 1

    """
