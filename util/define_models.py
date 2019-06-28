""" Generate Earth models to work with surface_waves

Classes:
    InversionModel - Vsv at certain depths
    MINEOSModel - Vsv, Vsh, Vpv, Vph, Eta finely sampled in radius

Functions:



"""


#import collections
import typing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class SetupModel(typing.NamedTuple):
    """ Vsv model interpolated between defined values at given depths.

    The inversion model is made up of a vector of Vsv values at certain
    points, s.  The depth of certain layers (e.g. Moho, LAB) is also allowed
    to vary - the indices of these layers are given in .boundary_inds - by
    varying the thickness of the overlying layers, t.
    Thus, the actual model that goes into the least squares inversion,
    m = [s; t] = np.vstack((InversionModel.vsv,
                    InversionModel.thickness[InversionModel.boundary_inds -1]))

    We also set a linear scaling from Vsv to Vsh, from Vsv to Vpv, and from
    Vpv to Vph, and assume a constant value of Eta.

    Fields:
        vsv:
            - (n_boundary_depths_inverted_for * 2 + 2, ) np.array
            - Units:    km/s
            - Shear velocity at the surface, the top and bottom of boundaries
              of interest, and at the base of the model (pinned)
            - Velocities are assumed to be piecewise linear between these
              points.
        boundary_widths:
            - (n_boundary_depths_inverted_for, ) np.array
            - Units:   km
            - Width of the layer the boundaries of interest (i.e. Moho, LAB)
              in the model, fixed for a given inversion.
        boundary_depths:
            - (n_boundary_depths_inverted_for, ) np.array
            - Units:   km
            - Depth to the top of the boundaries of interest from a priori
              constraints (i.e. receiver functions).
        boundary_depth_uncertainty:
            - (n_boundary_depths_inverted_for, ) np.array
            - Units:   km
            - Uncertainty on the depth of the boundaries of interest from a
              priori constraints.
            - Used for setting up model layers and in weighting the constraints
              in the inversion.
        base_of_model_depth:
            - float
            - Units:    km
            - Base of the model that we are inverting for.
            - Beyond this depth, the model is fixed to our starting MINEOS
              model card (which extends throughout the whole Earth).
            - Note that .vsv[-1] is the shear velocity at this point, which
              is pinned to the value in the MINEOS model card for a seamless
              transition.
        vsv_vsh_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vsv to Vsh, default value = 1 (i.e. radial isotropy)
        vpv_vsv_ratio:
            - float   ***** Perhaps worth changing in crust vs mantle? *****
            - Units:    dimensionless
            - Ratio of Vpv to Vsv, default value = 1.75
        vpv_vph_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vpv to Vph, default value = 1 (i.e. radial isotropy)
        Moho_depth:
            - float
            - Units:    km
            - Crustal thickness - required for density scaling.
        min_layer_thickness:
            - float
            - Units:    km
            - Minimum thickness of the layer, should cover several knots in the
              MINEOS model card.

    """
    vsv: np.array
    boundary_widths: np.array
    boundary_depths: np.array
    boundary_depth_uncertainty: np.array
    base_of_model_depth: float
    min_layer_thickness: float
    Moho_depth: float
    vsv_vsh_ratio: float = 1.
    vpv_vsv_ratio: float = 1.75
    vpv_vph_ratio: float = 1.


class InversionModel(typing.NamedTuple):
    """ Model that will actually go into the inversion.

    Fields:
        vsv:
            - (n_layers, 1) np.array
            - Units:    km/s
            - Shear velocity at top of layer in the model.
            - Velocities are assumed to be piecewise linear.
        thickness:
            - (n_layers, 1) np.array
            - Units:    km
            - Thickness of layer above defined vsv point, such that
              depth of .vsv[i] point is at sum(thickness[:i]) km
            - That is, as the first .vsv point is defined at the surface, the
              first value of .thickness will be 0 always.
        boundary_inds:
            - (n_boundary_depths_inverted_for, ) np.array of integers
            - Units:    n/a
            - Indices in .vsv and .thickness identifying the boundaries of
              special interest, e.g. Moho, LAB.  For these boundaries, we
              will want to specifically prescribe the width of the layer
              (given in SetupModel.boundary_widths), and to invert for the
              thickness of the overlying layer (i.e. the depth to the top
              of this boundary).
            - That is, InversionModel.(vsv|thickness)[boundary_inds[i]] is
              the velocity at the top of the boundary and the thickness of the
              layer above it, defining depth.
            - InversionModel.(vsv|thickness)[boundary_inds[i + 1]] is the
              velocity at the bottom of the boundary and the thickness of the
              layer boundary itself, prescribed for an inversion run.


    """
    vsv: np.array
    thickness: np.array
    boundary_inds: np.array

class MINEOSCardModel(typing.NamedTuple):
    """ Model that is used for all the MINEOS interfacing.

    MINEOS requires radius, rho, vpv, vsv, vph, vsh, bulk and shear Q, and eta.
    Rows are ordered by increasing radius.  There should be some reference
    PREM MINEOS card that can be loaded in and have this pasted on the bottom
    for using with MINEOS, as MINEOS requires a card that goes all the way to
    the centre of the Earth.
    """
    radius: np.array
    rho: np.array
    vpv: np.array
    vsv: np.array
    q_kappa: np.array
    q_mu: np.array
    vph: np.array
    vsh: np.array
    eta: np.array



def setup_starting_model(setup_model):
    """ Convert from SetupModel to InversionModel.

    SetupModel is the bare bones of what we want to constrain for the starting
    model, which is in a different format to the model that we actually want
    to invert, m = np.vstack(
                    (InversionModel.vsv,
                     InversionModel.thickness[InversionModel.boundary_inds)
                     )

    Calculate appropriate layer thicknesses such that the inversion will have
    all the required flexibility when inverting for the depth of the
    boundaries of interest.  Starting model Vs is kind of just randomly bodged
    here, but that is probably ok as we will be inverting for all Vs points.

    Arguments:
        setup_model:
            - SetupModel
            - Units:    seismological, i.e. km, km/s
            - Starting model, defined elsewhere

    Returns:
        inversion_model:
            - InversionModel
            - Units:    seismological, i.e. km, km/s
            - Model primed for use in the inversion.
    """

    n_layers = setup_model.vsv.size
    n_bounds = setup_model.boundary_widths.size

    thickness = [0] # first point is at the surface
    vsv = [setup_model.vsv[0]]
    boundary_inds = []
    for i_b in range(n_bounds):
        # boundary[i_b] is our boundary of interest
        vsv_at_top = setup_model.vsv[(i_b + 1) * 2 -1]
        vsv_at_base = setup_model.vsv[(i_b + 1) * 2]
        top_of_layer = (setup_model.boundary_depths[i_b]
                        - setup_model.boundary_widths[i_b]/2)
        bottom_of_layer = top_of_layer + setup_model.boundary_widths[i_b]

        # Overlying layer is pinned in depth at the top but not the bottom,
        # so the thickness of the overlying layer defines the depth to the
        # boundary.
        top_of_layer_above = (top_of_layer
                              - setup_model.boundary_depth_uncertainty[i_b]
                              - setup_model.min_layer_thickness)
        bottom_of_layer_below = (bottom_of_layer
                              + setup_model.boundary_depth_uncertainty[i_b]
                              + setup_model.min_layer_thickness)

        dist_between_boundaries = top_of_layer - sum(thickness)
        n_layers_above = max((1,
                            int((top_of_layer_above - sum(thickness))
                                // setup_model.min_layer_thickness)
                            ))
        thick_layers_above = ((top_of_layer_above - sum(thickness))
                             / n_layers_above)
        vsv_grad_above = (vsv_at_top - vsv[-1]) / dist_between_boundaries

        for n in range(n_layers_above):
            vsv += [vsv[-1] + vsv_grad_above * thick_layers_above]
        vsv += [vsv_at_top, vsv_at_base, vsv_at_base]

        thickness += (
            [(top_of_layer_above - sum(thickness)) / n_layers_above] * n_layers_above
            + [top_of_layer - top_of_layer_above]
            + [setup_model.boundary_widths[i_b]]
            + [bottom_of_layer_below - bottom_of_layer]
        )

        # Retrieve boundary index,
        # i.e. thickness index for [top_of_layer - top_of_layer_above] layer
        boundary_inds += [len(thickness) - 3]

    # And add on everything to the base of the model
    dist_to_bottom = setup_model.base_of_model_depth - sum(thickness)
    n_layers_below = max((1,
                          int(dist_to_bottom // setup_model.min_layer_thickness)
                        ))
    thick_layers_below = dist_to_bottom / n_layers_below
    vsv_grad_below = (setup_model.vsv[-1] - vsv[-1]) / dist_to_bottom
    for n in range(n_layers_below):
        vsv += [vsv[-1] + vsv_grad_below * thick_layers_below]
    #vsv += [setup_model.vsv[-1]]
    thickness += [thick_layers_below] * n_layers_below

    return InversionModel(vsv = np.array(vsv)[np.newaxis].T,
                          thickness = np.array(thickness)[np.newaxis].T,
                          boundary_inds = np.array(boundary_inds))

def convert_inversion_model_to_mineos_model(inversion_model, setup_model):
    """
    """

    # Load PREM (http://ds.iris.edu/ds/products/emc-prem/)
    # Slightly edited to remove the water layer and give the model point
    # at 24 km depth lower crustal parameter values.
    prem = pd.read_csv('./data/earth_models/prem.csv', header=None)


    radius_Earth = 6371.
    radius_bottom_model = radius_Earth - setup_model.base_of_model_depth
    step = setup_model.min_layer_thickness / 3
    radius = np.arange(radius_bottom_model, radius_Earth, step)
    radius = np.append(radius, radius_Earth)
    depth = (radius_Earth - radius) # still in km at this point
    radius *= 1e3 # convert to SI

    vsv = np.interp(depth,
                    np.cumsum(inversion_model.thickness),
                    inversion_model.vsv.flatten()) * 1e3 # convert to SI
    vsh = vsv / setup_model.vsv_vsh_ratio
    vpv = vsv / setup_model.vpv_vsv_ratio
    vph = vpv / setup_model.vpv_vph_ratio
    eta = np.ones(vsv.shape)
    q_mu = np.interp(radius, prem['radius'], prem['q_mu'])
    q_kappa = np.interp(radius, prem['radius'], prem['q_kappa'])
    rho = np.interp(radius, prem['radius'], prem['q_kappa'])

    # Now paste the two models together, with 100 km of smoothing between them
    model = pd.concat([
                prem[prem['radius'] < radius[0]],
                pd.DataFrame({
                    'radius': radius,
                    'rho': rho,
                    'vpv': vpv,
                    'vsv': vsv,
                    'q_kappa': q_kappa,
                    'q_mu': q_mu,
                    'vph': vph,
                    'vsh': vsh,
                    'eta': eta,
                })
    smoothing_z = 100
    #prem[prem['radius'] ]





        # radius: np.array
        # rho: np.array
        # vpv: np.array
        # vsv: np.array
        # q_kappa: np.array
        # q_mu: np.array
        # vph: np.array
        # vsh: np.array
        # eta: np.array
