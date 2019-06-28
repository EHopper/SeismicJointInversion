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
            - (n_boundary_depths_inverted_for * 2, ) np.array
            - Units:    km/s
            - Shear velocity at top and bottom of boundaries of interest
            - For the starting model, we assume velocities are linear within
              the boundaries of interest, and constant outside of them.
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
    vsv_vsh_ratio: float = 1.
    vpv_vsv_ratio: float = 1.75
    vpv_vph_ratio: float = 1.

def InversionModel(typing.NamedTuple):
    """ Model that will actually go into the inversion.

    Fields:
        vsv:
            - (n_layers, ) np.array
            - Units:    km/s
            - Shear velocity at top of layer in the model.
            - Velocities are assumed to be piecewise linear.
        thickness:
            - (n_layers, ) np.array
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
            - That is, InversionModel.(vsv|thickness)[boundary_inds[i]] is the
              velocity at the top of the boundary and the thickness of the
              layer above it, defining depth.
            - InversionModel.(vsv|thickness)[boundary_inds[i + 1]] is the
              velocity at the bottom of the boundary and the thickness of the
              layer boundary itself, prescribed for an inversion run.


    """
    vsv: np.array
    thickness: np.array
    boundary_inds: np.array

def setup_starting_model(setup_model):
    """
    """

    n_layers = setup_model.vsv.size
    n_bounds = setup_model.boundary_widths.size

    thickness = [0] # first point is at the surface
    vsv = [setup_model.vsv[0]]
    for i_b in range(n_bounds):
        top_of_layer = (setup_model.boundary_depths[i_b]
                        - setup_model.boundary_widths[i_b]/2)
        # Overlying layer is pinned in depth at the top but not the bottom,
        # so the thickness of the overlying layer defines the depth to the
        # boundary.
        top_of_layer_above = (top_of_layer
                              - setup_model.boundary_depth_uncertainty[i_b]*2
                              - setup_model.minimum_layer_thickness)

        n_layers_above = ((top_of_layer_above - bottom_of_previous_boundary)
                          // setup_model.minimum_layer_thickness)

        thickness += ([top_of_layer_above / n_layers_above] * n_layers_above
                      + [top_of_layer_above - top_of_layer]
                      + [setup_model.boundary_widths[i_b]])
        vsv += ([setup_model.vsv[i_b] * n_layers_above])




    pass
