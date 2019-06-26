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

class InversionModel(typing.NamedTuple):
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
            - (n_layers, ) np.array
            - Units:    km/s
            - Shear velocity at top of layer in the model
            - Velocities are piece-wise linearly interpolated between these
              defined values.
        boundary_widths:
            -
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





      thickness:
          - (n_layers, ) np.array
          - Units:    km
          - Thickness of layer above defined vsv point, such that
            depth of .vsv[i] point is at sum(thickness[:i]) km
          - That is, if the first .vsv point is defined at the surface, the
            first value of .thickness will be 0.

        boundary_inds:
            - (n_layer_depths_to_invert, ) np.array of integers
            - Units:    n/a
            - Indices in .vsv and .thickness identifying the boundaries of
              special interest, e.g. Moho, LAB.  For these boundaries, we
              will want to specifically prescribe the width of the layer,
              and to invert for the thickness of the overlying layer (i.e. the
              depth to the top of this boundary).


    """
    vsv: np.array
    boundary_inds: np.array
