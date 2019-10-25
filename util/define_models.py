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
import os

from util import constraints

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
        id:
            - str
            - Unique name of model for saving things.
        boundaries:
            - tuple of length 3, where each item in the tuple is a tuple or list
              with (n_boundary_depths_inverted_for) items in it
            - Items in tuple:
                boundary names:
                    - Units: none
                    - Labels for each of the boundaries
                    - Default = ('Moho', 'LAB')
                boundary widths:
                    - Units: kilometres
                    - (Fixed) width of the boundary layer.  This is fixed for a given
                      inversion (though we will allow the depth of the boundary layer
                      as a whole to change).
                    - Default = [3., 10.]
                boundary dVs:
                    - Units: dimensionless
                    - Fractional change in Vs at each of the boundaries,
                    - Default = [0.05, -0.02]
        depth_limits:
            - (length 2) tuple
            - Units:    km
            - Top and base of the model that we are inverting for.
            - Outside of this range, the model is fixed to our starting MINEOS
              model card (which extends throughout the whole Earth).
            - Default value = (0, 300)
        min_layer_thickness:
            - float
            - Units:    km
            - Default value: 6
            - Minimum thickness of the layer, should cover several (three)
              knots in the MINEOS model card.
        vsv_vsh_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vsv to Vsh, default value = 1 (i.e. radial isotropy)
        vpv_vsv_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vpv to Vsv, default value = 1.75
        vpv_vph_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vpv to Vph, default value = 1 (i.e. radial isotropy)
        ref_card_csv_name:
            - str
            - Default value: 'data/earth_model/prem.csv'
            - Path to a .csv file containing the information for the reference
              full MINEOS model card that we'll be altering.
            - This could be some reference Earth model (e.g. PREM), or some
              more specific local model.
            - Note that this csv file should be in SI units (m, kg.m^-3, etc)

    """

    id: str
    boundaries: tuple = (('Moho', 'LAB'), [3., 10.], [0.05, -0.02])
    depth_limits: tuple = (0, 300)
    min_layer_thickness: float = 6.
    vsv_vsh_ratio: float = 1.
    vpv_vsv_ratio: float = 1.75
    vpv_vph_ratio: float = 1.
    ref_card_csv_name: str = 'data/earth_models/prem.csv'




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
              depth of .vsv[i] point is at sum(thickness[:i+1]) km.
                Note that this means the sum of thicknesses up to the ith point.
            - That is, as the first .vsv point is defined at the surface, the
              first value of .thickness will be 0 always.
        boundary_inds:
            - (n_boundary_depths_inverted_for, ) np.array of integers
            - Units:    n/a
            - Indices in .vsv and .thickness identifying the boundaries of
              special interest, e.g. Moho, LAB.  For these boundaries, we
              will want to specifically prescribe the width of the layer
              (originally set as SetupModel.boundary_widths), and to invert for
              the thickness of the overlying layer (i.e. the depth to the top
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


class ModelLayerIndices(typing.NamedTuple):
    """ Parameters used for smoothing.

    Fields:
        sediment:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for sedimentary layer.
        crust:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for crust.
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

    sediment: np.array
    crust: np.array
    lithospheric_mantle: np.array
    asthenosphere: np.array
    boundary_layers: np.array
    depth: np.array
    layer_names: list = ['sediment', 'crust',
                         'lithospheric_mantle', 'asthenosphere',
                         'boundary_layers']


# =============================================================================
#       Set up the models for various stages of the calculation
# =============================================================================


def setup_starting_model(setup_model, location):
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
        location:
            - tuple
            - Units:    degrees latitude (N), degrees longitude (E)
            - Location for constraints - used here to put in appropriate
              crustal structure, Moho, and LAB

    Returns:
        inversion_model:
            - InversionModel
            - Units:    seismological, i.e. km, km/s
            - Model primed for use in the inversion.
    """
    # Set up directory to save to
    if not os.path.exists('output/' + setup_model.id):
        os.mkdir('output/' + setup_model.id)
    else:
        print('This model ID has already been used!')

    bnames, bwidths, bdvs = setup_model.boundaries
    n_bounds = len(bnames)
    boundary_inds = []

    # Load in Crust 1.0 crustal structure, defined globally (but coarsely)
    t, vs = constraints.get_vels_Crust1(location)

    # Load in observed constraints
    rfs = constraints._extract_rf_constraints(
        location, setup_model.id, setup_model.boundaries,
        setup_model.vpv_vsv_ratio
    )

    # Check if we are inverting for the Moho & correct Crust 1 accordingly
    if 'Moho' in bnames:
        Moho_ind = bnames.index('Moho')
        Moho_depth_ish = int(rfs.loc[Moho_ind, 'tt'] * 3.5)
        if Moho_depth_ish > sum(t[:-1]):
            t[-1] = Moho_depth_ish - sum(t[:-1])

        # FOR NOW, assuming that Moho_ind = 0, i.e. no BLs within the crust!
        boundary_inds += [len(t) - 1]
        t += [bwidths[Moho_ind]]
        vs += [vs[-1] * (1 + bdvs[Moho_ind])]
    else:
        Moho_ind = -1

    # Add in all BLs beneath the Moho (i.e. LAB)
    for i_b in range(Moho_ind + 1, n_bounds):
        # boundary[i_b] is our boundary of interest
        bound_depth_ish = np.round(rfs.loc[i_b, 'tt'] * 4.2)

        boundary_inds += [len(t)]
        # Top of the boundary layer
        t += [bound_depth_ish - sum(t)]
        vs += [vs[-1]]
        # Bottom of the boundary layer
        t += [bwidths[i_b]]
        vs += [vs[-1] * (1 + bdvs[i_b])]

    # And fill in to the base of the model
    ref_model =  pd.read_csv(setup_model.ref_card_csv_name)
    ref_depth = (ref_model['radius'].iloc[-1] - ref_model['radius']) * 1e-3
    ref_vs = ref_model['vsv'] * 1e-3
    # Remove discontinuities and make depth increasing for purposes of interp
    ref_depth[np.append(np.diff(ref_depth), 0) == 0] += 0.01
    ref_depth = ref_depth.iloc[::-1].values
    ref_vs = ref_vs.iloc[::-1].values

    t += [setup_model.depth_limits[1] - sum(t)]
    vs += [np.interp(setup_model.depth_limits[1], ref_depth, ref_vs)]
    return t, vs, boundary_inds

    t, vs, boundary_inds = _return_evenly_spaced_model(t, vs, boundary_inds)

    return InversionModel(vsv = np.array(vsv)[np.newaxis].T,
                          thickness = np.array(thickness)[np.newaxis].T,
                          boundary_inds = np.array(boundary_inds))

def _return_evenly_spaced_model(t, vs, boundary_inds, min_layer_thickness):
    """
    Important to keep the boundary layer thicknesses the same.
    """

    previous_boundary_depth = 0
    new_t = [0]
    new_vs = []
    new_bi = []
    for ib in boundary_inds:
        boundary_depth = sum(t[:ib + 1])
        inter_boundary_depth = boundary_depth - previous_boundary_depth
        n_layers = inter_boundary_depth // min_layer_thickness
        layer_t = inter_boundary_depth / n_layers
        ts = [layer_t] * int(n_layers)

        for il in range(len(ts) - 1):
            if il == 0:
                new_vs += [_mean_val_in_interval(vs, t, sum(new_t),
                                                 sum(new_t) + layer_t / 2)]
                new_t += ts
            else:
                d = np.cumsum(ts[:il + 1]) + previous_boundary_depth
                new_vs += [_mean_val_in_interval(vs, t, d[il] - layer_t / 2,
                                                 d[il] + layer_t / 2)]

        new_bi += [len(new_t) - 1]
        new_t += [t[ib + 1]]
        new_vs += [vs[ib:ib + 2]]
        previous_boundary_depth = sum(new_t)

    return new_t, new_vs, new_bi

def _mean_val_in_interval(v, t, d1, d2):
    """
    """
    interp_d = np.arange(t[0], sum(t), 0.1)
    interp_vs = np.interp(interp_d, np.cumsum(t), v)

    return np.mean(interp_vs[np.logical_and(
                d1 <= interp_d, interp_d < d2
            )])

def convert_inversion_model_to_mineos_model(inversion_model, setup_model,
                                            **kwargs):
    """ Generate model that is used for all the MINEOS interfacing.

    MINEOS requires radius, rho, vpv, vsv, vph, vsh, bulk and shear Q, and eta.
    Rows are ordered by increasing radius.  There should be some reference
    MINEOS card that can be loaded in and have this pasted on the bottom
    for using with MINEOS, as MINEOS requires a card that goes all the way to
    the centre of the Earth.

    Arguments:
        - inversion_model:
            - InversionModel
            - Units:    seismological
        - setup_model:
            - SetupModel
            - Units:    seismological
        - kwargs
            - key word argument of format:
                - Moho = some_float
            - Units:    km
            - Depth of the Moho - needed to define density structure
    """
    if not os.path.exists('output/' + setup_model.id):
        os.mkdir('output/' + setup_model.id)
        print("This test ID hasn't been used before!")

    # Load PREM (http://ds.iris.edu/ds/products/emc-prem/)
    # Slightly edited to remove the water layer and give the model point
    # at 24 km depth lower crustal parameter values.
    ref_model =  pd.read_csv(setup_model.ref_card_csv_name)

    radius_Earth = ref_model['radius'].iloc[-1] * 1e-3
    radius_model_top = radius_Earth - setup_model.depth_limits[0]
    radius_model_base = radius_Earth - setup_model.depth_limits[1]
    step = setup_model.min_layer_thickness / 3
    radius = np.arange(radius_model_base, radius_model_top, step)
    radius = np.append(radius, radius_model_top)
    depth = (radius_Earth - radius) # still in km at this point
    radius *= 1e3 # convert to SI

    vsv = np.interp(depth,
                    np.cumsum(inversion_model.thickness),
                    inversion_model.vsv.flatten()) * 1e3 # convert to SI
    vsh = vsv / setup_model.vsv_vsh_ratio
    vpv = vsv * setup_model.vpv_vsv_ratio
    vph = vpv / setup_model.vpv_vph_ratio
    eta = np.ones(vsv.shape)
    q_mu = np.interp(radius, ref_model['radius'], ref_model['q_mu'])
    q_kappa = np.interp(radius, ref_model['radius'], ref_model['q_kappa'])
    rho = np.interp(radius, ref_model['radius'], ref_model['rho'])
    if 'Moho' in setup_model.boundary_names:
        Moho_ind = (inversion_model.boundary_inds[
                        setup_model.boundary_names.index('Moho')])
        Moho_depth = np.sum(inversion_model.thickness[:Moho_ind + 1])
    else:
        try:
            Moho_depth = kwargs['Moho']
        except:
            print('Moho depth never specified! Guessing 35 km.')
            Moho_depth = 35
    rho[(depth <= Moho_depth) & (2900 < rho)] = 2900

    # Now paste the models together, with 100 km of smoothing between
    new_model = pd.DataFrame({
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
    smoothed_below = smooth_to_ref_model_below(ref_model, new_model)
    smoothed_above = smooth_to_ref_model_above(ref_model, new_model)

    mineos_card_model = pd.concat([smoothed_below, new_model,
                                   smoothed_above]).reset_index(drop=True)
    mineos_card_model.to_csv('output/{0}/{0}.csv'.format(setup_model.id),
                             index=False)

    _write_mineos_card(mineos_card_model, setup_model.id)

    return mineos_card_model

def _write_mineos_card(mineos_card_model:pd.DataFrame, name:str):
    """ Write the MINEOS card model txt file to (name).card.

    Given a pandas DataFrame with the following columns:
        radius, rho, vpv, vsv, q_kappa, q_mu, vph, vsh, eta
    write a model card text file to output/(name)/(name).card.

    All of the values in the input DataFrame are assumed to have the correct
    units etc.  This code will work out how many inner core layers and total
    core layers there are for the header of the model card.

    The layout of the card file:
    First line:
                    name of the model card
    Second line:
                    if_anisotropic   t_ref   if_deck
        This is hardwired with if_anisotropic = 1, i.e. assume anisotropy
              (even if not truly anisotropic)
        tref is the reference period for dispersion calculation,
              However, we correct for dispersion separately, so setting it to
              < 1 means no dispersion corrections are done at this stage
        NOTE: If set tref > 0, MINEOS will break when running the Q correction
              through mineos_qcorrectphv, as the automatically generated
              input file assumes tref < 0, so it enters an if statement
              in the Fortran file that requires 'y'.  If tref > 0, having
              y in the runfile breaks the code.
        if_deck set to 1 for a model card or 0 for a polynomial model
    Third line:
                    total_layers, index_top_of_inner_core, i_top_of_outer_core
    Rest of file (each line is a depth point):
                    radius  rho  vpv  vsv  q_kappa  q_mu  vph  vsh  eta
    """

    # Write MINEOS model to .card (txt) file
    # Find the values for the header line
    outer_core = mineos_card_model[(mineos_card_model.vsv == 0)
                                   & (mineos_card_model.q_mu == 0)]
    n_inner_core_layers = outer_core.iloc[[0]].index[0]
    n_core_layers = outer_core.iloc[[-1]].index[0] + 1

    fid = open('output/{0}/{0}.card'.format(name), 'w')
    fid.write(name + '\n  1   -1   1\n')
    fid.write('  {0:d}   {1:d}   {2:d}\n'.format(mineos_card_model.shape[0],
            n_inner_core_layers, n_core_layers));
    # Now print the model
    for index, row in mineos_card_model.iterrows():
        fid.write('{0:6.0f}. {1:8.2f} {2:8.2f} {3:8.2f} '.format(
            row['radius'], row['rho'], row['vpv'], row['vsv']
        ))
        fid.write('{0:8.1f} {1:8.1f} {2:8.2f} {3:8.2f} {4:8.5f}\n'.format(
            row['q_kappa'], row['q_mu'], row['vph'], row['vsh'], row['eta']
        ))

    fid.close()



def smooth_to_ref_model_below(ref_model, new_model):
    """
    """

    smooth_z = 100 * 1e3  # 100 km in SI units - depth range to smooth over
    base_of_smoothing = new_model['radius'].iloc[0] - smooth_z
    unadulterated_ref_model = ref_model[ref_model['radius'] < base_of_smoothing]
    smoothed_ref_model = ref_model[
        (base_of_smoothing <= ref_model['radius'])
        & (ref_model['radius'] < new_model['radius'].iloc[0])
    ].copy()

    fraction_new_model = (
        (smoothed_ref_model['radius'] - base_of_smoothing) / smooth_z
    )

    for col in ref_model.columns.tolist()[1:]: # remove radius
        ref_value_at_model_base = np.interp(new_model['radius'].iloc[0],
                                            ref_model['radius'], ref_model[col])
        smoothed_ref_model[col] += (
            (new_model[col].iloc[0] - ref_value_at_model_base)
            * fraction_new_model
        )

    return pd.concat([unadulterated_ref_model, smoothed_ref_model])

def smooth_to_ref_model_above(ref_model, new_model):
    """
    """

    smooth_z = 100 * 1e3  # 100 km in SI units - depth range to smooth over
    top_of_smoothing = new_model['radius'].iloc[-1] + smooth_z
    unadulterated_ref_model = ref_model[top_of_smoothing < ref_model['radius']]
    smoothed_ref_model = ref_model[
        (new_model['radius'].iloc[-1] < ref_model['radius'])
        & (ref_model['radius'] <= top_of_smoothing )
    ].copy()

    fraction_new_model = (
        (top_of_smoothing - smoothed_ref_model['radius']) / smooth_z
    )

    for col in ref_model.columns.tolist()[1:]: # remove radius
        ref_value_at_model_top = np.interp(new_model['radius'].iloc[-1],
                                            ref_model['radius'], ref_model[col])
        smoothed_ref_model[col] += (
            (new_model[col].iloc[-1] - ref_value_at_model_top)
            * fraction_new_model
        )

    return pd.concat([smoothed_ref_model, unadulterated_ref_model])


def _set_model_indices(setup_model, model, **kwargs):
    """ Return the indices of different geological layers in the model.

    This is useful for if we want to damp the different layers with different
    parameters.

    Arguments:
        - setup_model:
            - SetupModel
            - Units:    seismological
        - model:
            - InversionModel
            - Units:    seismological
        - kwargs:
            - key word arguments of format
                - Moho = some_float
                - LAB = some_float
            - Units:    km
            - If Moho and LAB are not specified in
    """

    # Last model point is not included in the inversion
    depth = np.cumsum(model.thickness[:-1])

    sed_inds = [list(model.vsv).index(v) for v in model.vsv if v <= 3]

    if 'Moho' in setup_model.boundary_names:
        moho_ind = model.boundary_inds[setup_model.boundary_names.index('Moho')]
    else:
        try:
            moho_ind = np.argmin(np.abs(depth - kwargs['Moho']))
        except:
            print('Moho depth never specified! Guessing 35 km.')
            moho_ind = np.argmin(np.abs(depth - 35))

    if 'LAB' in setup_model.boundary_names:
        lab_ind = model.boundary_inds[setup_model.boundary_names.index('LAB')]
    else:
        try:
            lab_ind = np.argmin(np.abs(depth - kwargs['LAB']))
        except:
            print('LAB depth never specified! Guessing 80 km.')
            lab_ind = np.argmin(np.abs(depth - 80))


    return ModelLayerIndices(
        sediment = np.array(sed_inds),
        crust = np.arange(len(sed_inds), moho_ind + 1),
        lithospheric_mantle = np.arange(moho_ind + 1, lab_ind + 1),
        asthenosphere = np.arange(lab_ind + 1, len(depth)),
        boundary_layers = len(depth) + np.arange(len(model.boundary_inds)),
        depth = list(depth)
    )
