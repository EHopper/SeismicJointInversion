import unittest
from parameterized import parameterized
import shutil
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import define_models
from util import mineos
from util import inversion
from util import partial_derivatives
from util import weights




def test_G(setup_model, periods, model_perturbation):
    """ Test G by comparing the dV from G * dm to the dV output from MINEOS.

    From a given starting model, calculate phase velocities and kernels
    in MINEOS, and convert to the inversion G.

    Convert the starting model to the column vector.  Perturb the column
    vector in some way, and then convert it back to a MINEOS card.  Rerun
    MINEOS to get phase velocities.

    As G is dV/dm, then G * dm == mineos(m_perturbed) - mineos(m0).

    Plot this comparison:
    im = plt.scatter(dv_mineos, dv_from_Gdm*1e3, 10, periods)
    allv = list(dv_mineos) + list(dv_from_Gdm)
    plt.plot([min(allv), max(allv)], [min(allv), max(allv)], 'k:')
    plt.xlabel('dV from MINEOS')
    plt.ylabel('dV from G calculation')
    cbar = plt.gcf().colorbar(im)
    cbar.set_label('Periods (s)', rotation=90)
    """
    # Calculate the G matrix from a starting model
    model = define_models.setup_starting_model(setup_model)
    # No need for output on MINEOS model as saved to .card file
    minmod = define_models.convert_inversion_model_to_mineos_model(
        model, setup_model
    )
    params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
    ph_vel_pred, kernels = mineos.run_mineos_and_kernels(
        params, periods, setup_model.id
    )
    G = partial_derivatives._build_partial_derivatives_matrix(
        kernels, model, setup_model
    )
    # G_MINEOS = partial_derivatives._build_MINEOS_G_matrix(kernels)
    # G_MINEOS = G_MINEOS[:, :G_MINEOS.shape[1] // 5]
    # depth = kernels.loc[kernels['period'] == periods[0], 'z'].values
    # dvsv_dp_mat = partial_derivatives._convert_to_model_kernels(depth, model)
    # G = np.matmul(G_MINEOS, dvsv_dp_mat)
    plt.figure(figsize=(5,5))
    im = plt.imshow(G)
    plt.gca().set_aspect('auto')
    cb = plt.gcf().colorbar(im)


    # Apply the perturbation
    p = inversion._build_model_vector(model)
    p_perturbed = p.copy() * np.array(model_perturbation)[:, np.newaxis]
    perturbation = p_perturbed - p
    model_perturbed = inversion._build_inversion_model_from_model_vector(
        p_perturbed, model
    )
    minmod_perturbed = define_models.convert_inversion_model_to_mineos_model(
        model_perturbed, setup_model._replace(id='testcase_perturbed')
    )
    # minmod.vsv = minmod_perturbed.vsv
    # define_models._write_mineos_card(minmod, 'testcase_perturbed')
    ph_vel_perturbed, _ = mineos.run_mineos(
        params, periods, 'testcase_perturbed'
    )

    # calculate dv
    dc_mineos = ph_vel_perturbed - ph_vel_pred
    dc_from_Gdm = np.matmul(G, perturbation).flatten()

    plt.figure(figsize=(10,8))
    a1 = plt.subplot(1, 3, 1)
    plt.plot(model.vsv, np.cumsum(model.thickness), 'b-o')
    plt.plot(model_perturbed.vsv, np.cumsum(model_perturbed.thickness), 'r-o')
    plt.title('Vsv Models\nblue: original; red: perturbed')
    plt.gca().set_ylim([200, 0])
    plt.xlabel('Vsv (km/s)')
    plt.ylabel('Depth (km)')
    a2 = plt.subplot(2, 2, 2)
    #sc = 1.86
    im = plt.scatter(dc_mineos, dc_from_Gdm, 10, periods)
    #plt.scatter(dc_mineos, dc_from_Gdm * sc, 2, periods)
    allc = list(dc_mineos) + list(dc_from_Gdm)
    plt.plot([min(allc), max(allc)], [min(allc), max(allc)], 'k:')
    if min(allc) < 0 and max(allc) < 0:
        lims = [1.1 * min(allc), 0.9 * max(allc)]
    if min(allc) < 0 and max(allc) >= 0:
        lims = [1.1 * min(allc), 1.1 * max(allc)]
    if min(allc) >= 0 and max(allc) >= 0:
        lims = [0.9 * min(allc), 1.1 * max(allc)]
    plt.gca().set_xlim(lims)
    plt.gca().set_ylim(lims)
    plt.xlabel('dc from MINEOS')
    plt.ylabel('dc from G calculation')
    cbar = plt.gcf().colorbar(im)
    cbar.set_label('Periods (s)', rotation=90)
    a3 = plt.subplot(2, 2, 4)
    im = plt.scatter(periods, (dc_from_Gdm - dc_mineos) / dc_mineos, 10, dc_mineos)
    plt.xlabel('Period (s)')
    plt.ylabel('(Gdm - dc) / dc')
    cbar2 = plt.gcf().colorbar(im)
    cbar2.set_label('dc from MINEOS (km/s)', rotation=90)


    return dc_mineos, dc_from_Gdm
