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
from util import constraints
from util import plots




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


def run_test_G():
    setup_model = define_models.SetupModel(
        'testcase', np.array([25., 120.]), np.array([5, 20]),
        np.array([10, 30]), np.array([3.6, 4.0, 4.4, 4.3]),
        np.array([0, 300])
    )
    periods = [5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 120]
    model_perturbation = ([1.05] * 5 + [0.95] * 5 + [1.02] * 5 + [0.99] * 5
    + [1.06] * 5 + [0.97] * 5 + [1.01] * 5 + [1]
    + [1.1] * 2)
    dc_mineos, dc_Gdm = test_G(setup_model, periods, model_perturbation)
    print(dc_mineos, dc_Gdm)


def test_damping(n_iter):
    setup_model = define_models.SetupModel(
        'testcase', np.array([35., 100.]), np.array([5, 20]),
        np.array([10, 30]), np.array([3.6, 4.0, 4.4, 4.3]),
        np.array([0, 300])
    )
    data = constraints.extract_observations(35, -104)
    # To speed things up, remove some data
    data = data._replace(surface_waves =
        data.surface_waves.iloc[[3, 7, 9, 11, 12, 13], :].reset_index()
    )
    setup_model = setup_model._replace(boundary_names = [])
    periods = data.surface_waves.period.values
    save_name = 'output/{0}/{0}.q'.format(setup_model.id)

    f = plt.figure()
    f.set_size_inches((15,7))
    ax_m = f.add_axes([0.35, 0.1, 0.2, 0.8])
    ax_c = f.add_axes([0.6, 0.2, 0.35, 0.5])
    line, = ax_c.plot(periods, data.surface_waves.ph_vel.values,
                      'k-', linewidth=3)
    line.set_label('data')

    m = define_models.setup_starting_model(setup_model)
    m = m._replace(boundary_inds =  np.array([]))
    m = m._replace(thickness=np.array([0.] + [6.] * (len(m.vsv) - 1))[:, np.newaxis])
    plots.plot_model(m, 'm0', ax_m)
    for n in range(n_iter):
        m = inversion._inversion_iteration(setup_model, m, data)
        c = mineos._read_qfile(save_name, periods)
        plots.plot_model(m, 'm' + str(n + 1), ax_m)
        plots.plot_ph_vel(periods, c, 'm' + str(n), ax_c)
    # Run MINEOS on final model
    params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
    _ = define_models.convert_inversion_model_to_mineos_model(m, setup_model)
    c, _ = mineos.run_mineos(params, periods, setup_model.id)
    plots.plot_ph_vel(periods, c, 'm' + str(n + 1), ax_c)

    save_name = 'output/{0}/{0}'.format(setup_model.id)
    damp_s = pd.read_csv(save_name + 'damp_s.csv')
    damp_t = pd.read_csv(save_name + 'damp_t.csv')
    n = 0
    for label in ['roughness', 'to_m0', 'to_m0_grad']:
        ax_d = f.add_axes([0.05 + 0.1 * n, 0.1, 0.05, 0.8])
        ax_d.plot(damp_s[label], damp_s.Depth, 'ko-', markersize=3)
        ax_d.plot(damp_t[label], damp_t.Depth, 'ro', markersize=2)
        ax_d.set(title=label)
        ax_d.set_ylim([np.cumsum(m.thickness)[-1], 0])
        ax_d.xaxis.tick_top()
        n += 1
