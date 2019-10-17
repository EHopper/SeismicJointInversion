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

    vs_SL14 = pd.read_csv('data/earth_models/CP_SL14.csv',
                           header=None).values
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values


    for lat in [37]:#range(34, 41):
        for lon in [-115]: #range(-115, -105):
            for t_LAB in [5.]:#[30., 25., 20., 15., 10.]:
                location = (lat, lon)
                print('*********************** {}N, {}W, {}km LAB'.format(
                    lat, -lon, t_LAB
                ))

                setup_model = define_models.SetupModel(
                    'test2', (lat, lon), np.array([35., 80.]),
                    np.array([5, 10.]),
                    np.array([2, t_LAB]), np.array([3.5, 4.4,  4.4, 4.2]),
                    np.array([0, 300])
                )
                #location = (35, -112)
                obs, std_obs, periods = constraints.extract_observations(
                    setup_model
                )
                moho_z_ish = np.round(obs[-4] * 3.5)
                lab_z_ish = np.round(obs[-3] * 4.2)
                setup_model = setup_model._replace(
                    boundary_depths = np.concatenate((moho_z_ish, lab_z_ish))
                )


                # To speed things up, remove some data
                # i_p = list(range(3, 10, 2)) + list(range(10, 14))
                # i_rf = list(range(14, 18))
                # periods = periods[i_p]
                # obs = obs[i_p + i_rf]
                # std_obs = std_obs[i_p + i_rf]

                ic = list(range(len(obs) - 2 * len(setup_model.boundary_names)))
                i_rf = list(range(ic[-1] + 1, len(obs)))

                #setup_model = setup_model._replace(boundary_names = [])
                save_name = 'output/{0}/{0}.q'.format(setup_model.id)
                lat, lon = location

                f = plt.figure()
                f.set_size_inches((15,7))
                ax_m = f.add_axes([0.35, 0.1, 0.2, 0.8])
                ax_c = f.add_axes([0.6, 0.6, 0.35, 0.3])
                ax_map = f.add_axes([0.84, 0.64, 0.1, 0.2])
                ax_dc = f.add_axes([0.6, 0.375, 0.35, 0.15])
                im = ax_map.contourf(np.arange(-119, -100.9, 0.2),
                    np.arange(30, 45.1, 0.2), vs_SL14, levels=20,
                    cmap=plt.cm.RdBu, vmin=4, vmax=4.7)
                ax_map.plot(lon, lat, 'k*')
                ax_map.plot(cp_outline[:, 1], cp_outline[:, 0], 'k:')
                ax_c.set_title(
                    '{:.1f}N, {:.1f}W:  {:.0f} km LAB'.format(lat, -lon, t_LAB)
                )
                ax_rf = f.add_axes([0.6, 0.1, 0.35, 0.2])
                line, = ax_c.plot(periods, obs[ic],
                                  'k-', linewidth=3, label='data')
                plots.plot_rf_data_std(obs[i_rf], std_obs[i_rf], 'data', ax_rf)

                m = define_models.setup_starting_model(setup_model)
                # m = m._replace(boundary_inds =  np.array([]))
                # m = m._replace(thickness=np.array([0.] + [6.] * (len(m.vsv) - 1))[:, np.newaxis])
                plots.plot_model(m, 'm0', ax_m)
                p_rf = inversion._predict_RF_vals(m)
                plots.plot_rf_data(p_rf, 'm0', ax_rf)
                for n in range(n_iter):
                    print('****** ITERATION ' +  str(n) + ' ******')
                    m = inversion._inversion_iteration(setup_model, m, location)
                    p_rf = inversion._predict_RF_vals(m)
                    c = mineos._read_qfile(save_name, periods)
                    plots.plot_model(m, 'm' + str(n + 1), ax_m)
                    plots.plot_rf_data(p_rf, 'm' + str(n + 1), ax_rf)
                    plots.plot_ph_vel(periods, c, 'm' + str(n), ax_c)
                    dc = [c[i] - obs[ic[i]] for i in range(len(c))]
                    plots.plot_ph_vel(periods, dc, 'm' + str(n), ax_dc)

                # Run MINEOS on final model
                params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
                _ = define_models.convert_inversion_model_to_mineos_model(m, setup_model)
                c, _ = mineos.run_mineos(params, periods, setup_model.id)
                plots.plot_ph_vel(periods, c, 'm' + str(n + 1), ax_c)
                dc = [c[i] - obs[ic[i]] for i in range(len(c))]
                plots.plot_ph_vel(periods, dc, 'm' + str(n + 1), ax_dc)
                ax_dc.plot(periods, [0] * len(periods), 'k--')
                ax_dc.set(ylabel="dc (km/s)")
                ax_dc.set_ylim(max(abs(np.array(dc))) * 1.1 * np.array([-1, 1]))
                plots.make_plot_symmetric_in_y_around_zero(ax_rf)

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

                f.savefig(
                    '/media/sf_VM_Shared/rftests/{}N_{}W_{}kmLAB.png'.format(
                    lat, -lon, round(t_LAB),
                    )
                )
                plt.close(f)
