import unittest
from parameterized import parameterized
import shutil
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from util import define_models
from util import mineos
from util import inversion
from util import partial_derivatives
from util import weights
from util import constraints
from util import plots




def test_G(model_params, periods, model_perturbation):
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
    model = define_models.setup_starting_model(model_params)
    # No need for output on MINEOS model as saved to .card file
    minmod = define_models.convert_inversion_model_to_mineos_model(
        model, model_params
    )
    params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
    ph_vel_pred, kernels = mineos.run_mineos_and_kernels(
        params, periods, model_params.id
    )
    G = partial_derivatives._build_partial_derivatives_matrix(
        kernels, model, model_params
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
        model_perturbed, model_params._replace(id='testcase_perturbed')
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
    model_params = define_models.ModelParams(
        'testcase', np.array([25., 120.]), np.array([5, 20]),
        np.array([10, 30]), np.array([3.6, 4.0, 4.4, 4.3]),
        np.array([0, 300])
    )
    periods = [5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 120]
    model_perturbation = ([1.05] * 5 + [0.95] * 5 + [1.02] * 5 + [0.99] * 5
    + [1.06] * 5 + [0.97] * 5 + [1.01] * 5 + [1]
    + [1.1] * 2)
    dc_mineos, dc_Gdm = test_G(model_params, periods, model_perturbation)
    print(dc_mineos, dc_Gdm)


def test_damping(): #n_iter
    lab = 'roughness_1b'
    print(lab)
    vs_SL14 = pd.read_csv('data/earth_models/CP_SL14.csv',
                           header=None).values
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values

    vs_SLall = pd.read_csv('data/earth_models/SchmandtLinVs_only.csv',
                            header=None).values.flatten()
    # To save spave, have saved the lat, lon, depth ranges of the Schmandt & Lin
    # model just as three lines in a csv
    vs_lats = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=0,
                          nrows=1, header=None).values.flatten()
    vs_lons = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=1,
                          nrows=1, header=None).values.flatten()
    vs_deps = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=2,
                          nrows=1, header=None).values.flatten()
    vs_SLall = vs_SLall.reshape((vs_lats.size, vs_lons.size, vs_deps.size))


    for lat in [40]:#[33]: #range(33, 41, 2):#range(34, 41):
        for lon in [-115]:#[-111, -115]: #range(-115, -105, 2):
            for t_LAB in [5.]:#[30., 25., 20., 15., 10.]:
                location = (lat, lon)
                print('*********************** {}N, {}W, {}km LAB'.format(
                    lat, -lon, t_LAB
                ))

                model_params = define_models.ModelParams('test_damp' + lab,
                    boundaries=(('Moho', 'LAB'), [3., t_LAB]),
                    depth_limits=(0, 300),
                )
                #location = (35, -112)
                obs, std_obs, periods = constraints.extract_observations(
                    location, model_params.id, model_params.boundaries,
                    model_params.vpv_vsv_ratio,
                )

                model = define_models.setup_starting_model(model_params, location)

                run_plot_inversion(
                    model_params, model, obs, std_obs, periods, location
                    )

def test_MonteCarlo(n_MonteCarlo): #n_iter

    vs_SL14 = pd.read_csv('data/earth_models/CP_SL14.csv',
                           header=None).values
    cp_outline = pd.read_csv('data/earth_models/CP_outline.csv').values

    vs_SLall = pd.read_csv('data/earth_models/SchmandtLinVs_only.csv',
                            header=None).values.flatten()
    # To save space, have saved the lat, lon, depth ranges of the Schmandt & Lin
    # model just as three lines in a csv
    vs_lats = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=0,
                          nrows=1, header=None).values.flatten()
    vs_lons = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=1,
                          nrows=1, header=None).values.flatten()
    vs_deps = pd.read_csv('data/earth_models/SL_coords.csv', skiprows=2,
                          nrows=1, header=None).values.flatten()
    vs_SLall = vs_SLall.reshape((vs_lats.size, vs_lons.size, vs_deps.size))

    lat = 35.
    lon = -111.
    t_LAB = 5.
    location = (lat, lon)
    print('*********************** {}N, {}W, {}km LAB'.format(
        lat, -lon, t_LAB
    ))

    model_params = define_models.ModelParams('test_MC',
        boundaries=(('Moho', 'LAB'), [3., t_LAB]),
        depth_limits=(0, 420),
    )
    obs, std_obs, periods = constraints.extract_observations(
        location, model_params.id, model_params.boundaries,
        model_params.vpv_vsv_ratio,
    )


    ic = list(range(len(obs) - 2 * len(model_params.boundaries[0])))
    i_rf = list(range(ic[-1] + 1, len(obs)))

    #model_params = model_params._replace(boundary_names = [])
    save_name = 'output/{0}/{0}.q'.format(model_params.id)
    lat, lon = location

    f = plt.figure()
    f.set_size_inches((15,7))
    ax_m0 = f.add_axes([0.2, 0.3, 0.1, 0.6])
    ax_m150 = f.add_axes([0.35, 0.3, 0.2, 0.6])
    ax_mDeep = f.add_axes([0.35, 0.1, 0.2, 0.12])
    ax_c = f.add_axes([0.6, 0.6, 0.35, 0.3])
    ax_map = f.add_axes([0.84, 0.1, 0.1, 0.2])
    ax_dc = f.add_axes([0.6, 0.375, 0.35, 0.15])
    im = ax_map.contourf(np.arange(-119, -100.9, 0.2),
        np.arange(30, 45.1, 0.2), vs_SL14, levels=20,
        cmap=plt.cm.RdBu, vmin=4, vmax=4.7)
    ax_map.plot(lon, lat, 'k*')
    ax_map.plot(cp_outline[:, 1], cp_outline[:, 0], 'k:')
    ax_c.set_title(
        '{:.1f}N, {:.1f}W:  {:.0f} km LAB'.format(lat, -lon, t_LAB)
    )
    ax_rf = f.add_axes([0.6, 0.1, 0.2, 0.2])

    # Plot on data and Schmandt & Lin Vs model
    line, = ax_c.plot(periods, obs[ic],
                      'k-', linewidth=3, label='data')
    plots.plot_rf_data_std(obs[i_rf], std_obs[i_rf], 'data', ax_rf)
    vs_ilon = np.argmin(abs(vs_lons - lon))
    vs_ilat = np.argmin(abs(vs_lats - lat))
    ax_m150.plot(vs_SLall[vs_ilat, vs_ilon, :], vs_deps,
                 'k-', linewidth=3, label='SL14')
    ax_mDeep.plot(vs_SLall[vs_ilat, vs_ilon, :], vs_deps,
                  'k-', linewidth=3, label='SL14')

    for trial in range(n_MonteCarlo):
        m = define_models.setup_starting_model(model_params, location)
        plots.plot_model_simple(m, 'm0 ' + str(trial), ax_m0, (0, 150))

        dc = np.ones_like(periods)
        old_dc = np.zeros_like(periods)
        n = -1
        while np.sum(abs(dc - old_dc)) > 0.005 * periods.size and n < 10:
            n += 1
            old_dc = dc.copy()

            print('****** ITERATION ' +  str(n) + ' ******')
            m, G = inversion._inversion_iteration(model_params, m, location)
            c = mineos._read_qfile(save_name, periods)
            dc = np.array([c[i] - obs[ic[i]] for i in range(len(c))])


        # Plot up the model that reached convergence
        plots.plot_model_simple(m, 'm' + str(n + 1), ax_m150, (0, 150))
        plots.plot_model_simple(m, 'm' + str(n + 1), ax_mDeep,
                                (150, model_params.depth_limits[1]), False)
        p_rf = inversion._predict_RF_vals(m)
        plots.plot_rf_data(p_rf, 'm' + str(n + 1), ax_rf)
        # Run MINEOS on final model
        params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
        _ = define_models.convert_inversion_model_to_mineos_model(m, model_params)
        c, _ = mineos.run_mineos(params, periods, model_params.id)
        plots.plot_ph_vel_simple(periods, c, ax_c)
        dc = [c[i] - obs[ic[i]] for i in range(len(c))]
        plots.plot_ph_vel_simple(periods, dc, ax_dc)
        ax_dc.plot(periods, [0] * len(periods), 'k--')
        plots.make_plot_symmetric_in_y_around_zero(ax_dc)
        plots.make_plot_symmetric_in_y_around_zero(ax_rf)

    print(trial + 1, ' TRIALS')
    save_name = 'output/{0}/{0}'.format(model_params.id)

    f.savefig(
        '/media/sf_VM_Shared/rftests/MC_{}N_{}W_{}kmLAB_{}trials.png'.format(
        location[0], -location[1], round(t_LAB), trial + 1,
        )
    )

def run_plot_MC_inversion(model_params, orig_model,
                          obs, std_obs, periods, location,
                          n_MonteCarlo=5, max_runs=10):

    ic = list(range(len(obs) - 2 * len(model_params.boundaries[0])))
    i_rf = list(range(ic[-1] + 1, len(obs)))
    t_LAB = model_params.boundaries[1][1]

    # Setup the figure and reference plots
    f, ax_c, ax_dc, ax_rf, ax_m150, ax_mDeep, ax_map = (
        plots.setup_figure_layout(location, t_LAB)
    )
    # Add extra axes for comparisons
    ax_m1_150 = f.add_axes([0.15, 0.3, 0.15, 0.6])
    ax_m1_Deep = f.add_axes([0.15, 0.1, 0.15, 0.12])
    ax_diff_m150 = f.add_axes([0.05, 0.3, 0.05, 0.6])
    ax_diff_mDeep = f.add_axes([0.05, 0.1, 0.05, 0.12])

    plots.plot_area_map(location, ax_map)
    # Plot on SR16 values
    ax_m150.plot(orig_model.vsv, np.cumsum(orig_model.thickness), 'k-',
                 linewidth=3, label='SR16')
    ax_mDeep.plot(orig_model.vsv, np.cumsum(orig_model.thickness), 'k-',
                  linewidth=3, label='SR16')
    ax_m1_150.plot(orig_model.vsv, np.cumsum(orig_model.thickness), 'k-',
                 linewidth=3, label='SR16')
    ax_m1_Deep.plot(orig_model.vsv, np.cumsum(orig_model.thickness), 'k-',
                  linewidth=3, label='SR16')


    # Plot on data
    plots.plot_ph_vel_data_std(
        periods, obs[ic].flatten(), std_obs[ic].flatten(), 'data', ax_c
    )
    plots.plot_rf_data_std(obs[i_rf], std_obs[i_rf], 'data', ax_rf)
    ax_dc.errorbar(periods, [0] * len(periods), yerr=std_obs[ic].flatten(),
                   linestyle='--', color='k', ecolor='k')

    for trial in range(n_MonteCarlo):
        print('****** MC trial {} ******'.format(trial))
        # Add noise to starting model
        t = orig_model.thickness.copy()
        v = orig_model.vsv.copy()
        bi = orig_model.boundary_inds.copy()
        di = define_models._find_depth_indices(t, model_params.depth_limits)
        define_models._add_noise_to_starting_model(t, v, bi, di)
        model = define_models.InversionModel(v, t, bi, di)

        # Plot starting model of this iteration
        ax_m150.plot(model.vsv, np.cumsum(model.thickness), '-',
                     color='#BDBDBD', linewidth=1)
        ax_mDeep.plot(model.vsv, np.cumsum(model.thickness), '-',
                     color='#BDBDBD', linewidth=1)
        p = ax_m1_150.plot(model.vsv, np.cumsum(model.thickness), linewidth=0.5)
        pcol = p[0].get_color()
        ax_m1_Deep.plot(model.vsv, np.cumsum(model.thickness), linewidth=0.5)

        dc = np.ones_like(periods)
        old_dc = np.zeros_like(periods)
        n = 0
        while  np.sum(abs(dc - old_dc)) > 0.005 * periods.size and n < max_runs:
            old_dc = dc.copy()
            # Run inversion
            print('****** ITERATION ' +  str(n) + ' ******')
            model, G, o = inversion._inversion_iteration(model_params, model, location,
                                                      (obs, std_obs, periods))

            # Read ouput file to check change in predicted phase velocities
            c = mineos._read_qfile('output/{0}/{0}.q'.format(model_params.id), periods)
            dc = np.array([c[i] - obs[ic[i]] for i in range(len(c))])
            n += 1
            if n == 1:
                plots.plot_model(model, 'MC{}_m{}'.format(trial, n), ax_m1_150,
                                 (0, 150), True, pcol)
                plots.plot_model(model, 'MC{}_m{}'.format(trial, n), ax_m1_Deep,
                                 (150, model_params.depth_limits[1]), False, pcol)

        # Plot values for final model
        plots.plot_model(model, 'MC{}_m{}'.format(trial, n), ax_m150, (0, 150))
        plots.plot_model(model, 'MC{}_m{}'.format(trial, n), ax_mDeep,
                         (150, model_params.depth_limits[1]), False)
        if trial == 0:
            base_z = np.arange(0, model_params.depth_limits[1], 0.5)
            base_v = np.interp(base_z, np.cumsum(model.thickness), model.vsv.ravel())
            orig_v = np.interp(base_z, np.cumsum(orig_model.thickness), orig_model.vsv.ravel())

            # Plot on mismatch to SR16
            ax_diff_m150.plot(orig_v - base_v, base_z, 'k--', label='SR16')
            ax_diff_m150.xaxis.set_label_position('top')
            ax_diff_m150.xaxis.tick_top()
            ax_diff_m150.set(xlabel='Difference in Vsv (km/s)', ylabel='Depth (km)')
            ax_diff_mDeep.plot(orig_v - base_v, base_z, 'k--', label='SR16')
            ax_diff_mDeep.xaxis.set_label_position('top')
            ax_diff_mDeep.xaxis.tick_top()
            ax_diff_mDeep.set(xlabel='Difference in Vsv (km/s)', ylabel='Depth (km)')
            maxdiff = 0

        # Plot on difference compared to other MC trials
        trial_v = np.interp(base_z, np.cumsum(model.thickness), model.vsv.ravel())
        ax_diff_m150.plot(trial_v - base_v, base_z, '-',
                          label='MC{}_m{}'.format(trial, n), color=pcol)
        ax_diff_mDeep.plot(trial_v - base_v, base_z, '-',
                           label='MC{}_m{}'.format(trial, n), color=pcol)
        maxdiff = max(max(abs(trial_v - base_v)), maxdiff)


        p_rf = inversion._predict_RF_vals(model)
        plots.plot_rf_data(p_rf, 'MC{}_m{}'.format(trial, n), ax_rf)
        c = mineos.calculate_c_from_card(model_params, model, periods)
        dc = [c[i] - obs[ic[i]] for i in range(len(c))]
        plots.plot_ph_vel(periods, c, 'MC{}_m{}'.format(trial, n), ax_c)
        plots.plot_dc(periods, dc, ax_dc)

    # Tidy up the plots
    ax_dc.set_ylim(max(max(abs(np.array(dc))) * 1.25, 0.06) * np.array([-1, 1]))
    xl = maxdiff * 1.1 * np.array([-1, 1])
    ax_diff_m150.set(ylim = [150, 0], xlim = xl)
    ax_diff_mDeep.set(ylim = [model_params.depth_limits[1], 150], xlim = xl)
    plots.make_plot_symmetric_in_y_around_zero(ax_rf)
    ax_m150.set_xlim([2, 5.5])
    ax_mDeep.set_xlim([2, 5.5])
    ax_m1_150.set_xlim([2, 5.5])
    ax_m1_Deep.set_xlim([2, 5.5])
    ax_c.set_ylim([2, 4.5])

    f.savefig(
        '/media/sf_VM_Shared/rftests/{}N_{}W_{}kmLAB{}_{}MC.png'.format(
        location[0], -location[1], round(t_LAB), model_params.id, n_MonteCarlo
        )
    )

def run_plot_inversion(model_params, model,
                       obs, std_obs, periods, location, m, max_runs=10):

    ic = list(range(len(obs) - 2 * len(model_params.boundaries[0])))
    i_rf = list(range(ic[-1] + 1, len(obs)))

    t_LAB = model.thickness.item(model.boundary_inds[-1] + 1)

    # Setup the figure and reference plots
    f, ax_c, ax_dc, ax_rf, ax_m150, ax_mDeep, ax_map = (
        plots.setup_figure_layout(location, t_LAB)
    )
    plots.plot_area_map(location, ax_map)
    #plots.plot_SL14_profile(location, ax_m150)
    #plots.plot_SL14_profile(location, ax_mDeep)
    ph_vel_pred = mineos.calculate_c_from_card(model_params, m, periods)
    ax_m150.plot(m.vsv, np.cumsum(m.thickness), 'k-', linewidth=3, label='SR16')
    ax_mDeep.plot(m.vsv, np.cumsum(m.thickness), 'k-', linewidth=3, label='SR16')
    ax_c.plot(periods, ph_vel_pred, 'k--o', markersize=3, label='SR16')
    ax_dc.plot(periods, ph_vel_pred.flatten() - obs[ic].flatten(), 'k--o', markersize=3, label='SR16')

    # Plot on data
    plots.plot_ph_vel_data_std(
        periods, obs[ic].flatten(), std_obs[ic].flatten(), 'data', ax_c
    )
    plots.plot_rf_data_std(obs[i_rf], std_obs[i_rf], 'data', ax_rf)
    ax_dc.errorbar(periods, [0] * len(periods), yerr=std_obs[ic].flatten(),
                   linestyle='--', color='k', ecolor='k')

    dc = np.ones_like(periods)
    old_dc = np.zeros_like(periods)
    n = 0
    while  np.sum(abs(dc - old_dc)) > 0.005 * periods.size and n < max_runs:
        old_dc = dc.copy()

        # Plot starting model of this iteration
        plots.plot_model(model, 'm' + str(n), ax_m150, (0, 150))
        plots.plot_model(model, 'm' + str(n), ax_mDeep,
                         (150, model_params.depth_limits[1]), False)

        # Plot predicted RF vals of this iteration
        p_rf = inversion._predict_RF_vals(model)
        plots.plot_rf_data(p_rf, 'm' + str(n), ax_rf)

        # Run inversion
        print('****** ITERATION ' +  str(n) + ' ******')
        model, G, o = inversion._inversion_iteration(model_params, model, location,
                                                  (obs, std_obs, periods))

        # Plot predicted c from previous iteration (calculated for inversion)
        c = mineos._read_qfile('output/{0}/{0}.q'.format(model_params.id), periods)
        dc = np.array([c[i] - obs[ic[i]] for i in range(len(c))])
        plots.plot_ph_vel(periods, c, 'm' + str(n), ax_c)
        plots.plot_dc(periods, dc, ax_dc)

        n += 1


    # Plot values for final model
    plots.plot_model(model, 'm' + str(n), ax_m150, (0, 150))
    plots.plot_model(model, 'm' + str(n), ax_mDeep,
                     (150, model_params.depth_limits[1]), False)
    p_rf = inversion._predict_RF_vals(model)
    plots.plot_rf_data(p_rf, 'm' + str(n), ax_rf)
    c = mineos.calculate_c_from_card(model_params, model, periods)
    dc = [c[i] - obs[ic[i]] for i in range(len(c))]
    plots.plot_ph_vel(periods, c, 'm' + str(n + 1), ax_c)
    plots.plot_dc(periods, dc, ax_dc)

    # Tidy up the plots
    ax_dc.set_ylim(max(max(abs(np.array(dc))) * 1.25, 0.06) * np.array([-1, 1]))
    plots.make_plot_symmetric_in_y_around_zero(ax_rf)
    ax_m150.set_xlim([2, 5.5])
    ax_mDeep.set_xlim([2, 5.5])
    ax_c.set_ylim([2, 4.5])

    # Plot on the damping parameters
    #plots.plot_damping_params(model_params.id, f)

    # Print on the observed constraints
    obs_c_t = ['{:3.0f} s: {:.3f} {:s} {:.2f} km/s'.format(
        periods[i], obs.item(i), u'\u00B1', std_obs.item(i)
        ) for i in range(len(periods))]
    obs_rf_t = ['Moho tt: {:3.1f} {:s} {:.2f} s'.format(
                        obs.item(-4), u'\u00B1', std_obs.item(-4)),
                'Moho dV: {:3.0f} {:s} {:.0f} %'.format(
                            obs.item(-2) * 100, u'\u00B1', std_obs.item(-2) * 100),
                ' LAB tt: {:3.1f} {:s} {:.2f} s'.format(
                            obs.item(-3), u'\u00B1', std_obs.item(-3)),
                ' LAB dV: {:3.0f} {:s} {:.0f} %'.format(
                            obs.item(-1) * 100, u'\u00B1', std_obs.item(-1) * 100)]
    ax_t = f.add_axes([0.05, 0.1, 0.3, 0.15])
    ax_t.set_axis_off()
    i = 0
    for ss in obs_c_t:
        ax_t.text(0.15, 1 - i/10, ss, fontsize=6)
        i += 1
    i = 0
    for ss in obs_rf_t:
        ax_t.text(0.5, 1 - i/10, ss, fontsize=6)
        i += 1


    f.savefig(
        '/media/sf_VM_Shared/rftests/{}N_{}W_{}kmLAB_{}.png'.format(
        location[0], -location[1], round(t_LAB), model_params.id
        )
    )
    plt.close()
    return model, G, o


def try_run(location:tuple, t_BLs:tuple, id:str):

    t_Moho, t_LAB = t_BLs
    sm = define_models.ModelParams(id,
                                  min_layer_thickness=6,
                                  depth_limits=(0, 350),
                                  boundaries=(('Moho','LAB'), [t_Moho, t_LAB]),
                                  )
    t, v = constraints.get_vels_ShenRitzwoller2016(location)
    #d = np.cumsum(t)
    #v[0:3] = [v[3]] * 3 # remove lowest velocity layer
    #v[64:200] = [v[64]] * (200 - 64) # add in a clear high velocity lid
    #v[64:] = [v[64]] * (len(v) - 64)
    # Add in boundary layers at arbitary places
    bi = np.array([63, 199])
    t[bi[0] + 1] = t_Moho
    t[bi[1] + 1] = t_LAB

    define_models._fill_in_base_of_model(t, v, sm)
    t = np.array(t)
    v = np.array(v)

    # Smooth from base of the lithosphere to the base of the model
    # v = np.hstack((
    #     v[:200], np.interp(sum(t[:200]) + np.cumsum(t[200:]),
    #                        [sum(t[:200]), sum(t)],
    #                        [v[200], v[-1]]
    #                        )
    # ))

    thickness, vsv, boundary_inds = define_models._return_evenly_spaced_model(
        t, v, bi, sm.min_layer_thickness,
    )

    m = define_models.InversionModel(
        vsv = vsv[np.newaxis].T,
        thickness = thickness[np.newaxis].T,
        boundary_inds = np.array(boundary_inds),
        d_inds = define_models._find_depth_indices(thickness, sm.depth_limits),
    )
    # mineos_model = define_models.convert_inversion_model_to_mineos_model(m, sm)

    # Get predicted values
    # rf_p = inversion._predict_RF_vals(m)
    obs, std_obs, periods = constraints.extract_observations(
            location, sm.id, sm.boundaries, sm.vpv_vsv_ratio
            )
    #ph_vel_pred = mineos.calculate_c_from_card(sm, m, periods)

    m0 = define_models.InversionModel(
            vsv = vsv[np.newaxis].T * 0 + vsv[-1],
            thickness = thickness[np.newaxis].T,
            boundary_inds = np.array(boundary_inds),
            d_inds = define_models._find_depth_indices(thickness, sm.depth_limits),
        )
    max_runs = 10
    # return run_plot_inversion(sm, m0, np.hstack((ph_vel_pred, rf_p))[:, np.newaxis],
    #                    std_obs, periods, location, m, max_runs
    #                    )
    return run_plot_inversion(sm, m0, obs,
                       std_obs, periods, location, m, max_runs
                       )
    #return run_plot_MC_inversion(sm, m, obs, std_obs, periods, location)

def loop_through_locs():

    broken = ((37, -107), (39, -106), (40, -108), (41, -108), (41, -107))#((33, -115), (41, -108))
    id = '_noMohoLAB'
    for t_LAB in [5]:
        for lat in range(33, 43, 1):
            for lon in range(-117, -102):#range(-117, -102, 1):

                if (lat, lon) in broken:
                    print('{}, {} is broken'.format(lat, lon))
                    continue

                t_Moho = 3.
                fname = '{}N_{}W_{}kmLAB{}'.format(lat, lon, t_LAB, id)

                if not os.path.isfile('output/models/{}.csv'.format(fname)):
                    print('Doing {}, {}!'.format(lat, lon))
                    m, G, o = try_run((lat, lon), (t_Moho, t_LAB), id)
                    define_models.save_model(m, fname)
                else:
                    print('Done {}, {} already!'.format(lat, lon))

def load_models(zmax=350):
    z = np.arange(0, min((zmax, 350)), 0.5)
    lats = np.arange(34, 41)#(33, 41)
    lons = np.arange(-114, -106)#(-117, -102)
    t_LAB = 5
    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB, '_smoothed')

    return vs, bls, bis, z, lats, lons

def compare_models(ref):

    #ref = 'S15', 'SR16'
    ds = constraints.load_literature_vel_model(ref)
    vs, bls, bis, z, lats, lons = load_models(ds.depth.values[-1])
    vs_a = constraints.interpolate_lit_model(ref, z, lats, lons)

    # for d in range(60, 100, 5):
    #     plots.plot_map(vs - vs_a, lats, lons, z, d)
    #     cl = plt.gci().get_clim()
    #     #plt.clim(max(np.abs(cl)) * np.array([-1, 1]))
    #     plt.clim([-0.3, 0.3])
    #     plt.title('My model - {}'.format(ref))

    return vs, vs_a, lats, lons, z

def load_stuff():

    z = np.arange(0, 300, 0.5)
    lats = np.arange(33, 43)
    lons = np.arange(-115, -102)
    t_LAB = 5
    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB, '_highQ')
    half_tlab_i = int(t_LAB / np.diff(z[:2]) / 2)

    v_above_LAB = np.zeros_like(vs[:, :, 0])
    v_below_LAB = np.zeros_like(v_above_LAB)
    v_above_Moho = np.zeros_like(v_above_LAB)
    v_below_Moho = np.zeros_like(v_above_LAB)
    av_grad_SCLM = np.zeros_like(v_above_LAB)
    max_grad_SCLM = np.zeros_like(v_above_LAB)
    min_grad_SCLM = np.zeros_like(v_above_LAB)
    for ila in range(vs.shape[0]):
        for ilo in range(vs.shape[1]):
            v_above_Moho[ila, ilo] = vs[ila, ilo, bis[ila, ilo, 0]]
            v_below_Moho[ila, ilo] = vs[ila, ilo, bis[ila, ilo, 1]]
            v_above_LAB[ila, ilo] = vs[ila, ilo, bis[ila, ilo, 2]]
            v_below_LAB[ila, ilo] = vs[ila, ilo, bis[ila, ilo, 3]]
            av_grad_SCLM[ila, ilo] = (
                (v_above_LAB[ila, ilo] - v_below_Moho[ila, ilo])
                / (z[bis[ila, ilo, 2]] - z[bis[ila, ilo, 1]])
            )
            max_grad_SCLM[ila, ilo] = (
                max(
                    np.diff(vs[ila, ilo, bis[ila, ilo, 1]:bis[ila, ilo, 2] + 1])
                    / np.diff(z[:2])
                )
            )
            min_grad_SCLM[ila, ilo] = (
                min(
                    np.diff(vs[ila, ilo, bis[ila, ilo, 1]:bis[ila, ilo, 2] + 1])
                    / np.diff(z[:2])
                )
            )
    return (z, lats, lons, vs, bls, bis, v_above_LAB, v_below_LAB, v_above_Moho,
            v_below_Moho, av_grad_SCLM, max_grad_SCLM, min_grad_SCLM)

def get_vlayer(offset_km=-5, layer='LAB'):
    z = np.arange(0, 300, 0.5)
    lats = np.arange(33, 43)
    lons = np.arange(-115, -102)
    t_LAB = 5
    vs, bls, bis = define_models.load_all_models(z, lats, lons, t_LAB, '_highQ')
    # offset_km = -5

    vlayer = np.zeros_like(vs[:, :, 0])
    for ila in range(vs.shape[0]):
        for ilo in range(vs.shape[1]):
            if offset_km <= 0:
                if layer == 'Moho': # above top of Moho
                    BL = bis[ila, ilo, 0]
                if layer == 'LAB': # above top of LAB
                    BL = bis[ila, ilo, 2]
            else:
                if layer == 'Moho': # below bottom of Moho
                    BL = bis[ila, ilo, 1]
                if layer == 'LAB': # below bottom of LAB
                    BL = bis[ila, ilo, 3]

            offset = int(offset_km // np.diff(z[:2]))
            dind = BL + offset
            vlayer[ila, ilo] = vs[ila, ilo, dind]
    return vlayer

def pull_Moho(ref):

    ds = constraints.load_literature_vel_model(ref)
    lats = ds.latitude.values
    lons = ds.longitude.values
    z = ds.depth.values
    vs = ds.vs.values
    moho_z = np.zeros_like(vs[0, :, :])
    moho_dv = np.zeros_like(vs[0, :, :])
    ila = 0
    ilo = 0
    iz = np.argmax(z > 20) # only returns index of first max val
    for lat in lats:
        for lon in lons:
            Moho_ind = np.argmax(np.diff(vs[iz:, ila, ilo])) + iz
            moho_z[ila, ilo] = z[Moho_ind]
            maxi = 1 #np.argmax(vs[Moho_ind:Moho_ind + int(10 / np.diff(z[:2])), ila, ilo])
            moho_dv[ila, ilo] = (
                vs[Moho_ind + maxi, ila, ilo] - vs[Moho_ind, ila, ilo])
                    #vs[Moho_ind + maxi, ila, ilo] / vs[Moho_ind, ila, ilo] - 1)
            ilo += 1
        ila += 1
        ilo = 0

    if lons[0] > 180:
        lons -= 360

    return moho_z, moho_dv, lats, lons
