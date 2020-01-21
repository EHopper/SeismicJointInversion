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

                setup_model = define_models.SetupModel('test_damp' + lab,
                    boundaries=(('Moho', 'LAB'), [3., t_LAB]),
                    depth_limits=(0, 300),
                )
                #location = (35, -112)
                obs, std_obs, periods = constraints.extract_observations(
                    location, setup_model.id, setup_model.boundaries,
                    setup_model.vpv_vsv_ratio,
                )

                model = define_models.setup_starting_model(setup_model, location)

                run_plot_inversion(
                    setup_model, model, obs, std_obs, periods, location
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

    setup_model = define_models.SetupModel('test_MC',
        boundaries=(('Moho', 'LAB'), [3., t_LAB]),
        depth_limits=(0, 420),
    )
    obs, std_obs, periods = constraints.extract_observations(
        location, setup_model.id, setup_model.boundaries,
        setup_model.vpv_vsv_ratio,
    )


    ic = list(range(len(obs) - 2 * len(setup_model.boundaries[0])))
    i_rf = list(range(ic[-1] + 1, len(obs)))

    #setup_model = setup_model._replace(boundary_names = [])
    save_name = 'output/{0}/{0}.q'.format(setup_model.id)
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
        m = define_models.setup_starting_model(setup_model, location)
        plots.plot_model_simple(m, 'm0 ' + str(trial), ax_m0, (0, 150))

        dc = np.ones_like(periods)
        old_dc = np.zeros_like(periods)
        n = -1
        while np.sum(abs(dc - old_dc)) > 0.005 * periods.size and n < 10:
            n += 1
            old_dc = dc.copy()

            print('****** ITERATION ' +  str(n) + ' ******')
            m, G = inversion._inversion_iteration(setup_model, m, location)
            c = mineos._read_qfile(save_name, periods)
            dc = np.array([c[i] - obs[ic[i]] for i in range(len(c))])


        # Plot up the model that reached convergence
        plots.plot_model_simple(m, 'm' + str(n + 1), ax_m150, (0, 150))
        plots.plot_model_simple(m, 'm' + str(n + 1), ax_mDeep,
                                (150, setup_model.depth_limits[1]), False)
        p_rf = inversion._predict_RF_vals(m)
        plots.plot_rf_data(p_rf, 'm' + str(n + 1), ax_rf)
        # Run MINEOS on final model
        params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
        _ = define_models.convert_inversion_model_to_mineos_model(m, setup_model)
        c, _ = mineos.run_mineos(params, periods, setup_model.id)
        plots.plot_ph_vel_simple(periods, c, ax_c)
        dc = [c[i] - obs[ic[i]] for i in range(len(c))]
        plots.plot_ph_vel_simple(periods, dc, ax_dc)
        ax_dc.plot(periods, [0] * len(periods), 'k--')
        plots.make_plot_symmetric_in_y_around_zero(ax_dc)
        plots.make_plot_symmetric_in_y_around_zero(ax_rf)

    print(trial + 1, ' TRIALS')
    save_name = 'output/{0}/{0}'.format(setup_model.id)
    damp_s = pd.read_csv(save_name + 'damp_s.csv')
    damp_t = pd.read_csv(save_name + 'damp_t.csv')
    n = 0
    for label in ['roughness', 'to_m0', 'to_m0_grad']:
        ax_d = f.add_axes([0.025 + 0.0475 * n, 0.3, 0.0275, 0.6])
        ax_d.plot(damp_s[label], damp_s.Depth, 'ko-', markersize=3)
        ax_d.plot(damp_t[label], damp_t.Depth, 'ro', markersize=2)
        ax_d.set(title=label)
        ax_d.set_ylim([150, 0])
        ax_d.xaxis.tick_top()
        plt.rcParams.update({'axes.titlesize': 'xx-small',
                             'axes.labelsize': 'xx-small',
                             'xtick.labelsize': 'xx-small',
                             'ytick.labelsize': 'xx-small'})
        n += 1

    f.savefig(
        '/media/sf_VM_Shared/rftests/MC_{}N_{}W_{}kmLAB_{}trials.png'.format(
        location[0], -location[1], round(t_LAB), trial + 1,
        )
    )


def run_plot_inversion(setup_model, model,
                       obs, std_obs, periods, location, m, max_runs=10):

    ic = list(range(len(obs) - 2 * len(setup_model.boundaries[0])))
    i_rf = list(range(ic[-1] + 1, len(obs)))

    #setup_model = setup_model._replace(boundary_names = [])
    save_name = 'output/{0}/{0}.q'.format(setup_model.id)
    t_LAB = model.thickness.item(model.boundary_inds[-1] + 1)

    # Setup the figure and reference plots
    f, ax_c, ax_dc, ax_rf, ax_m150, ax_mDeep, ax_map = (
        plots.setup_figure_layout(location, t_LAB)
    )
    plots.plot_area_map(location, ax_map)
    #plots.plot_SL14_profile(location, ax_m150)
    #plots.plot_SL14_profile(location, ax_mDeep)
    ph_vel_pred = mineos.calculate_c_from_card(setup_model, m, periods)
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
                         (150, setup_model.depth_limits[1]), False)

        # Plot predicted RF vals of previous iteration
        p_rf = inversion._predict_RF_vals(model)
        plots.plot_rf_data(p_rf, 'm' + str(n), ax_rf)

        # Run inversion
        print('****** ITERATION ' +  str(n) + ' ******')
        model, G, o = inversion._inversion_iteration(setup_model, model, location,
                                                  (obs, std_obs, periods))

        # Plot predicted c from previous iteration (calculated for inversion)
        c = mineos._read_qfile(save_name, periods)
        dc = np.array([c[i] - obs[ic[i]] for i in range(len(c))])
        plots.plot_ph_vel(periods, c, 'm' + str(n), ax_c)
        plots.plot_dc(periods, dc, ax_dc)

        n += 1


    # Plot values for final model
    plots.plot_model(model, 'm' + str(n), ax_m150, (0, 150))
    plots.plot_model(model, 'm' + str(n), ax_mDeep,
                     (150, setup_model.depth_limits[1]), False)
    p_rf = inversion._predict_RF_vals(model)
    plots.plot_rf_data(p_rf, 'm' + str(n), ax_rf)
    c = mineos.calculate_c_from_card(setup_model, model, periods)
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
    plots.plot_damping_params(setup_model.id, f)

    # Print on the observed constraints
    obs_c_t = ['{:3.0f} s: {:.3f} {:s} {:.2f} km/s'.format(
        periods[i], obs.item(i), u'\u00B1', std_obs.item(i)
        ) for i in range(len(periods))]
    obs_rf_t = ['Moho tt: {:3.1f} {:s} {:.2f} s'.format(obs.item(-4), u'\u00B1', std_obs.item(-4)),
                'Moho dV: {:3.2f} {:s} {:.2f} km/s'.format(obs.item(-2), u'\u00B1', std_obs.item(-2)),
                ' LAB tt: {:3.1f} {:s} {:.2f} s'.format(obs.item(-3), u'\u00B1', std_obs.item(-3)),
                ' LAB dV: {:3.2f} {:s} {:.2f} km/s'.format(obs.item(-1), u'\u00B1', std_obs.item(-1))]
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
        location[0], -location[1], round(t_LAB), setup_model.id
        )
    )
    plt.close()
    return model, G, o


def try_run(location:tuple, t_LAB:float):


    sm = define_models.SetupModel('r_0_0g_05_lab5',
                                  min_layer_thickness=6,
                                  depth_limits=(0, 350))
    t, v = constraints.get_vels_ShenRitzwoller2016(location)
    d = np.cumsum(t)
    #v[0:3] = [v[3]] * 3 # remove lowest velocity layer
    #v[64:200] = [v[64]] * (200 - 64) # add in a clear high velocity lid
    #v[64:] = [v[64]] * (len(v) - 64)
    define_models._fill_in_base_of_model(t, v, sm)
    bi = np.array([63, 199])
    t = np.array(t)
    t[bi[1] + 1] = t_LAB
    v = np.hstack((
        v[:200], np.interp(sum(t[:200]) + np.cumsum(t[200:]),
                           [sum(t[:200]), sum(t)],
                           [v[200], v[-1]]
                           )
    ))
    # Smooth across base of model
    thickness, vsv, boundary_inds = define_models._return_evenly_spaced_model(
        t, v, bi, sm.min_layer_thickness,
    )

    m = define_models.InversionModel(
        vsv = vsv[np.newaxis].T,
        thickness = thickness[np.newaxis].T,
        boundary_inds = np.array(boundary_inds),
        d_inds = define_models._find_depth_indices(thickness, sm.depth_limits),
    )
    mineos_model = define_models.convert_inversion_model_to_mineos_model(m, sm)

    # Get predicted values
    rf_p = inversion._predict_RF_vals(m)
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

def loop_through_locs():

    for lat in range(34, 41, 1):#range(34, 41):
        for lon in range(-115, -105, 1):
            for t_LAB in [5.]:#[30., 25., 20., 15., 10.]:

                m, G, o = try_run((lat, lon), t_LAB)

                fname = '{}N_{}W_{}kmLAB'.format(lat, lon, t_LAB)
                define_models.save_model(m, fname)
