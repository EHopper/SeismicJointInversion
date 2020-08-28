""" Test the inversion pipeline
"""

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

skipMINEOS = False


class PipelineTest(unittest.TestCase):

    # =========================================================================
    # Set up specific assertions for locally defined classes
    # =========================================================================
    def assertSetupModelEqual(self, actual, expected):
        self.assertEqual(actual.id, expected.id)
        act_bnames, act_bwidths = actual.boundaries
        exp_bnames, exp_bwidths = expected.boundaries
        self.assertTupleEqual(act_bnames, exp_bnames)
        self.assertEqual(act_bwidths, exp_bwidths)
        self.assertTupleEqual(actual.depth_limits, expected.depth_limits)
        self.assertEqual(actual.min_layer_thickness,
                         expected.min_layer_thickness)
        self.assertEqual(actual.vsv_vsh_ratio,
                         expected.vsv_vsh_ratio)
        self.assertEqual(actual.vpv_vsv_ratio,
                         expected.vpv_vsv_ratio)
        self.assertEqual(actual.vpv_vph_ratio,
                         expected.vpv_vph_ratio)
        self.assertEqual(actual.ref_card_csv_name,
                         expected.ref_card_csv_name)

    def assertInversionModelEqual(self, actual, expected):
        np.testing.assert_allclose(actual.vsv, expected.vsv)
        # Cannot compare values == 0 to relative precision via. assert_allclose
        self.assertEqual(actual.thickness[0], expected.thickness[0])
        np.testing.assert_allclose(actual.thickness[1:],
                                   expected.thickness[1:])
        np.testing.assert_array_equal(actual.boundary_inds,
                                      expected.boundary_inds)


    # =========================================================================
    # The tests to run
    # =========================================================================


    # ************************* #
    #      define_models.py     #
    # ************************* #

    # test_SetupModel
    @parameterized.expand([
            (
                "Using default values",
                (
                    'moho lab',
                    ('Moho', 'LAB'), [3., 10.],
                    (0, 400), 6, 1, 1.75, 1,
                    'data/earth_models/prem.csv',
                ),
                define_models.SetupModel('moho lab'),
            ),
            (
                "Moho only, no defaults",
                (
                    'moho',
                    ('Moho',), [5.],
                    (-10., 100.), 10., 1.1, 1.8, 0.9,
                    'prem.csv',
                ),
                define_models.SetupModel(
                    id = 'moho',
                    boundaries = (('Moho',), [5.]),
                    depth_limits = (-10., 100.),
                    min_layer_thickness = 10.,
                    vsv_vsh_ratio = 1.1, vpv_vsv_ratio = 1.8,
                    vpv_vph_ratio = 0.9, ref_card_csv_name = 'prem.csv',
                )
            )

    ])
    def test_SetupModel(self, name, inputs, expected):
        """ Test that the SetupModel class defaults work as expected.
        """
        id, bds, bw, dl, mlt, svsh, ps, pvph, csv = inputs
        self.assertSetupModelEqual(
            define_models.SetupModel(
                id=id,
                boundaries=(bds, bw),
                min_layer_thickness=mlt, depth_limits=dl,
                vsv_vsh_ratio=svsh, vpv_vph_ratio=pvph, vpv_vsv_ratio=ps,
                ref_card_csv_name=csv,
            ),
            expected
        )

    # test_fill_in_base_of_model
    @parameterized.expand([
        (
            'basic model',
            define_models.SetupModel('testcase',
                                     ref_card_csv_name='data/earth_models/test.csv'),
            ([0], [4.3]),
            ([0.] + [400 / 66] * 66,
             [4.3] + [3.2 + i * (4.7699 - 3.2) / 66 for i in range(1, 67)])
        ),
        (
            'prem',
            define_models.SetupModel('testcase', min_layer_thickness=5.,
                                     depth_limits=(0., 80.)),
            ([0], [3.2]),
            ([0.] + [5.] * 16,
             ([3.2] * 4 + [3.9]
              + [(25. + i * 5. - 24.41) / (42.933 - 24.41) * (4.40029 - 3.9) + 3.9 for i in range(4)]
              + [(45. + i * 5. - 42.933) / (61.467 - 42.933) * (4.40456 - 4.40029) + 4.40029 for i in range(4)]
              + [(65. + i * 5. - 61.467) / (80. - 61.467) * (4.40883 - 4.40456) + 4.40456 for i in range(4)]
             )
            )
        ),
        (
            'complex starting model',
            define_models.SetupModel('testcase',
                                     ref_card_csv_name='data/earth_models/test.csv'),
            ([0., 30., 3., 40., 10.], [3.4, 3.6, 4.0, 4.2, 4.15]),
            ([0., 30., 3., 40., 10.] + [317. / 52.] * 52,
             ([3.4, 3.6, 4.0, 4.2, 4.15]
              + [(83. + i * 317. / 52.) / 400 * (4.7699 - 3.2) + 3.2 for i in range(1, 53)]
             ),
            )
        ),
        (
            'starting model matches depth lims',
            define_models.SetupModel('testcase', depth_limits=(0., 50.)),
            ([0., 5., 5., 10., 10., 10., 10.], [3.4 + 0.1 * i for i in range(7)]),
            ([0., 5., 5., 10., 10., 10., 10.], [3.4 + 0.1 * i for i in range(7)]),
        ),
        (
            'starting model exceeds depth lims',
            define_models.SetupModel('testcase', depth_limits=(0., 50.)),
            ([0., 5., 5., 10., 10., 10., 12.], [3.4 + 0.1 * i for i in range(7)]),
            ([0., 5., 5., 10., 10., 10., 10.], [3.4 + 0.1 * i for i in range(6)] + [3.9 + 10/12 * 0.1]),
        ),
    ])
    def test_fill_in_base_of_model(self, name, setup_model, inputs, expected):
        """
        """

        t, vs = inputs
        define_models._fill_in_base_of_model(t, vs, setup_model)

        exp_t, exp_vs = expected
        self.assertEqual(t, exp_t)
        np.testing.assert_almost_equal(vs, exp_vs)

    # test_add_BLs_to_starting_model
    @parameterized.expand([
        (
            'basic model',
            define_models.SetupModel('testcase'),
            [0.] + [50.] * 8,
            ([0., 50., 50., 3., 50., 50., 10., 50., 137.], [2, 5]),
        ),
        (
            'crustal model on top',
            define_models.SetupModel('testcase', depth_limits=(0., 216.)),
            [0., 5., 15., 16.] + [6.] * 30,
            ([0., 5., 15., 16.] + [6.] * 3 + [3.] + [6.] * 9 + [10.] + [6.] * 15 + [5.],
             [6, 16]),
        ),
        (
            'more complicated BLs',
            define_models.SetupModel(
                'testcase',
                (('Mid-crust', 'Moho', 'MLD', 'LAB'), [2.5, 3., 5., 10.]),
            ),
            [0.] + [5.] * 80,
            (([0.] + [5.] * 14 + [2.5] + [5.] * 13 + [3.] + [5.] * 12 + [5.]
              + [5.] * 13 + [10.] + [5.] * 23 + [4.5]),
             [14, 28, 41, 55]),
        ),
    ])
    def test_add_BLs_to_starting_model(
            self, name, setup_model, thicknesses, expected):
        """ Test the addition of the boundary layer frameworks.

        Requires some shallower model with the starting thickness and Vs
        to base the rest off.
        """

        calc_bi = define_models._add_BLs_to_starting_model(thicknesses, setup_model)

        exp_t, exp_bi = expected
        self.assertEqual(thicknesses, exp_t)
        self.assertEqual(calc_bi, exp_bi)

    # test_add_noise_to_starting_model
    @parameterized.expand([
        (
            'basic model',
            (np.array([0., 30., 3., 40., 10., 137.]),
             np.array([3.4, 3.6, 4.0, 4.2, 4.1, 4.4]),
             np.array([1, 3]),
             list(range(5))),
            42,
            (np.array([0., 29.2976, 3., 46.3169, 10., 131.3855]),
             np.array([3.4993, 3.5723, 4.1295, 4.5046, 4.0532, 4.4]),
             np.array([1, 3]),
             list(range(5))),
        ),
    ])
    def test_add_noise_to_starting_model(self, name, input, seed, expected):
        """
        """
        t, vs, bi, d_inds = input
        np.random.seed(seed)
        define_models._add_noise_to_starting_model(t, vs, bi, d_inds)

        exp_t, exp_vs, exp_bi, exp_di = expected
        np.testing.assert_allclose(np.array(t), np.array(exp_t), atol=1e-3)
        np.testing.assert_almost_equal(np.array(vs), np.array(exp_vs), decimal=4)
        self.assertEqual(list(bi), list(exp_bi))
        self.assertEqual(d_inds, exp_di)

    # test_return_evenly_spaced_model
    @parameterized.expand([
        (
            'basic model',
            (
                [0, 30., 3., 24., 10., 30.], # t
                [3.0, 3.5, 4.0, 4.6, 4.4, 4.65], # vs
                [1, 3], # bi
                6. #min layer thickness
             ),
             (
                [0., 6., 6., 6., 6., 6., 3., 6., 6., 6., 6., 10.,
                 6., 6., 6., 6., 6.], # expected t
                [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0, 4.15, 4.3,
                 4.45, 4.6, 4.4, 4.45, 4.5, 4.55, 4.6, 4.65], # expected vs
                [5, 10] # expected bi
             ),

        ),
    ])
    def test_return_evenly_spaced_model(self, name, inputs, expected):
        """
        """
        t, vs, bi, min_layer_thickness = inputs
        thickness, vsv, bound_inds = define_models._return_evenly_spaced_model(
            np.array(t), np.array(vs), np.array(bi), min_layer_thickness,
        )

        exp_t, exp_vs, exp_bi = expected
        np.testing.assert_array_equal(thickness, np.array(exp_t))
        np.testing.assert_allclose(vsv, np.array(exp_vs), rtol=0.01, atol=0.05)
        np.testing.assert_array_equal(bound_inds, np.array(exp_bi))



    # ************************* #
    #         mineos.py         #
    # ************************* #

    # test_mineos
    @parameterized.expand([
        (
            'NoMelt models and results, courtesy Josh Russell, 2019',
            'NoMeltRayleigh',
            [15.5556, 16.4706, 20, 25, 32, 40, 50, 60, 80, 100, 120, 140, 150],
        )
    ])
    @unittest.skipIf(skipMINEOS, "MINEOS runs too slow to test every time")
    def test_mineos(self, name, model_id, periods):
        """

        card = pd.read_csv('files_for_testing/mineos/' + model_id + '.card',
                           skiprows=3, header=None, sep='\s+')
        card.columns = ['r','rho', 'vpv', 'vsv', 'q_kappa', 'q_mu',
                        'vph', 'vsh', 'eta']

        n = 0
        p = periods[n]
        ke = kernels_calc[kernels_calc.period == p]
        kj = kernels_expected[kernels_expected.period == p]
        plt.plot(ke.vsv, ke.z)
        plt.plot(kj.vsv * card.vsv.values[::-1] / phv_calc[n], kj.z, '--')
        plt.gca().set_ylim([400, 0])
        maxk = 1.1 * max((kj.loc[kj.z < 400, 'vsv'].max(),
                    ke.loc[ke.z < 400, 'vsv'].max()))
        mink = min((kj.loc[kj.z < 400, 'vsv'].min(),
                    ke.loc[ke.z < 400, 'vsv'].min()))
        if mink < 0: mink *= 1.1;
        else: mink *= 0.9
        plt.gca().set_xlim([mink, maxk])
        n += 1

        p = periods[n]
        ke = kernels_calc[kernels_calc.period == p]
        plt.plot(ke.vsv, ke.z)
        plt.gca().set_ylim([400, 0])
        n += 1
        """

        # All previously calculated test outputs should be saved in
        expected_output_dir = './files_for_testing/mineos/'
        # And newly calculated test outputs should be saved in
        actual_output_dir = 'output/testcase/'

        # Copy the test card into the right directory for the workflow
        if not os.path.exists('output'):
            os.mkdir('output')
        if os.path.exists(actual_output_dir):
            shutil.rmtree(actual_output_dir)

        os.mkdir(actual_output_dir)
        shutil.copyfile(expected_output_dir + model_id + '.card',
                        actual_output_dir + 'testcase.card')

        # Run MINEOS
        params = mineos.RunParameters(
            freq_max = 1000 / min(periods) + 1,
            qmod_path = expected_output_dir + model_id + '.qmod',
        )
        phv_calc, kernels_calc = mineos.run_mineos_and_kernels(
            params, periods, 'testcase'
        )

        # Load previously calculated test outputs
        phv_expected = mineos._read_qfile(
            expected_output_dir + model_id + '.q', periods
        )
        # Something super weird with the kernels that Josh sent me
        kernels_expected = mineos._read_kernels(
            expected_output_dir + model_id, periods
        )
        kernels_expected['type'] = params.Rayleigh_or_Love

        np.testing.assert_allclose(phv_calc, phv_expected, rtol=1e-7)

        kernels_expected.drop(columns=['eta', 'rho'], inplace=True)
        kernels_calc.drop(columns=['eta', 'rho'], inplace=True)
        # Make Josh's kernels into m/s units (from 1e-3 m/s)
        kernels_expected.loc[:, ['vsv', 'vpv', 'vsh', 'vph']] *= 1e3
        # Make calculated kernels into m/s units (from km/s)
        kernels_calc.loc[:, ['vsv', 'vpv', 'vsh', 'vph']] *= 1e-3
        # check_less_precise is the number of sig figs that must match
        pd.testing.assert_frame_equal(kernels_calc, kernels_expected,
            check_exact=False, check_less_precise=1,
        )




    # ************************* #
    #       inversion.py        #
    # ************************* #

    # test_G
    @parameterized.expand([
        (
            'basic model',
            define_models.SetupModel('testcase', depth_limits=(0, 200)),
            (35, -110),
            [10, 20, 30, 40, 60, 80, 100],
            [1.05] + [1] * 33,
        ),
        (
            'more perturbed',
            define_models.SetupModel('testcase', depth_limits=(0, 150)),
            (35, -120),
            [5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 120],
            ([1.025] * 5 + [0.975] * 5 + [1.02] * 5 + [0.99] * 5 + [1.06] * 2 + [1]
            + [1.1] * 2),
        ),
        (
            'small perturbations',
            define_models.SetupModel('testcase', depth_limits=(0, 175)),
            (35, -100),
            [10, 20, 30, 40, 60, 80, 100],
            ([1.005] * 5 + [0.995] * 5 + [1.01] * 5 + [0.997] * 5
            + [1.001] * 4 + [0.999] * 2 + [1]
            + [1.05] * 2),
        ),
        (
            'sharp LAB',
            define_models.SetupModel(
                'testcase', boundaries=(('Moho', 'LAB'), [3., 3.]),
                depth_limits=(0, 150)
            ),
            (35, -110),
            [10, 20, 30, 40, 60, 80, 100],
            ([1.005] * 5 + [0.995] * 5 + [1.01] * 5 + [0.997] * 5 + [1.0001] * 4 + [1]
            + [1.05] * 2),
        ),
    ])
    @unittest.skipIf(skipMINEOS, "MINEOS runs too slow to test every time")
    def test_G_sw(self, name, setup_model, location, periods, model_perturbation):
        """ Test G by comparing the dV from G * dm to the dV output from MINEOS.

        From a given starting model, calculate phase velocities and kernels
        in MINEOS, and convert to the inversion G.

        Convert the starting model to the column vector.  Perturb the column
        vector in some way, and then convert it back to a MINEOS card.  Rerun
        MINEOS to get phase velocities.

        Note that the mistmatch between these two methods gets worse for larger model perturbations, especially at shorter periods.

        As G is dV/dm, then G * dm == mineos(m_perturbed) - mineos(m0).

        Plot this comparison:
        plt.figure(figsize=(10,8))
        a1 = plt.subplot(1, 3, 1)
        plt.plot(model.vsv, np.cumsum(model.thickness), 'b-o')
        plt.plot(model_perturbed.vsv, np.cumsum(model_perturbed.thickness), 'r-o')
        plt.title('Vsv Models\nblue: original; red: perturbed')
        plt.gca().set_ylim([200, 0])
        plt.xlabel('Vsv (km/s)')
        plt.ylabel('Depth (km)')
        a2 = plt.subplot(2, 2, 2)
        im = plt.scatter(dc_mineos, dc_from_Gdm, 10, periods)
        plt.xlabel('dc from MINEOS')
        plt.ylabel('dc from G calculation')
        cbar = plt.gcf().colorbar(im)
        cbar.set_label('Periods (s)', rotation=90)
        a3 = plt.subplot(2, 2, 4)
        im = plt.scatter(periods,
                         (dc_from_Gdm - dc_mineos) / dc_mineos, 10, dc_mineos)
        plt.xlabel('Period (s)')
        plt.ylabel('(Gdm - dc) / dc')
        cbar2 = plt.gcf().colorbar(im)
        cbar2.set_label('dc from MINEOS (km/s)', rotation=90)
        """
        # Calculate the G matrix from a starting model
        np.random.seed(42) # need to set the seed because model length can be
                           # different depending on the random variation
        model = define_models.setup_starting_model(setup_model, location)
        # No need for output on MINEOS model as saved to .card file
        _ = define_models.convert_inversion_model_to_mineos_model(
            model, setup_model
        )
        params = mineos.RunParameters(
            freq_max = 1000 / min(periods) + 1, max_run_N = 5,
        )
        ph_vel_pred, kernels = mineos.run_mineos_and_kernels(
            params, periods, setup_model.id
        )
        G = partial_derivatives._build_partial_derivatives_matrix_sw(
            kernels, model, setup_model,
        )

        # Apply the perturbation
        p = inversion._build_model_vector(model, setup_model.depth_limits)
        p_perturbed = p.copy() * np.array(model_perturbation)[:, np.newaxis]
        perturbation = p_perturbed - p
        model_perturbed = inversion._build_inversion_model_from_model_vector(
            p_perturbed, model
        )
        _ = define_models.convert_inversion_model_to_mineos_model(
            model_perturbed, setup_model._replace(id='testcase_perturbed')
        )
        ph_vel_perturbed, _ = mineos.run_mineos(
            params, periods, 'testcase_perturbed'
        )

        # calculate dv
        dc_mineos = ph_vel_perturbed - ph_vel_pred
        dc_from_Gdm = np.matmul(G, perturbation).flatten()

        np.testing.assert_allclose(
            dc_mineos, dc_from_Gdm, atol=0.01, rtol=0.05,
        )


    # ************************* #
    #   partial_derivatives.py  #
    # ************************* #

    # test_build_MINEOS_G_matrix
    @parameterized.expand([
        (
            'simple',
            pd.DataFrame({
                'z': [0, 10, 20, 0, 10, 20],
                'period': [10, 10, 10, 30, 30, 30],
                'vsv': [0.11, 0.12, 0.13, 0.101, 0.102, 0.103],
                'vpv': [0.21, 0.22, 0.23, 0.201, 0.202, 0.203],
                'vsh': [0.31, 0.32, 0.33, 0.301, 0.302, 0.303],
                'vph': [0.41, 0.42, 0.43, 0.401, 0.402, 0.403],
                'eta': [0.51, 0.52, 0.53, 0.501, 0.502, 0.503],
                'rho': [0.61, 0.62, 0.63, 0.601, 0.602, 0.603],
                'type': ['Rayleigh'] * 6,
            }),
            np.array([
                [0.11, 0.12, 0.13, 0, 0, 0, 0.21, 0.22, 0.23,
                 0.41, 0.42, 0.43, 0.51, 0.52, 0.53],
                [0.101, 0.102, 0.103, 0, 0, 0, 0.201, 0.202, 0.203,
                 0.401, 0.402, 0.403, 0.501, 0.502, 0.503],
            ])
        ),
    ])
    def test_build_MINEOS_G_matrix(self, name, kernels, expected_G_MINEOS):
        """
        """

        np.testing.assert_allclose(
            partial_derivatives._build_MINEOS_G_matrix(kernels),
            expected_G_MINEOS
        )

    # test_calculate_dm_ds
    @parameterized.expand([
        (
            'Simple',
            define_models.InversionModel(
                vsv = np.array([[3.5, 4, 4.5, 4.6, 4.7]]).T,
                thickness = np.array([[0., 30., 40., 50., 20.]]).T,
                boundary_inds = [], d_inds=[],
            ),
            np.arange(0, 150, 10),
            np.array([
                [1., 0., 0., 0., 0.],
                [2./3., 1./3., 0, 0, 0.],
                [1./3., 2./3., 0, 0, 0.],
                [0, 1., 0, 0, 0.],
                [0, 3./4., 1./4., 0, 0.],
                [0, 1./2., 1./2., 0, 0.],
                [0, 1./4., 3./4., 0, 0.],
                [0, 0, 1., 0, 0.],
                [0, 0, 4./5., 1./5., 0.],
                [0, 0, 3./5., 2./5., 0.],
                [0, 0, 2./5., 3./5., 0.],
                [0, 0, 1./5., 4./5, 0.],
                [0, 0, 0, 1., 0.],
                [0, 0, 0, 1./2., 1./2.],
                [0, 0, 0, 0, 1.],
            ]),
        ),
        (
            'Moho & LAB',
            define_models.InversionModel(
                vsv = np.array([[3.5, 3.6, 4., 4.2, 4.4, 4.3, 4.35, 4.4]]).T,
                thickness = np.array([[0., 30., 10., 22., 22., 15., 22., 22.]]).T,
                boundary_inds = np.array([1, 4]), d_inds=[],
            ),
            np.arange(0, 150, 10),
            np.array([
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [2./3., 1./3., 0., 0., 0., 0., 0., 0.],
                [1./3., 2./3., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 12./22., 10./22., 0., 0., 0., 0.],
                [0., 0., 2./22., 20./22, 0., 0., 0., 0.],
                [0., 0., 0., 14./22., 8./22., 0., 0., 0.],
                [0., 0., 0., 4./22., 18./22., 0., 0., 0.],
                [0., 0., 0., 0., 9./15., 6./15., 0., 0.],
                [0., 0., 0., 0., 0., 21./22., 1./22., 0.],
                [0., 0., 0., 0., 0., 11./22., 11./22., 0.],
                [0., 0., 0., 0., 0., 1./22., 21./22., 0.],
                [0., 0., 0., 0., 0., 0., 13./22., 9./22.],
                [0., 0., 0., 0., 0., 0., 3./22., 19./22.],
            ]),
        ),
    ])
    def test_calculate_dm_ds(self, name, model, depth, expected):
        """
        """
        np.testing.assert_allclose(
            partial_derivatives._calculate_dm_ds(model, depth),
            expected
        )

    # test_convert_kernels_d_shallowerm_by_d_s
    @parameterized.expand([
        (
            'Simple',
            define_models.InversionModel(
                vsv = np.array([[3.5, 4, 4.5, 4.6, 4.7]]).T,
                thickness = np.array([[0, 30, 40, 50, 10]]).T,
                boundary_inds = [], d_inds=[],
            ),
            np.arange(0, 150, 10),
            np.array([
                [0, 0, 0, 0],
                [0, 10/30, 0, 0],
                [0, 20/30, 0, 0],
                [0, 30/30, 0, 0],
                [0, 0, 10/40, 0],
                [0, 0, 20/40, 0],
                [0, 0, 30/40, 0],
                [0, 0, 40/40, 0],
                [0, 0, 0, 10/50],
                [0, 0, 0, 20/50],
                [0, 0, 0, 30/50],
                [0, 0, 0, 40/50],
                [0, 0, 0, 50/50],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ])

        ),
    ])
    def test_convert_kernels_d_shallowerm_by_d_s(
        self, name, model, depth, expected_dm_ds):
        """
        """
        n_layers = model.vsv.size - 1
        dm_ds_mat = np.zeros((depth.size, n_layers))
        for i in range(1, n_layers):
            dm_ds_mat = (
                partial_derivatives._convert_kernels_d_shallowerm_by_d_s(
                model, i, depth, dm_ds_mat
                )
            )
        np.testing.assert_allclose(dm_ds_mat, expected_dm_ds)

    # test_convert_kernels_d_deeperm_by_d_s
    @parameterized.expand([
        (
            'Simple',
            define_models.InversionModel(
                vsv = np.array([[3.5, 4, 4.5, 4.6, 4.7]]).T,
                thickness = np.array([[0, 30, 40, 55, 10]]).T,
                boundary_inds = [], d_inds=[],
            ),
            np.arange(0, 150, 10),
            np.array([
                [30/30, 0, 0, 0],
                [20/30, 0, 0, 0],
                [10/30, 0, 0, 0],
                [0, 40/40, 0, 0],
                [0, 30/40, 0, 0],
                [0, 20/40, 0, 0],
                [0, 10/40, 0, 0],
                [0, 0, 55/55, 0],
                [0, 0, 45/55, 0],
                [0, 0, 35/55, 0],
                [0, 0, 25/55, 0],
                [0, 0, 15/55, 0],
                [0, 0, 5/55, 0],
                [0, 0, 0, 5/10],
                [0, 0, 0, 0],
            ])

        ),
    ])
    def test_convert_kernels_d_deeperm_by_d_s(
        self, name, model, depth, expected_dm_ds):
        """
        """
        n_layers = model.vsv.size - 1
        dm_ds_mat = np.zeros((depth.size, n_layers))
        for i in range(n_layers):
            dm_ds_mat = (
                partial_derivatives._convert_kernels_d_deeperm_by_d_s(
                model, i, depth, dm_ds_mat
                )
            )
        np.testing.assert_allclose(dm_ds_mat, expected_dm_ds)

    # test_calculate_dm_dt
    @parameterized.expand([
        (
            'Moho & LAB',
            define_models.InversionModel(
                vsv = np.array([[3.5, 3.6, 4., 4.2, 4.4, 4.3, 4.35, 4.4]]).T,
                thickness = np.array([[0., 30., 10., 22., 21., 15., 23., 24.]]).T,
                boundary_inds = np.array([1, 4]), d_inds=[],
            ),
            np.arange(0, 150, 10),
            np.array([
                [0, 0], #   0 - pegged
                [(3.5 - 3.6) * 10 / 30 ** 2, 0], #  10
                [(3.5 - 3.6) * 20 / 30 ** 2, 0], #  20
                [(3.6 - 4.) / 10, 0], #  30
                [(4. - 4.2) * 22 / 22 ** 2, 0], #  40
                [(4. - 4.2) * 12 / 22 ** 2, 0], #  50
                [(4. - 4.2) * 2 / 22 ** 2, 0], #  60
                [0, (4.2 - 4.4) * 8 / 21 ** 2], #  70
                [0, (4.2 - 4.4) * 18 / 21 ** 2], #  80
                [0, (4.4 - 4.3) / 15], #  90
                [0, (4.3 - 4.35) * 21 / 23 ** 2 ], # 100
                [0, (4.3 - 4.35) * 11 / 23 ** 2], # 110
                [0, (4.3 - 4.35) * 1 / 23 ** 2], # 120
                [0, 0], # 130 - pegged
                [0, 0], # 140 - pegged
            ])
        ),
    ])
    def test_calculate_dm_dt(self, name, model, depth, expected):
        """
        """
        np.testing.assert_allclose(
            partial_derivatives._calculate_dm_dt(model, depth),
            expected
        )

    # test_convert_to_model_kernels
    @parameterized.expand([
        (
            'Moho & LAB',
            define_models.InversionModel(
                vsv = np.array([[3.5, 3.6, 4., 4.2, 4.4, 4.3, 4.35, 4.4]]).T,
                thickness = np.array([[0., 30., 10., 22., 21., 15., 23., 24.]]).T,
                boundary_inds = np.array([1, 4]), d_inds=[],
            ),
            np.arange(0, 150, 10),
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #   0
                [2/3, 1/3, 0, 0, 0, 0, 0, 0, (3.5 - 3.6) * 10 / 30 ** 2, 0], #  10
                [1/3, 2/3, 0, 0, 0, 0, 0, 0, (3.5 - 3.6) * 20 / 30 ** 2, 0], #  20
                [0, 1, 0, 0, 0, 0, 0, 0, (3.6 - 4.) / 10, 0], #  30
                [0, 0, 1, 0, 0, 0, 0, 0, (4. - 4.2) * 22 / 22 ** 2, 0], #  40
                [0, 0, 12/22, 10/22, 0, 0, 0, 0, (4. - 4.2) * 12 / 22 ** 2, 0], #  50
                [0, 0, 2/22, 20/22, 0, 0, 0, 0, (4. - 4.2) * 2 / 22 ** 2, 0], #  60
                [0, 0, 0, 13/21, 8/21, 0, 0, 0, 0, (4.2 - 4.4) * 8 / 21 ** 2], #  70
                [0, 0, 0, 3/21, 18/21, 0, 0, 0, 0, (4.2 - 4.4) * 18 / 21 ** 2], #  80
                [0, 0, 0, 0, 8/15, 7/15, 0, 0, 0, (4.4 - 4.3) / 15], #  90
                [0, 0, 0, 0, 0, 21/23, 2/23, 0, 0, (4.3 - 4.35) * 21 / 23 ** 2 ], # 100
                [0, 0, 0, 0, 0, 11/23, 12/23, 0, 0, (4.3 - 4.35) * 11 / 23 ** 2], # 110
                [0, 0, 0, 0, 0, 1/23, 22/23, 0, 0, (4.3 - 4.35) * 1 / 23 ** 2], # 120
                [0, 0, 0, 0, 0, 0, 15/24, 9/24, 0, 0], # 130 - pegged
                [0, 0, 0, 0, 0, 0, 5/24, 19/24, 0, 0], # 140 - pegged
            ])
        )
    ])
    def test_convert_to_model_kernels(self, name, model, depth, expected):
        """
        """

        np.testing.assert_allclose(
            partial_derivatives._convert_to_model_kernels(depth, model),
            expected
        )

    # test_scale_dvsv_dp_to_other_variables
    @parameterized.expand([
        (
            'Simple',
            np.array([
                [30/30, 0, 0, 0],
                [20/30, 10/30, 0, 0],
                [10/30, 20/30, 0, 0],
                [0, 40/40, 0, 0],
            ]),
            define_models.SetupModel('basic',
                vsv_vsh_ratio = 1.1, vpv_vsv_ratio = 1.8,
                vpv_vph_ratio = 0.9, ref_card_csv_name = 'prem.csv'
            ),
            np.array([
                [1, 0, 0, 0],     # vsv
                [2/3, 1/3, 0, 0],
                [1/3, 2/3, 0, 0],
                [0, 1, 0, 0],
                [1/1.1, 0, 0, 0],     # vsh
                [2/3/1.1, 1/3/1.1, 0, 0],
                [1/3/1.1, 2/3/1.1, 0, 0],
                [0, 1/1.1, 0, 0],
                [1.8, 0, 0, 0],     # vpv
                [2/3*1.8, 1/3*1.8, 0, 0],
                [1/3*1.8, 2/3*1.8, 0, 0],
                [0, 1.8, 0, 0],
                [2, 0, 0, 0],     # vph
                [4/3, 2/3, 0, 0],
                [2/3, 4/3, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],    # eta
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        ),
    ])
    def test_scale_dvsv_dp_to_other_variables(self, name, dvsv_dp,
                                              setup_model, expected):
        """
        """
        np.testing.assert_allclose(
            partial_derivatives._scale_dvsv_dp_to_other_variables(
                dvsv_dp, setup_model
            ),
            expected,
        )


    # ************************* #
    #         weights.py        #
    # ************************* #

    # test_set_layer_values
    @parameterized.expand([
        (
            'simple',
            define_models.ModelLayerIndices(
                np.arange(2), np.arange(2, 5), np.arange(5, 10),
                np.arange(10, 16), np.arange(16, 20),
                np.arange(0, 16 * 3, 3)
            ),
            (1, 2, [3, 4, 5, 6, 7], 2, [1, 2, 3, 4]),
            pd.DataFrame({
                'Depth': np.arange(0, 16 * 3, 3),
                'simple': [1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 2, 2, 2, 2, 2, 2]
            }),
            pd.DataFrame({
                'Depth': [0, 1, 2, 3],
                'simple': [1, 2, 3, 4]
            })
        )
    ])
    def test_set_layer_values(self, name, layers, damping,
                                  expected_s, expected_t):
        damp_s = pd.DataFrame({'Depth': layers.depth})
        damp_t = pd.DataFrame({'Depth': np.arange(len(layers.boundary_layers))})
        weights._set_layer_values(damping, layers, damp_s, damp_t, name)

        pd.testing.assert_frame_equal(damp_s, expected_s)
        pd.testing.assert_frame_equal(damp_t, expected_t)

    # test_damp_constraints
    @parameterized.expand([
        (
            'simple_square',
            np.array([
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [1, 8, 3, 1, 2, 3, 1, 2, 3],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [1, 2, 2, 1, 2, 3, 1, 2, 3],
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [0, 8, 2, 0, 1, 2, 0, 1, 2]
            ]),
            np.array([10, 20, 10, 20, 10, 30, 10, 40, 10])[:, np.newaxis],
            pd.DataFrame({'simple_square': [1] * 2 + [2] + [4, 5, 6, 7]}),
            pd.DataFrame({'simple_square': [1, 2]}),
            np.array([
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [2, 16, 6, 2, 4, 6, 2, 4, 6],
                [0, 4, 8, 0, 4, 8, 0, 4, 8],
                [5, 10, 10, 5, 10, 15, 5, 10, 15],
                [6, 12, 18, 6, 12, 18, 6, 12, 18],
                [0, 7, 14, 0, 7, 14, 0, 7, 14],
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [0, 16, 4, 0, 2, 4, 0, 2, 4]
            ]),
            np.array([10, 20, 20, 80, 50, 180, 70, 40, 20])[:, np.newaxis],
        ),
    ])
    def test_damp_constraints(self, name, H, h, damp_s, damp_t,
                              expected_H, expected_h):

        H, h = weights._damp_constraints((H, h, name), damp_s, damp_t)

        np.testing.assert_array_equal(H, expected_H)
        np.testing.assert_array_equal(h, expected_h)

    # test_build_smoothing_constraints
    @parameterized.expand([
        (
            'simple',
            define_models.InversionModel(
                vsv = np.array([3] * 13)[:, np.newaxis],
                thickness = np.array(
                    [0] + [6] * 3
                    + [7, 5, 4] + [6] * 1
                    + [10, 15, 12] + [6] * 1)[:, np.newaxis],
                boundary_inds = np.array([4, 8]), d_inds=[],
            ),
            np.array([
                [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                [6, -12,   6,   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  6.,-12.,  6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  7.,-13.,  6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  7.,-13.,  6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  6.,-10.,  4.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  6.,-10.,  4.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  0., 10.,-16.,  6.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  0., 10.,-16.,  6.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  6.,-18., 12.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  6.,-18., 12.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 10.,-16.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            ]),
            np.array([0] * 11 + [-6 * 4.411829121] + [0] * 2)[:, np.newaxis],
        ),
    ])
    def test_build_smoothing_constraints(self, name, model,
                                         expected_H, expected_h):

        sm = define_models.SetupModel('junk') # need PREM path only
        H, h, label = weights._build_smoothing_constraints(model, sm)

        self.assertEqual(label, 'roughness')
        np.testing.assert_allclose(H, expected_H)
        np.testing.assert_allclose(h, expected_h)

        n_bls = model.boundary_inds.shape[0]
        n_layers = expected_h.shape[0] - n_bls
        damp_s = pd.DataFrame({'Depth': np.cumsum(model.thickness)})
        damp_t = pd.DataFrame({'Depth': [0] * n_bls})
        layers = define_models.ModelLayerIndices(
            np.arange(2), np.arange(2, 3), np.arange(3, 4),
            np.arange(4, n_layers), np.arange(n_layers, n_layers + n_bls),
            np.arange(0)
        )
        weights._set_layer_values(
            (1, 1, 1, 1, 10), layers, damp_s, damp_t, 'roughness',
        )
        H_sc, h_sc = weights._damp_constraints((H, h, label), damp_s, damp_t)
        self.assertEqual(label, 'roughness')
        np.testing.assert_allclose(H_sc, expected_H)
        np.testing.assert_allclose(h_sc, expected_h)

    # test_build_constraint_damp_to_m0
    @parameterized.expand([
        (
            'simple',
            np.array([1, 2, 3, 4, 5])[:, np.newaxis],
            np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]),
            np.array([1, 2, 3, 4, 5])[:, np.newaxis],
        ),
    ])
    def test_build_constraint_damp_to_m0(self, name, p, expected_H, expected_h):
        H, h, label = weights._build_constraint_damp_to_m0(p)
        self.assertEqual(label, 'to_m0')
        np.testing.assert_array_equal(H, expected_H)
        np.testing.assert_array_equal(h, expected_h)

    # test_build_constraint_damp_original_gradient
    @parameterized.expand([
        (
            'simple',
            define_models.InversionModel(
                vsv = np.array([1, 2, 3, 4, 5, 6., 7., 8.])[:, np.newaxis],
                thickness = np.array([6., 6., 6., 6., 6., 6., 6., 6.])[:, np.newaxis],
                boundary_inds = np.array([2, 4]), d_inds=[],
            ),
            np.array([
                [1./1., -1./2., 0, 0, 0, 0, 0, 0, 0],
                [0, 1./2., -1./3., 0, 0, 0, 0, 0, 0],
                [0, 0, 1./3., -1./4., 0, 0, 0, 0, 0],
                [0, 0, 0, 1./4., -1./5., 0, 0, 0, 0],
                [0, 0, 0, 0, 1./5., -1./6., 0, 0, 0],
                [0, 0, 0, 0, 0, 1./6., -1./7., 0, 0],
                [0, 0, 0, 0, 0, 0, 1./7., 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])[:, np.newaxis],
        ),
    ])
    def test_build_constraint_damp_original_gradient(self, name, model,
                                              expected_H, expected_h):
        H, h, label = weights._build_constraint_damp_original_gradient(model)
        self.assertEqual(label, 'to_m0_grad')
        np.testing.assert_array_equal(H, expected_H)
        np.testing.assert_array_equal(h, expected_h)



if __name__ == "__main__":
    unittest.main()
