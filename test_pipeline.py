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
skip_influx = True

class PipelineTest(unittest.TestCase):

    # =========================================================================
    # Set up specific assertions for locally defined classes
    # =========================================================================
    def assertSetupModelEqual(self, actual, expected):
        self.assertEqual(actual.id,
                         expected.id)
        np.testing.assert_array_equal(actual.boundary_depths,
                                      expected.boundary_depths)
        np.testing.assert_array_equal(actual.boundary_depth_uncertainty,
                                      expected.boundary_depth_uncertainty)
        np.testing.assert_array_equal(actual.boundary_widths,
                                      expected.boundary_widths)
        np.testing.assert_array_equal(actual.boundary_vsv,
                                      expected.boundary_vsv)
        np.testing.assert_array_equal(actual.depth_limits,
                                      expected.depth_limits)
        self.assertTupleEqual(actual.boundary_names,
                              expected.boundary_names)
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
        self.assertEqual(actual.ref_sed_thickness_name,
                         expected.ref_sed_thickness_name)

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
                    'moho lab', (35, -120), np.array([30, 100.]),
                     np.array([3, 10]),
                    np.array([5, 10]), np.array([3.5, 4, 4.3, 4.25]),
                    np.array([0, 200]), ('Moho', 'LAB'), 6, 1, 1.75, 1,
                    'data/earth_models/prem.csv',
                    'data/earth_models/sedthk_crust1.csv'
                ),
                define_models.SetupModel(
                    id = 'moho lab', location = (35, -120),
                    boundary_depth_uncertainty = np.array([3, 10]),
                    boundary_depths = np.array([30, 100]),
                    boundary_widths = np.array([5, 10]),
                    boundary_vsv = np.array([3.5, 4, 4.3, 4.25]),
                    depth_limits = np.array([0, 200]),
                )
            ),
            (
                "Moho only, no defaults",
                (
                    'moho', (-10., 100.), np.array([30]),
                    np.array([2.]), np.array([5]),
                    np.array([3.5, 4]), np.array([0, 200]), ('Moho',),
                    10, 1.1, 1.8, 0.9, 'prem.csv', 'sed.csv'
                ),
                define_models.SetupModel(
                    id = 'moho', location = (-10., 100.),
                    boundary_depths = np.array([30.]),
                    boundary_depth_uncertainty = np.array([2.]),
                    boundary_widths = np.array([5.]),
                    boundary_vsv = np.array([3.5, 4.]),
                    depth_limits = np.array([0., 200.]),
                    boundary_names = ('Moho',), min_layer_thickness = 10.,
                    vsv_vsh_ratio = 1.1, vpv_vsv_ratio = 1.8,
                    vpv_vph_ratio = 0.9, ref_card_csv_name = 'prem.csv',
                    ref_sed_thickness_name = 'sed.csv',
                )
            )

    ])
    def test_SetupModel(self, name, inputs, expected):
        """ Test that the SetupModel class defaults work as expected.
        """
        id, loc, bd, bdu, bw, bv, dl, bn, mlt, svsh, ps, pvph, csv, sed = inputs
        self.assertSetupModelEqual(
            define_models.SetupModel(
                id=id, location=loc, boundary_depths=bd,
                boundary_depth_uncertainty=bdu,
                boundary_widths=bw, boundary_vsv=bv, boundary_names=bn,
                min_layer_thickness=mlt, depth_limits=dl,
                vsv_vsh_ratio=svsh, vpv_vph_ratio=pvph, vpv_vsv_ratio=ps,
                ref_card_csv_name=csv,
                ref_sed_thickness_name=sed,
            ),
            expected
        )

    # test_setup_starting_model
    @parameterized.expand([
        (
            'basic model',
            define_models.SetupModel(
                'testcase', (35., -110.), np.array([35., 90.]), np.array([3, 10]),
                np.array([5, 20]), np.array([3.5, 4.0, 4.2, 4.1]),
                np.array([0, 200])
            ),
            define_models.InversionModel(
                vsv = np.array([[
                    3.2       , 3.27230769, 3.34461538, 3.41692308, 3.5       ,
                    4.        , 4.07230769, 4.10566016, 4.13901263, 4.2       ,
                    4.1       , 4.13335247, 4.15500413, 4.17665579, 4.19830745,
                    4.21995911, 4.24161077, 4.26326243, 4.28491409, 4.30656575,
                    4.32821741, 4.34986907, 4.37152073, 4.39317239, 4.41482405,
                    4.43647571]]).T,
                thickness = np.array([[
                    0.        ,  7.83333333,  7.83333333,  7.83333333,  9.,
                    5.        ,  9.        ,  8.75      ,  8.75      , 16.,
                    20.        , 16.        ,  6.        ,  6.        ,  6.,
                    6.        ,  6.        ,  6.        ,  6.        ,  6.,
                    6.        ,  6.        ,  6.        ,  6.        ,  6.,
                    6.        ]]).T,
                boundary_inds = np.array([4, 9])
            )
        ),
        ( # the code doesn't work well for overlapping BLs, or for a shallow
          # velocity constraint that is slower than the PREM surface Vs
          # e.g. see strong negative gradient at top of the output here
          # This may be something to revisit if we were interested in doing
          # crustal structure.
          # If just using Moho and LAB, still need to be careful that the
          # lower bound of the Moho and the upper bound of the LAB don't overlap
          # - code doesn't break if they do, but the model makes no sense
          # Something that should be fixed!
            'complicated model',
            define_models.SetupModel(
                'testcase', (35., -100.), np.array([10, 35., 90., 170]),
                np.array([1, 1, 10, 20]),
                np.array([1, 5, 20, 40]),
                np.array([3.0, 3.2, 3.5, 4.0, 4.2, 4.1, 4.5, 4.6]),
                np.array([0, 400])
            ),
            define_models.InversionModel(
                vsv = np.array([[
                    3.2       , 3.14736842, 3.        , 3.2       , 3.14736842,
                    3.3354386 , 3.5       , 4.        , 4.18807018, 4.19025451,
                    4.19243884, 4.19462318, 4.2       , 4.1       , 4.10218433,
                    4.19578802, 4.5       , 4.6       , 4.69360369, 4.6961469 ,
                    4.69869011, 4.70123332, 4.70377653, 4.70631974, 4.70886295,
                    4.71140616, 4.71394937, 4.71649258, 4.71903579, 4.721579  ,
                    4.72412221, 4.72666542, 4.72920863, 4.73175184, 4.73429505,
                    4.73683826, 4.73938147, 4.74192468, 4.7444679 , 4.74701111,
                    4.74955432, 4.75209753, 4.75464074, 4.75718395, 4.75972716,
                    4.76227037, 4.76481358, 4.76735679, 4.7699]]).T,
                thickness = np.array([[
                    0.  ,  2.5       ,  7.        ,  1.        ,  7.,
                    8.  ,  7.        ,  5.        ,  7.        ,  6.5,
                    6.5 ,  6.5       , 16.        , 20.        , 16.,
                    8.  , 26.        , 40.        , 26.        ,  6.13333333,
                    6.13333333, 6.13333333, 6.13333333, 6.13333333, 6.13333333,
                    6.13333333, 6.13333333, 6.13333333, 6.13333333, 6.13333333,
                    6.13333333, 6.13333333, 6.13333333, 6.13333333, 6.13333333,
                    6.13333333, 6.13333333, 6.13333333, 6.13333333, 6.13333333,
                    6.13333333, 6.13333333, 6.13333333, 6.13333333, 6.13333333,
                    6.13333333, 6.13333333, 6.13333333, 6.13333333]]).T,
                boundary_inds = np.array([2, 6, 12, 16])
            )
        )

    ])
    def test_setup_starting_model(self, name, setup_model, expected):
        """ Test conversion from SetupModel to InversionModel

        This works as it is told to, but is hampered by...
        1) Can't deal with overlapping boundaries
        2) It is pinned at the surface and at depth to PREM (or whatever other
           ERM it is pointed to in SetupModel)

        To plot up the models and compare them:

        plt.plot(model.vsv, np.cumsum(model.thickness), 'b-')
        plt.plot(model.vsv, np.cumsum(model.thickness), 'b.')
        plt.gca().set_ylim(setup_model.depth_limits[[1, 0]])
        top = setup_model.boundary_depths - setup_model.boundary_widths/2
        bot = setup_model.boundary_depths + setup_model.boundary_widths/2
        plt.plot(
            np.hstack((setup_model.boundary_vsv[::2],
                       setup_model.boundary_vsv[1::2])),
            np.hstack((top, bot)), 'r*'
        )
        vslims = np.hstack((min(model.vsv), max(model.vsv)))
        tt = top - setup_model.boundary_depth_uncertainty
        bb = bot + setup_model.boundary_depth_uncertainty
        mlt = setup_model.min_layer_thickness
        for ib in range(len(setup_model.boundary_depths)):
            plt.plot(vslims, tt[[ib, ib]], 'k:')
            plt.plot(vslims, bb[[ib, ib]], 'k:')
            plt.plot(vslims, tt[[ib, ib]] - mlt, ':', color='0.5')
            plt.plot(vslims, bb[[ib, ib]] + mlt, ':', color='0.5')

        """
        model = define_models.setup_starting_model(setup_model)
        self.assertInversionModelEqual(model, expected)



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
            define_models.SetupModel(
                'testcase', (35., -110), np.array([35., 90.]), np.array([3, 10]),
                np.array([5, 20]), np.array([3.5, 4.0, 4.2, 4.1]),
                np.array([0, 200])
            ),
            [10, 20, 30, 40, 60, 80, 100],
            [1.05] + [1] * 26,
        ),
        (
            'more perturbed',
            define_models.SetupModel(
                'testcase', (35., -110), np.array([25., 120.]), np.array([5, 20]),
                np.array([10, 30]), np.array([3.6, 4.0, 4.4, 4.3]),
                np.array([0, 300])
            ),
            [5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 120],
            ([1.05] * 5 + [0.95] * 5 + [1.02] * 5 + [0.99] * 5
            + [1.06] * 5 + [0.97] * 5 + [1.01] * 5 + [1]
            + [1.1] * 2),
        ),
        (
            'small perturbations',
            define_models.SetupModel(
                'testcase', (35., -110), np.array([35., 90.]), np.array([3, 10]),
                np.array([5, 20]), np.array([3.5, 4.0, 4.2, 4.1]),
                np.array([0, 200])
            ),
            [10, 20, 30, 40, 60, 80, 100],
            ([1.005] * 5 + [0.995] * 5 + [1.01] * 5 + [0.997] * 5
            + [1.01] * 4 + [1]
            + [1.05] * 2),
        ),
        (
            'sharp LAB',
            define_models.SetupModel(
                'testcase', (35., -110), np.array([35., 90.]), np.array([3, 10]),
                np.array([5, 10]), np.array([3.5, 4.0, 4.2, 4.1]),
                np.array([0, 200])
            ),
            [10, 20, 30, 40, 60, 80, 100],
            ([1.005] * 5 + [0.995] * 5 + [1.01] * 5 + [0.997] * 5
            + [1.01] * 5 + [1]
            + [1.05] * 2),
        ),

    ])
    @unittest.skipIf(skipMINEOS, "MINEOS runs too slow to test every time")
    def test_G_sw(self, name, setup_model, periods, model_perturbation):
        """ Test G by comparing the dV from G * dm to the dV output from MINEOS.

        From a given starting model, calculate phase velocities and kernels
        in MINEOS, and convert to the inversion G.

        Convert the starting model to the column vector.  Perturb the column
        vector in some way, and then convert it back to a MINEOS card.  Rerun
        MINEOS to get phase velocities.

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
        model = define_models.setup_starting_model(setup_model)
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
        p = inversion._build_model_vector(model)
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
        dv_mineos = ph_vel_perturbed - ph_vel_pred
        dv_from_Gdm = np.matmul(G, perturbation).flatten()

        np.testing.assert_allclose(
            dv_mineos, dv_from_Gdm, atol=0.01, rtol=0.05,
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
                thickness = np.array([[0, 30, 40, 50, 20]]).T,
                boundary_inds = []
            ),
            np.arange(0, 150, 10),
            np.array([
                [1., 0., 0., 0.],
                [2./3., 1./3., 0, 0],
                [1./3., 2./3., 0, 0],
                [0, 1., 0, 0],
                [0, 3./4., 1./4., 0],
                [0, 1./2., 1./2., 0],
                [0, 1./4., 3./4., 0],
                [0, 0, 1., 0],
                [0, 0, 4./5., 1./5.],
                [0, 0, 3./5., 2./5.],
                [0, 0, 2./5., 3./5.],
                [0, 0, 1./5., 4./5],
                [0, 0, 0, 1.],
                [0, 0, 0, 1./2.],
                [0, 0, 0, 0],
            ]),
        ),
        (
            'Moho & LAB',
            define_models.InversionModel(
                vsv = np.array([[3.5, 3.6, 4., 4.2, 4.4, 4.3, 4.35, 4.4]]).T,
                thickness = np.array([[0., 30., 10., 22., 22., 15., 22., 22.]]).T,
                boundary_inds = np.array([1, 4])
            ),
            np.arange(0, 150, 10),
            np.array([
                [1., 0., 0., 0., 0., 0., 0.],
                [2./3., 1./3., 0., 0., 0., 0., 0.],
                [1./3., 2./3., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 12./22., 10./22., 0., 0., 0.],
                [0., 0., 2./22., 20./22, 0., 0., 0.],
                [0., 0., 0., 14./22., 8./22., 0., 0.],
                [0., 0., 0., 4./22., 18./22., 0., 0.],
                [0., 0., 0., 0., 9./15., 6./15., 0.],
                [0., 0., 0., 0., 0., 21./22., 1./22.],
                [0., 0., 0., 0., 0., 11./22., 11./22.],
                [0., 0., 0., 0., 0., 1./22., 21./22.],
                [0., 0., 0., 0., 0., 0., 13./22.],
                [0., 0., 0., 0., 0., 0., 3./22.],
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
                boundary_inds = []
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
                boundary_inds = []
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
                boundary_inds = np.array([1, 4])
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
                boundary_inds = np.array([1, 4])
            ),
            np.arange(0, 150, 10),
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0], #   0
                [2/3, 1/3, 0, 0, 0, 0, 0, (3.5 - 3.6) * 10 / 30 ** 2, 0], #  10
                [1/3, 2/3, 0, 0, 0, 0, 0, (3.5 - 3.6) * 20 / 30 ** 2, 0], #  20
                [0, 1, 0, 0, 0, 0, 0, (3.6 - 4.) / 10, 0], #  30
                [0, 0, 1, 0, 0, 0, 0, (4. - 4.2) * 22 / 22 ** 2, 0], #  40
                [0, 0, 12/22, 10/22, 0, 0, 0, (4. - 4.2) * 12 / 22 ** 2, 0], #  50
                [0, 0, 2/22, 20/22, 0, 0, 0, (4. - 4.2) * 2 / 22 ** 2, 0], #  60
                [0, 0, 0, 13/21, 8/21, 0, 0, 0, (4.2 - 4.4) * 8 / 21 ** 2], #  70
                [0, 0, 0, 3/21, 18/21, 0, 0, 0, (4.2 - 4.4) * 18 / 21 ** 2], #  80
                [0, 0, 0, 0, 8/15, 7/15, 0, 0, (4.4 - 4.3) / 15], #  90
                [0, 0, 0, 0, 0, 21/23, 2/23, 0, (4.3 - 4.35) * 21 / 23 ** 2 ], # 100
                [0, 0, 0, 0, 0, 11/23, 12/23, 0, (4.3 - 4.35) * 11 / 23 ** 2], # 110
                [0, 0, 0, 0, 0, 1/23, 22/23, 0, (4.3 - 4.35) * 1 / 23 ** 2], # 120
                [0, 0, 0, 0, 0, 0, 15/24, 0, 0], # 130 - pegged
                [0, 0, 0, 0, 0, 0, 5/24, 0, 0], # 140 - pegged
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
            define_models.SetupModel(
                id = 'moho', boundary_depths = np.array([30.]),
                location = (35., -110.),
                boundary_widths = np.array([5.]),
                boundary_depth_uncertainty = np.array([3.]),
                boundary_vsv = np.array([3.5, 4.]),
                depth_limits = np.array([0., 200.]),
                boundary_names = ('Moho',), min_layer_thickness = 10.,
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
                [1, 2, 6, 4, 10, 18, 7, 2, 6],
                [0, 1, 4, 0, 5, 12, 0, 1, 4],
                [1, 8, 6, 4, 10, 18, 7, 2, 6],
                [0, 1, 4, 0, 5, 12, 0, 1, 4],
                [1, 2, 4, 4, 10, 18, 7, 2, 6],
                [1, 2, 6, 4, 10, 18, 7, 2, 6],
                [0, 1, 4, 0, 5, 12, 0, 1, 4],
                [1, 2, 6, 4, 10, 18, 7, 2, 6],
                [0, 8, 4, 0, 5, 12, 0, 1, 4]
            ]),
            np.array([10, 20, 20, 80, 50, 180, 70, 40, 20])[:, np.newaxis],
        ),
        (
            'fewer_constraints',
            np.array([
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
            ]),
            np.array([10, 20])[:, np.newaxis],
            pd.DataFrame({'fewer_constraints': [1] * 2 + [2] + [4, 5, 6, 7]}),
            pd.DataFrame({'fewer_constraints': [1, 2]}),
            np.array([
                [1, 2, 6, 4, 10, 18, 7, 2, 6],
                [0, 1, 4, 0, 5, 12, 0, 1, 4],
            ]),
            np.array([10, 20])[:, np.newaxis], # Note that this is the expected
                                               # behaviour (h unchanged), but
                                               # it will cause issues unless
                                               # h == 0 always OR it has
                                               # already been damped
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
                boundary_inds = np.array([4, 8])
            ),
            np.array([
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  1, -13/7,  6/7,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  6/4, -10/4,  1,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  1, -16/10,  6/10,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  6/12, -18/12,  1,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            ]),
            np.array([0] * 12)[:, np.newaxis],
        ),
    ])
    @unittest.skipIf(skip_influx, "Changed this bit but not the test yet")
    def test_build_smoothing_constraints(self, name, model,
                                         expected_H, expected_h):

        H, h, label = weights._build_smoothing_constraints(model)

        self.assertEqual(label, 'roughness')
        np.testing.assert_array_equal(H, expected_H)
        np.testing.assert_array_equal(h, expected_h)

        n_layers = expected_h.shape[0]
        n_bls = model.boundary_inds.shape[0]
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
        np.testing.assert_array_equal(H_sc, expected_H)
        np.testing.assert_array_equal(h_sc, expected_h)

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

    # test_build_constraint_damp_to_m0
    # @parameterized.expand([
    #     (
    #         'simple',
    #         np.array([1, 2, 3, 4, 5])[:, np.newaxis],
    #         np.array([
    #             [1, 0, 0, 0, 0],
    #             [0, 1, 0, 0, 0],
    #             [0, 0, 1, 0, 0],
    #             [0, 0, 0, 1, 0],
    #             [0, 0, 0, 0, 1],
    #         ]),
    #         np.array([1, 2, 3, 4, 5])[:, np.newaxis],
    #     ),
    # ])
    # def test_build_constraint_damp_to_m0_grad(self, name, model,
    #                                           expected_H, expected_h):
    #     H, h, label = weights._build_constraint_damp_to_m0_grad(model)
    #     self.assertEqual(label, 'to_m0_grad')
    #     np.testing.assert_array_equal(H, expected_H)
    #     np.testing.assert_array_equal(h, expected_h)



if __name__ == "__main__":
    unittest.main()
