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

    # Test SetupModel class
    @parameterized.expand([
            (
                "Using default values",
                (
                    'moho lab', np.array([30, 100.]), np.array([3, 20.5]),
                    np.array([5, 10]), np.array([3.5, 4, 4.3, 4.25]),
                    np.array([0, 200]), ('Moho', 'LAB'), 6, 1, 1.75, 1,
                    'data/earth_models/prem.csv'
                ),
                define_models.SetupModel(
                    id = 'moho lab', boundary_depths = np.array([30, 100]),
                    boundary_depth_uncertainty = np.array([3, 20.5]),
                    boundary_widths = np.array([5, 10]),
                    boundary_vsv = np.array([3.5, 4, 4.3, 4.25]),
                    depth_limits = np.array([0, 200]),
                )
            ),
            (
                "Moho only, no defaults",
                (
                    'moho', np.array([30]), np.array([3]), np.array([5]),
                    np.array([3.5, 4]), np.array([0, 200]), ('Moho',),
                    10, 1.1, 1.8, 0.9, 'prem.csv'
                ),
                define_models.SetupModel(
                    id = 'moho', boundary_depths = np.array([30.]),
                    boundary_depth_uncertainty = np.array([3.]),
                    boundary_widths = np.array([5.]),
                    boundary_vsv = np.array([3.5, 4.]),
                    depth_limits = np.array([0., 200.]),
                    boundary_names = ('Moho',), min_layer_thickness = 10.,
                    vsv_vsh_ratio = 1.1, vpv_vsv_ratio = 1.8,
                    vpv_vph_ratio = 0.9, ref_card_csv_name = 'prem.csv'
                )
            )

    ])
    def test_SetupModel(self, name, inputs, expected):
        """ Test that the SetupModel class defaults work as expected.
        """
        id, bd, bdu, bw, bv, dl, bn, mlt, vsvvsh, vpvs, vpvvph, csv = inputs
        self.assertSetupModelEqual(
            define_models.SetupModel(
                id=id, boundary_depths=bd, boundary_depth_uncertainty=bdu,
                boundary_widths=bw, boundary_vsv=bv, boundary_names=bn,
                min_layer_thickness=mlt, depth_limits=dl,
                vsv_vsh_ratio=vsvvsh, vpv_vph_ratio=vpvvph, vpv_vsv_ratio=vpvs,
                ref_card_csv_name=csv
            ),
            expected
        )

    @parameterized.expand([
        (
            'basic model',
            define_models.SetupModel(
                'testcase', np.array([35., 90.]), np.array([3, 10]),
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
                'testcase', np.array([10, 35., 90., 170]),
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

    @parameterized.expand([
        (
            'NoMelt models and results, courtesy Josh Russell, 2019',
            'NoMeltRayleigh',
            [15.5556, 16.4706, 20, 25, 32, 40, 50, 60, 80, 100, 120, 140, 150],
        )
    ])
    #@unittest.skip("Skip this test because MINEOS is too slow to run every time")
    def test_mineos(self, name, model_id, periods):

        # All previously calculated test outputs should be saved in
        expected_output_dir = './files_for_testing/mineos/'
        # And newly calculated test outputs should be saved in
        actual_output_dir = 'output/testcase/'

        # Copy the test card into the right directory for the workflow
        try:
            shutil.rmtree(actual_output_dir)
        except:
            pass
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
        # pd.testing.assert_frame_equal(kernels_calc, kernels_expected,
        #     check_exact=False, check_less_precise=0,
        # )

        # n = 0
        # p = periods[n]
        # kke = ke[ke.period == p]
        # kk = kj[kj.period == p]
        # plt.plot(kke.vsv, kke.z)
        # plt.plot(kk.vsv*1e3, kk.z)
        # plt.gca().set_ylim([400, 0])
        # n += 1



    # ************************* #
    #       inversion.py        #
    # ************************* #


    @parameterized.expand([
        (
            'basic model',
            define_models.SetupModel(
                'testcase', np.array([35., 90.]), np.array([3, 10]),
                np.array([5, 20]), np.array([3.5, 4.0, 4.2, 4.1]),
                np.array([0, 200])
            ),
            [10, 20, 30, 40, 60, 80, 100],
            [1.05] + [1] * 26,
        )
    ])
    @unittest.skip("Skip this test because MINEOS is too slow to run every time")
    def test_G(self, name, setup_model, periods, model_perturbation):
        """ Test G by comparing the dV from G * dm to the dV output from MINEOS.

        From a given starting model, calculate phase velocities and kernels
        in MINEOS, and convert to the inversion G.

        Convert the starting model to the column vector.  Perturb the column
        vector in some way, and then convert it back to a MINEOS card.  Rerun
        MINEOS to get phase velocities.

        As G is dV/dm, then G * dm == mineos(m_perturbed) - mineos(m0).

        Plot this comparison:
        im = plt.scatter(dv_mineos, dv_from_Gdm, 10, periods)
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
        _ = define_models.convert_inversion_model_to_mineos_model(
            model, setup_model
        )
        params = mineos.RunParameters(freq_max = 1000 / min(periods) + 1)
        ph_vel_pred, kernels = mineos.run_mineos_and_kernels(
            params, periods, setup_model.id
        )
        G = partial_derivatives._build_partial_derivatives_matrix(
            kernels, model, setup_model
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

        np.testing.assert_allclose(dv_mineos, dv_from_Gdm, rtol=0.25)









if __name__ == "__main__":
    unittest.main()
