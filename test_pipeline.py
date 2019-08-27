""" Test the inversion pipeline
"""

import unittest
from parameterized import parameterized
import numpy as np

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


    # =========================================================================
    # The tests to run
    # =========================================================================


    # ************************* #
    #      define_models.py     #
    # ************************* #
    @parameterized.expand([
            (
                "Using default values",
                (
                    'moho lab',
                    np.array([30, 100.]),
                    np.array([3, 20.5]),
                    np.array([5, 10]),
                    np.array([3.5, 4, 4.3, 4.25]),
                    np.array([0, 200]),
                    ('Moho', 'LAB'),
                    6,
                    1,
                    1.75,
                    1,
                    'data/earth_models/prem.csv'
                ),
                define_models.SetupModel(
                    id = 'moho lab',
                    boundary_depths = np.array([30, 100]),
                    boundary_depth_uncertainty = np.array([3, 20.5]),
                    boundary_widths = np.array([5, 10]),
                    boundary_vsv = np.array([3.5, 4, 4.3, 4.25]),
                    depth_limits = np.array([0, 200]),
                )
            ),
            (
                "Moho only, no defaults",
                (
                    'moho',
                    np.array([30]),
                    np.array([3]),
                    np.array([5]),
                    np.array([3.5, 4]),
                    np.array([0, 200]),
                    ('Moho',),
                    10,
                    1.1,
                    1.8,
                    0.9,
                    'prem.csv'
                ),
                define_models.SetupModel(
                    id = 'moho',
                    boundary_depths = np.array([30.]),
                    boundary_depth_uncertainty = np.array([3.]),
                    boundary_widths = np.array([5.]),
                    boundary_vsv = np.array([3.5, 4.]),
                    depth_limits = np.array([0., 200.]),
                    boundary_names = ('Moho',),
                    min_layer_thickness = 10.,
                    vsv_vsh_ratio = 1.1,
                    vpv_vsv_ratio = 1.8,
                    vpv_vph_ratio = 0.9,
                    ref_card_csv_name = 'prem.csv'
                )
            )

    ])
    def test_SetupModel(self, name, inputs, expected):
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

    #@unittest.skip("Skip this test because MINEOS is too slow to run every time")
    @parameterized.expand([
        (
            'basic model',
            define_models.SetupModel(
                'test', np.array([35., 90.]), np.array([3, 10]),
                np.array([5, 20]), np.array([3.5, 4.0, 4.2, 4.1]),
                np.array([0, 200])
            ),
            [10, 20, 30, 40, 60, 80, 100],
            [1.05] + [1] * 26,
        )
    ])
    def test_G(self, name, setup_model, periods, model_perturbation):

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
            model_perturbed, setup_model._replace(id='test_perturbed')
        )
        ph_vel_perturbed, _ = mineos.run_mineos(
            params, periods, 'test_perturbed'
        )

        # calculate dv
        dv_mineos = ph_vel_perturbed - ph_vel_pred
        dv_from_Gdm = np.matmul(G, perturbation).flatten()

        np.testing.assert_allclose(dv_mineos, dv_from_Gdm, rtol=0.25)

        im = plt.scatter(dv_mineos, dv_from_Gdm, 10, periods)
        allv = list(dv_mineos) + list(dv_from_Gdm)
        plt.plot([min(allv), max(allv)], [min(allv), max(allv)], 'k:')
        plt.xlabel('dV from MINEOS')
        plt.ylabel('dV from G calculation')
        cbar = plt.gcf().colorbar(im)
        cbar.set_label('Periods (s)', rotation=90)



















if __name__ == "__main__":
    unittest.main()
