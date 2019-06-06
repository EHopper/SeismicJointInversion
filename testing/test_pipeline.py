""" Test all of the functions!


"""
import sys
import os
import pathlib
import random
import unittest
from parameterized import parameterized
import numpy as np

base_dir = pathlib.Path.home() / 'Documents/SurfaceWaves/INVERSION/'
sys.path.insert(0, str(base_dir / 'util'))
import matlab
import define_earth_model
import surface_waves


class PipelineTest(unittest.TestCase):
    """ Set up the test pipeling.

    Start by defining extra testing assertions for locally
    defined classes.

    For each test, set up base inputs, some range of them to
    test (within @parameterized.expand()), then define the
    test for each function.
    """

    # ==========================================================
    #  Local class specific assertions
    # ==========================================================
    def assert_PhaseVelocity_equal(self, actual, expected):
        np.testing.assert_array_almost_equal(
            actual.c, expected.c,
            decimal = 2
            )
        np.testing.assert_array_equal(
            actual.period, expected.period
            )

    def assert_EarthModel_equal(self, actual, expected):
        # np.testing.assert_array_equal() accepts floats etc too
        np.testing.assert_array_almost_equal(
            actual.vs, expected.vs,
            decimal = 2)
        np.testing.assert_array_almost_equal(
            actual.vs, expected.vs,
            decimal = 2)
        np.testing.assert_array_almost_equal(
            actual.rho, expected.rho,
            decimal = 2)
        np.testing.assert_array_almost_equal(
            actual.depth, expected.depth,
            decimal = 2)
        np.testing.assert_array_almost_equal(
            actual.thickness, expected.thickness,
            decimal = 2)

    # ===========================================================
    #  Test output of functions
    # ===========================================================

    # ----------- Velocity Model (define_earth_model) ----------
    # Test loading model from saved Vs model (.nc file from IRIS)


    # Test simplification of velocity model to layers
    # @parameterized.expand([
    #     ("Model at 35N, 115W",
    #         np.array([3.297, 3.513, 3.664, 3.939, 4.411, 4.232])),
    #         np.array([5., 19.5, 25., 30.5, 40.])
    # ])
    #

    # Test full model generation (Vp, rho scaling)
    vs = np.array([4., 4.5, 4.7, 5.])
    depth = np.array([30., 70., 130.])
    max_crustal_vs = 4.
    @parameterized.expand([
            ("Input model",
                vs,
                depth,
                max_crustal_vs,
                define_earth_model.EarthModel(
                    vs = np.array([4., 4.5, 4.7, 5.]),
                    vp = np.array([6.9357, 7.875, 8.225, 8.75]),
                    thickness = np.array([30., 40., 60., 100.]),
                    depth = np.array([30., 70., 130.]),
                    rho = np.array([2.9496, 3.4268, 3.4390, 3.4368]),
                    )
            ),
            ("Vary vels",
                np.array([3., 4.2, 4.6, 5.2]),
                depth,
                4.2,
                define_earth_model.EarthModel(
                    vs = np.array([3., 4.2, 4.6, 5.2]),
                    vp = np.array([5.0506, 7.3307, 8.05, 9.1]),
                    thickness = np.array([30., 40., 60., 100.]),
                    depth = np.array([30., 70., 130.]),
                    rho = np.array([2.5426, 3.0674, 3.4329, 3.4406]),
                    )
            ),
            ("Vary deps",
                vs,
                np.array([6., 27.5, 115.]),
                max_crustal_vs,
                define_earth_model.EarthModel(
                    vs = np.array([4., 4.5, 4.7, 5.]),
                    vp = np.array([6.9357, 7.875, 8.225, 8.75]),
                    thickness = np.array([6., 21.5, 87.5, 100.]),
                    depth = np.array([6., 27.5, 115.]),
                    rho = np.array([2.9496, 3.4268, 3.4379, 3.4368]),
                    )
            ),
            ("Vary depth and crustal cutoff",
                np.array([4., 4.5, 4.7, 5.]),
                np.array([6., 60., 147.5]),
                4.5,
                define_earth_model.EarthModel(
                    vs = np.array([4., 4.5, 4.7, 5.]),
                    vp = np.array([6.9357, 7.9062, 8.225, 8.75]),
                    thickness = np.array([6., 54., 87.5, 100.]),
                    depth = np.array([6., 60., 147.5]),
                    rho = np.array([2.9496, 3.2579, 3.4391, 3.4386]),
                    )
            ),
            ("Crustal vels only",
                np.array([1.7, 2.6, 3.6, 4.4]),
                depth,
                4.5,
                define_earth_model.EarthModel(
                     vs = np.array([1.7, 2.6, 3.6, 4.4]),
                     vp = np.array([3.238876, 4.408495,6.1488126, 7.7179102]),
                     thickness = np.array([30., 40., 60., 100.]),
                     depth = np.array([30., 70., 130.]),
                     rho = np.array([2.2723679, 2.4495676, 2.7493737, 3.193219]),
                     )
            ),
            ("Mantle vels only",
                np.array([4.5, 4.6, 4.7, 5.]),
                depth,
                max_crustal_vs,
                define_earth_model.EarthModel(
                     vs = np.array([4.5, 4.6, 4.7, 5.]),
                     vp = np.array([7.875, 8.05, 8.225, 8.75]),
                     thickness = np.array([30., 40., 60., 100.]),
                     depth = np.array([30., 70., 130.]),
                     rho = np.array([3.4268, 3.431978, 3.438984, 3.43739]),
                     )
             ),
             ("Last layer only",
                np.array([4.7]),
                np.array([]),
                max_crustal_vs,
                define_earth_model.EarthModel(
                    vs = np.array([4.7]),
                    vp = np.array([8.225]),
                    thickness = np.array([100]),
                    depth = np.array([]),
                    rho = np.array([3.4371]),
                    )
             ),
    ])
    def test_MakeFullVelModel(self, name, vs, depth, max_crustal_vs, expected):
        vp, rho, thickness = define_earth_model._calculate_vp_rho(
            vs, depth, max_crustal_vs)
        full_model = define_earth_model.EarthModel(
            vs = vs, vp = vp, thickness = thickness, depth = depth, rho = rho,
        )
        self.assert_EarthModel_equal(full_model, expected)

    del vs, depth, max_crustal_vs


    # ---------- Surface wave dispersion (surface_waves) ----------

    # Set up some starting values for surface_waves testing
    swd_obs = surface_waves.ObsPhaseVelocity(
                period = 1/np.arange(0.02, 0.11, 0.01),
                c = np.array([3.9583, 3.8536, 3.6781,
                             3.4724, 3.3217, 3.2340,
                             3.1850, 3.1572, 3.1410]),
                std = np.ones(9,))
    depth = np.array([30.])
    vs = np.array([3.4, 4.5])

    # Test inputs for surface_waves._Rayleigh_phase_velocity_in_half_space:
    @parameterized.expand([("Moho only", vs[[0]], 3.1099)])
    def test_homogeneous(self, name, vs, expected):
        """ Test Rayleigh phase velocity calculation in a half space """
        vp, rho, thickness = define_earth_model._calculate_vp_rho(
            vs, np.array([])) # Use only the first values as half space
        np.testing.assert_array_almost_equal(
            surface_waves._Rayleigh_phase_velocity_in_half_space(vp, vs),
            expected,
            decimal = 4)

    #  Test inputs for surface_waves._secular:
    @parameterized.expand([
            ("Moho only",
                vs, depth, swd_obs.period,
                0.9, 0, 1.2031,
            ),
            ("Moho only",
                vs, depth, swd_obs.period,
                1.5, 3, 3.6431,
            ),
            ("Three layers",
                np.array([3.2, 4.0, 4.6]),
                np.array([7., 30.]),
                swd_obs.period,
                1.5, 3, 0.5127,
            ),
            ("Many layers",
                np.array([3.2, 3.5, 3.8, 4.2, 4.5]),
                np.array([1.8, 5.7, 14.2, 35.]),
                swd_obs.period,
                1.9, 8, 10.5059,
            ),
            ("Many slow layers",
                np.array([0.7, 1.8, 2.2, 3.5]),
                np.array([1.2, 11., 42.5]),
                np.array([47]),
                1.32, 0, 94.027,
            ),
            ("Exact match between Vs and c",
                np.array([3.2, 3.5, 3.8, 4.2, 4.5]),
                np.array([1.8, 5.7, 14.2, 35.]),
                swd_obs.period,
                2, 7, 6.331,
            ),
    ])
    def test_secular(self, name, vs, depth, periods, c_sc, swd_i, expected):
        """ Test the calculation of the secular function.

        The period of the surface wave dispersion is given by swd_i, so
        swd_i should be between 0 and len(swd_obs.period) - 1.

        The wavenumber for the secular function is given by k_sc, which
        is a multiplier for the input model Vs.  This gives the guess
        at the phase velocity by c = omega/wavenumber, which is then run
        through the secular function to see how small a value is output.
        k_sc should vary between 0.9 and 2; when k_sc < 1, it is just used
        as a straight multiplier of the minimum model velocity; when k_sc > 1,
        it is used to phase velocity between the minimum and maximum velocity.

        """

        # Calculate the input velocity model
        vp, rho, thickness = define_earth_model._calculate_vp_rho(
            vs, depth, 4.4)

        # Set guessed phase velocity so it is similar to the input Vs model
        # by scaling according to c_sc
        if c_sc <= 1:
            c = c_sc * np.min(vs)
        else:
            c = ((c_sc-1) * np.ptp(vs) + np.min(vs))

        omega = 2 * np.pi / periods[swd_i]
        mu = rho * vs**2
        self.assertAlmostEqual(
            surface_waves._secular(omega/c, omega, thickness, mu, rho, vp, vs),
            expected, places = 3)


    @parameterized.expand([
            ("Moho only",
                vs, depth, swd_obs,
            ),
            ("Halfspace only",
                np.array([3.]),
                np.array([10]),
                swd_obs._replace(c = np.ones((9))*2.7406),
            ),
            ("Complicated model",
            # 2, 17, 42, 58, 91
                np.array([4., 1.2, 4.3, 2.7, 3.1]),
                np.array([1.9, 5.9, 13.2, 34.5]),
                swd_obs._replace(c = np.array([2.7731, 2.6669, 2.5958,
                                               2.5647, 2.5514, 2.5301,
                                               2.4498, 2.1482, 1.8616])),
            ),
    ])
    def test_synthesise_surface_wave(self, name, vs, depth, swd_obs):
        pass
        vp, rho, thickness = define_earth_model._calculate_vp_rho(
            vs, depth, 4.4)
        full_model = define_earth_model.EarthModel(
            vs = vs, vp = vp, thickness = thickness, depth = depth, rho = rho,
        )
        self.assert_PhaseVelocity_equal(
            surface_waves.synthesise_surface_wave(full_model, swd_obs.period),
            swd_obs,
        )






if __name__ == "__main__":
    unittest.main()
