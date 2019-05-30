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
sys.path.insert(0, str(bin_dir / 'functions'))
import surface_waves
import matlab


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
    def assertSurfaceWaveDispEqual(self, actual, expected):
        np.testing.assert_array_almost_equal(
            actual.c, expected.c,
            decimal = 2
            )
        np.testing.assert_array_equal(
            actual.period, expected.period
            )
        np.testing.assert_array_almost_equal(
            actual.std, expected.std,
            decimal = 2
            )
    def assertModelEqual(self, actual, expected):
        # np.testing.assert_array_equal() accepts floats etc too
        np.testing.assert_array_almost_equal(actual.vs, expected.vs)
        np.testing.assert_array_almost_equal(actual.vs, expected.vs)
        np.testing.assert_array_almost_equal(actual.rho, expected.rho)
        np.testing.assert_array_almost_equal(actual.depths, expected.depths)
        np.testing.assert_array_almost_equal(
            actual.thickness, expected.thickness)

    # ===========================================================
    #  Test output of functions
    # ===========================================================

    # Test surface wave dispersion calculations
    swd_obs = surface_waves.SurfaceWaveDispersion(
                period = 1/np.arange(0.02, 0.11, 0.01),
                c = np.array([3.9583, 3.8536, 3.6781,
                             3.4724, 3.3217, 3.2340,
                             3.1850, 3.1572, 3.1410]),
                std = np.ones(9,))
    deps = np.concatenate((np.arange(0,10,0.2),
                           np.arange(10,60,1),
                           np.arange(60,201,5))
                         )
    model = pipeline.Model(vs = np.array([3.4, 4.5]),
                           all_deps = deps,
                           idep = np.array([60, 80]),
                           std_rf_sc = 0, lam_rf = 0, # irrelevant
                           std_swd_sc = 0) # irrelevant

    @parameterized.expand([("Moho only", model, 3.1099)])
    def test_homogeneous(self, name, model, expected):
        model = pipeline.MakeFullModel(model)
        self.assertAlmostEqual(
                pipeline._CalcRaylPhaseVelInHalfSpace(
                    model.vp[0], model.vs[0]),
                    expected, places = 4)


    @parameterized.expand([
            ("Moho only", model, swd_obs, 0.9, 0, 1.2031),
            ("Moho only", model, swd_obs, 1.5, 3, 3.6431),
            ("Three layers", model._replace(vs = np.array([3.2, 4.0, 4.6]),
                                            idep = np.array([20, 50, 90])),
                                swd_obs, 1.5, 3, 0.5127),
            ("Many layers", model._replace(vs = np.array([3.2, 3.5, 3.8, 4.2, 4.5]),
                                           idep = np.array([3, 15, 42, 60, 90])),
                                swd_obs, 1.9, 8, 10.5059),
            ("Many slow layers", model._replace(vs =np.array([0.7, 1.8, 2.2, 3.5]),
                                                idep = np.array([2, 10, 60, 101])),
                                swd_obs._replace(period = np.array([47])),
                                1.32, 0, 94.0398),
            ("Exact match", model._replace(vs = np.array([3.2, 3.5, 3.8, 4.2, 4.5]),
                                           idep = np.array([3, 15, 42, 60, 90])),
                                swd_obs, 2, 7, 6.33),
        ])
    def test_secular(self, name, model, swd_obs, k_sc, swd_i, expected):
        # k_sc should be between 0.9-2; swd_i between 0-len(swd_obs.period)-1
        model = pipeline.MakeFullModel(model)
        if k_sc <= 1:
            v = k_sc * np.min(model.vs)
        else:
            v = ((k_sc-1)*np.diff([np.min(model.vs), np.max(model.vs)]) +
                    np.min(model.vs))
        om = 2*np.pi/swd_obs.period[swd_i]
        mu = model.rho * model.vs**2
        self.assertAlmostEqual(pipeline._Secular(om/v, om, model.thickness,
                                                 mu, model.rho, model.vp,
                                                 model.vs),
            expected, places = 4)


    @parameterized.expand([
            ("Moho only", model, swd_obs),
            ("Halfspace only", model._replace(vs = np.array([3.]),
                                              idep = np.array([40])),
                                swd_obs._replace(c = np.ones((9))*2.7406)),
            ("Complicated model", model._replace(vs = np.array([4., 1.2, 4.3, 2.7, 3.1]),
                                                 idep = np.array([2, 17, 42, 58, 91])),
                                swd_obs._replace(c = np.array([2.7731, 2.6669, 2.5958,
                                                               2.5647, 2.5514, 2.5301,
                                                               2.4498, 2.1482, 1.8616]))),
            ])

    def test_SurfaceWaveDisp(self, name, model, swd_obs):
        model = pipeline.MakeFullModel(model)
        self.assertSurfaceWaveDispEqual(pipeline.SynthesiseSWD(model, swd_obs, 1e6),
                                        swd_obs)






if __name__ == "__main__":
    unittest.main()
