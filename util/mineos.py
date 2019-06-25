""" Wrapper for using Fortran MINEOS codes.

This WILL BE a Python wrapper to calculate surface wave phase velocities
from a given starting velocity model using MINEOS.  Basically a translation
of Zach Eilon's MATLAB wrapper - https://github.com/eilonzach/matlab_to_mineos.

For now, try and get everything running with some kind of arbitrary kernels
calculated in MATLAB that we aren't going to update, and the surface wave
code that is already working.

Classes:
    RunParameters, with fields

Functions:

"""

#import collections
import typing
import numpy as np

import surface_waves


# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class RunParameters(typing.NamedTuple):
    """ Parameters needed to run MINEOS.

    Fields:
        - Rayleigh_or_Love (str): 'R' or 'L' for Rayleigh or Love
        - phase_or_group_velocity (str): 'ph' or 'gr' for phase or group velocity
        - l_min (int)=0: minimum angular order
        - l_max (int)=3500: expected max angular order
        - freq_min (float)=0.05: min frequency (mHz)
        - freq_max (float)=200.05: max frequency (mHz) - reset by min period
        - l_increment_standard (int)=2: how much to increment lmin by normally*
        - l_increment_failed (int)=5: how much to incrememnt lmin by if broken*
        - max_run_N (int)=5e2: how often to try rerunning MINEOS if it's broken
        - qmod_path (str)='[...]/safekeeping/qmod'

    The l_increment_... is for when MINEOS breaks and has to be restarted with
    a higher lmin.  Normally, it is restarted at l_min = the last successfully
    calculated l (l_last) + l_increment_standard.  If the last attempt didn't
    do any successful calculations, l_min is instead incrememented by
    l_increment_failed from l_last.  If the code has to restart MINEOS more than
    max_run_N times, it will return an error.

    """

    Rayleigh_or_Love: str = 'R'
    phase_or_group_velocity: str = 'ph'
    l_min: int = 0
    l_max: int = 3500
    freq_min: float = 0.05
    freq_max: float = 200.05
    l_increment_standard: int = 2
    l_increment_failed: int = 2
    max_run_N: int = 500
    qmod_path: str = './mineos_data/safekeeping/qmod'



# =============================================================================
#       Run MINEOS - calculate phase velocity, group velocity, kernels
# =============================================================================

def run_mineos(parameters:RunParameters, periods:np.array,
               earth_model_name:str) -> np.array:
    """
    Given an earth_model_name (MINEOS card saved as a text file), run MINEOS.
    """

    _run_mineos_until_not_broken()

    _fix_eigfiles()

    _do_Q_correction()

    pass


def _run_mineos_until_not_broken(min_desired_period):
    min_calculated_period = 1e10
    while min_calculated_period > min_desired_period:
        min_calculated_period = _run_mineos()


    pass


def _run_mineos():

    _write_modefile()
    _write_execfile()
    _run_execfile()
    Tmin = _check_output()

    return Tmin

def _write_modefile():
    pass

def _write_execfile():
    pass

def _run_execfile():
    pass
