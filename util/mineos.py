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
import subprocess
import os
import shutil
import pandas as pd

#import surface_waves


# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class RunParameters(typing.NamedTuple):
    """ Parameters needed to run MINEOS.

    Fields:
        Rayleigh_or_Love:
            - str
            - 'Rayleigh' or 'Love' for Rayleigh or Love
            - Default value = 'Rayleigh'
        phase_or_group_velocity:
            - str
            - 'ph' or 'gr' for phase or group velocity
            - Default value =  'ph'
        l_min:
            - int
            - Minimum angular order for calculations
            - Default value = 0
        l_max:
            - int
            - Expected max angular order for calculations
            - Default value = 3500.
        freq_min:
            - float
            - Units:    mHz
            - Minimum frequency for calculations.
            - Default value = 0.05 mHz (i.e. 20,000 s)
        freq_max:
            - float
            - Units:    mHz
            - Maximum frequency - should be set to 1000/min(sw_periods) + 1
                need to compute a little bit beyond the ideal minimum period
        l_increment_standard:
            - int
            - When MINEOS breaks and has to be restarted with a higher lmin,
              it is normally restarted at l_min = the last successfully
              calculated l (l_last) + l_increment_standard.
            - Default value = 2
        l_increment_failed:
            - int
            - When MINEOS breaks and has to be restarted with a higher lmin,
              if the last attempt produced no successful calculations, l_min
              is instead l_last + l_increment_failed.
            - Default value = 5 how much to incrememnt lmin by if broken*
        max_run_N:
            - int
            - When MINEOS breaks and has to be restarted with a higher lmin,
              if it tries to restart more than max_run_N times, it will
              return an error instead.
            - Default value = 5e2
        qmod_path:
            - str
            - Path to the standard qmod file for attenuation corrections.
            - Default value = './data/earth_models/qmod'
        bin_path:
            - str
            - Path to the FORTRAN executables for MINEOS
            - Default value = '../MINEOS/bin'

    """

    freq_max: float
    freq_min: float = 0.05
    Rayleigh_or_Love: str = 'Rayleigh'
    phase_or_group_velocity: str = 'ph'
    l_min: int = 0
    l_max: int = 3500
    l_increment_standard: int = 2
    l_increment_failed: int = 2
    max_run_N: int = 500
    qmod_path: str = './data/earth_models/qmod'
    bin_path: str = '../MINEOS/bin'



# =============================================================================
#       Run MINEOS - calculate phase velocity, group velocity, kernels
# =============================================================================

def run_mineos(parameters:RunParameters, periods:np.array,
               card_name:str) -> np.array:
    """
    Given a card_model_name (MINEOS card saved as a text file), run MINEOS.
    """

    save_name = 'output/{0}/{0}'.format(card_name)

    # Run MINEOS - and re-run repeatedly if/when it breaks
    min_desired_period = np.min(periods)
    min_calculated_period = min_desired_period + 1 # arbitrarily larger
    n_runs = 0
    l_run = 0
    l_min = parameters.l_min

    while min_calculated_period > min_desired_period:
        print('Run {:3.0f}, min. l {:3.0f}'.format(n_runs, l_min))
        new_min_T, l_run, l_min = (
            _run_mineos(parameters, save_name, l_run, l_min)
        )
        if new_min_T:
            min_calculated_period = new_min_T

        n_runs += 1
        if n_runs > parameters.max_run_N:
            print('Too many tries! Breaking MINEOS eig loop')
            break

    for run in range(l_run):
        execfile = _write_eig_recover(parameters, save_name, run)
        _run_execfile(execfile)


    execfile, qfile = _write_q_correction(parameters, save_name, l_run)
    _run_execfile(execfile)

    #phase_vel, group_vel = _read_qfile(qfile, periods)

    return #phase_vel, group_vel



def _run_mineos(parameters:RunParameters, save_name:str, l_run:int, l_min:int):
    """ Write all the files, then run the fortran code.
    """

    # Set filenames
    execfile = '{0}_{1}.run_mineos'.format(save_name, l_run)
    ascfile = '{0}_{1}.asc'.format(save_name, l_run)
    eigfile = '{0}_{1}.eig'.format(save_name, l_run)
    modefile = '{0}_{1}.mode'.format(save_name, l_run)
    logfile = '{0}.log'.format(save_name)
    cardfile = '{0}.card'.format(save_name)

    _write_modefile(modefile, parameters, l_min)
    _write_execfile(execfile, cardfile, modefile, eigfile, ascfile, logfile,
                    parameters)

    _run_execfile(execfile)

    Tmin, l_last = _check_output([ascfile])

    if not l_last:
        l_last = l_min + parameters.l_increment_failed
        os.remove(eigfile)
        os.remove(ascfile)
    else:
        l_run += 1

    return Tmin, l_run, l_last + parameters.l_increment_standard

def _write_modefile(modefile, parameters, l_min):
    """
     mode table looks like
    1.d-12  1.d-12  1.d-12 .126
    [3 (if spherical) or 2 (if toroidal)]
    (minL) (maxL) (minF) (maxF) (N_[ST]modes)
    0
    - top line: EPS EPS1 EPS2 WGRAV   (accuracy of mode calculation)
    where EPS controls the accuracy of the integration scheme, EPS1 controls
    the precision with which a root is found, EPS2 is the minimum separation
    of two roots.  WGRAV is the frequency (rad/s) above which gravitational
    terms are neglected (much faster calculation)
    - next line: 1 (radial modes); 2 (toroidal modes); 3 (speheroidal modes)
           4 (inner core toroidal modes); 0 (quit the program)
    - min/max L are angular order range
    - min/max F are frequency range (in mHz)
    - N_[TS]modes are number of mode branches for Love and Rayleigh
      i.e. 0 if just fundamental mode
    """
    try:
        os.remove(modefile)
    except:
        pass

    m_codes = ['quit', 'radial', 'toroidal', 'spheroidal', 'inner core']
    wave_to_mode = {
        'Rayleigh': 'spheroidal',
        'Love': 'toroidal',
    }
    m_code = m_codes.index(wave_to_mode[parameters.Rayleigh_or_Love])

    fid = open(modefile, 'w')
    # First line: accuracy of mode calculation - eps eps1 eps2 wgrav
    #       where eps - accuracy of integration scheme
    #             eps1 - precision when finding roots
    #             eps2 - minimum separation of roots
    #             wgrav - freqeuncy (rad/s) above which gravitational terms
    #                     are neglected
    # Second line - m_code gives the index in the list above, m_codes
    fid.write('1.d-12  1.d-12  1.d-12 .126\n{}\n'.format(m_code))
    # Third line: lmin and lmax give range of angular order, fmin and fmax
    #       give the range of frequency (mHz), final parameter is the number
    #       of mode branches for Love and Rayleigh - hardwired to be 1 for
    #       fundamental mode (= 2 would include first overtone)
    # Fourth line: not entirely sure what this 0 means
    fid.write('{:.0f} {:.0f} {:.3f} {:.3f} 1\n0\n'.format(l_min,
        parameters.l_max, parameters.freq_min, parameters.freq_max))

    fid.close()

def _write_execfile(execfile:str, cardfile:str, modefile:str, eigfile:str,
                    ascfile:str, logfile:str, params:RunParameters):
    """
    """
    try:
        os.remove(execfile)
    except:
        pass

    fid = open(execfile, 'w')
    fid.write('{}/mineos_nohang << ! > {}\n'.format(params.bin_path, logfile))
    fid.write('{0}\n{1}\n{2}\n{3}\n!\n#\n#rm {4}\n'.format(cardfile, ascfile,
        eigfile, modefile, logfile))
    fid.close()

def _run_execfile(execfile):
    """
    """
    subprocess.run(['chmod', 'u+x', './{}'.format(execfile)])
    subprocess.run(['timeout', '100', './{}'.format(execfile)])

def _check_output(ascfiles:list):

    modes = _read_ascfiles(ascfiles)

    if modes.empty:
        return None, None

    last_fundamental_l = max(modes[(modes['n'] == 0)]['l'])
    lowest_fundamental_period = min(modes[(modes['n'] == 0)]['T_sec'])

    return lowest_fundamental_period, last_fundamental_l


def _read_ascfiles(ascfiles:list):
    """
    """

    n = [] # mode - number of nodes in radius
    l = [] # angular order - number of nodes in latitude
    w_rad_per_s = [] # anglar frequency in rad/s
    w_mHz = [] # frequency in mHz
    T_sec = [] # period (s)
    grV_km_per_s = [] # group velocity
    Q = [] # quality factor

    for ascfile in ascfiles:
        fid = open(ascfile, 'r')
        at_mode = False

        for line in fid:
            # Find beginning of mode description
            if 'MODE' in line:
                fid.readline() # skip blank line
                line = fid.readline()
                at_mode = True

            if at_mode:
                mode_line = line.split()
                try:
                    n += [float(mode_line[0])]
                    l += [float(mode_line[2])]
                    w_rad_per_s += [float(mode_line[3])]
                    w_mHz += [float(mode_line[4])]
                    T_sec += [float(mode_line[5])]
                    grV_km_per_s += [float(mode_line[6])]
                    Q += [float(mode_line[7])]
                except:
                    print('Line in .asc file improperly formatted')
                    pass


        fid.close()

    return pd.DataFrame({
        'n': n,
        'l': l,
        'w_rad_per_s': w_rad_per_s,
        'w_mHz': w_mHz,
        'T_sec': T_sec,
        'grV_km_per_s': grV_km_per_s,
        'Q': Q,
    })


def _write_eig_recover(params, save_name, l_run):
    """
    Note: eig_recover will save a new file [filename].eig_fix
    """

    execfile ='{0}_{1}.eig_recover'.format(save_name, l_run)
    eigfile = '{0}_{1}.eig'.format(save_name, l_run)

    try:
        os.remove(eigrecoverfile)
    except:
        pass

    ascfile = '{0}_{1}.asc'.format(save_name, l_run)
    modes = _read_ascfiles([ascfile])
    l_last = modes['l'].iloc[-1]

    fid = open(execfile, 'w')
    fid.write('#!/bin/bash\n#\n')
    fid.write('{}/eig_recover << ! \n'.format(params.bin_path))
    fid.write('{0}\n{1:.0f}\n!\n'.format(eigfile, l_last))
    fid.close()

    return execfile


def _write_q_correction(params, save_name, l_run):
    """
    NOTE: qmod is probably complete bullshit!  Took Zach's qmod and then
    changed the 0 for q_mu in the inner core to 100000, because otherwise
    get a divide by 0 error and a whole bunch on NaN.  Clearly, the Q of
    a liquid should be pretty low (?!?) so this doesn't seem to make sense.
    BUT it does mean it runs ok.

    Josh says he thinks that running the Q correction might be falling out
    of favour with Jim, so can always use the uncorrected phase vel etc.
    """

    execfile = '{}.run_mineosq'.format(save_name)
    qfile = '{}.q'.format(save_name)
    logfile = '{}.log'.format(save_name)

    try:
        os.remove(execfile)
    except:
        pass


    fid = open(execfile, 'w')
    fid.write('#!/bin/bash\n#\necho "Q-correcting velocities"\n')
    fid.write('{}/mineos_qcorrectphv << ! >> {}\n'.format(params.bin_path, logfile))
    fid.write('{0}\n{1}\n'.format(params.qmod_path, qfile))
    for run in range(l_run):
        fid.write('{}_{}.eig_fix\n'.format(save_name, run))
        if run == 0:
            fid.write('y\n')
    fid.write('\n!\n')#echo "Done velocity calculation, cleaning up..."\n')
    #fid.write('rm {}\n'.format(logfile))
    fid.close()

    return execfile, qfile


def _read_qfile(qfile, periods):
    """
    """

    fid = open(qfile, 'r')
    lines = fid.readlines()
    n_q_lines = int(lines[0])

    for line in lines[n_q_lines + 2:]:
        line = line.split()
        n += [float(line[0])]
        l += [float(line[1])]
        w_mHz += [float(line[2]) / (2 * pi) * 1000] # convert rad/s to mHz
        T_sec += [float(line[9]])]
        T_qcorrected += [float(line[8])]
        grV_km_per_s += [float(line[6])]
        Q += [float(line[3])]
        phi += [float(line[4])]
        ph_vel += [float(line[5])]
        gr_vel += [float(line[6])]
        ph_vel_qcorrected += [float(line[7])]

    fid.close()

    pass
