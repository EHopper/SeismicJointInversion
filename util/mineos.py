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
    l_max: int = 7000
    l_increment_standard: int = 2
    l_increment_failed: int = 2
    max_run_N: int = 500
    qmod_path: str = './data/earth_models/qmod'
    bin_path: str = '../MINEOS/bin'

class LoveKernels(typing.NamedTuple):
    """ Frechet kernels for Love (toroidal) waves.

    Kernels for different periods are appended to each other.

    Fields:
        period:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    seconds
            - Period of Love wave of interest.
        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - Depth vector for kernel.
        vsv:
            - (n_depth_points * n_love_periods, ) np.array
            - Units:    assumes velocities in km/s
            - Vertically polarised S wave kernel.
        vsh:
            - (n_depth_points * n_love_periods, ) np.array
            - Units:    assumes velocities in km/s
            - Horizontally polarised S wave kernel.
        type: str = 'love'
            - Default shouldn't be changed
            - This is because was having issues with isinstance() for locally
              defined classes.


    """

    period: np.array
    depth: np.array
    vsv: np.array
    vsh: np.array
    type: str = 'love'

class RayleighKernels(typing.NamedTuple):
    """ Frechet kernels for Rayleigh (spheroidal) waves.

    Kernels for different periods are stacked on top of each other.

    Fields:
        period:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    seconds
            - Period of Rayleigh wave of interest.
        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - Depth vector for kernel.
        vsv:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Vertically polarised S wave kernel.
        vpv:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Vertically polarised P wave kernel.
        vph:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Horizontally polarised P wave kernel.
        eta:
            - (n_love_periods * n_depth_points, ) np.array
            - Units:    assumes velocities in km/s
            - Anellipticity (eta = F/(A-2L)) kernel.
        type: str = 'rayleigh'
            - Default shouldn't be changed.
            - This is because was having issues with isinstance() for locally
              defined classes.

    """

    period: np.array
    depth: np.array
    vsv: np.array
    vpv: np.array
    vph: np.array
    eta: np.array
    type: str = 'rayleigh'

# =============================================================================
#       Run MINEOS - calculate phase velocity, group velocity, kernels
# =============================================================================

def run_mineos_and_kernels(parameters:RunParameters, periods:np.array,
                           card_name:str):

    ph_vel, n_runs = run_mineos(parameters, periods, card_name)
    kernels = run_kernels(parameters, periods, card_name, n_runs)

    return ph_vel, kernels

def run_kernels(parameters:RunParameters, periods:np.array,
                card_name:str, n_runs:int):

    save_name = 'output/{0}/{0}'.format(card_name)
    execfile = _write_kernel_files(parameters, periods, save_name, n_runs)
    _run_execfile(execfile)

    kernels = _read_kernels(save_name, periods)
    kernels['type'] = parameters.Rayleigh_or_Love

    return kernels
    

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
        execfile = _write_run_mineos(parameters, save_name, l_run, l_min)
        _run_execfile(execfile)

        # Find parameters for re-running
        # Note l_run will only increase if the above run was at least
        # partially successful - c.f. n_runs which increments every time (below)
        min_calculated_period, l_min, l_run = _check_mineos_run(
            save_name, l_run, l_min, parameters
        )

        n_runs += 1
        if n_runs > parameters.max_run_N:
            print('Too many tries! Breaking MINEOS eig loop')
            break

    # Recover eig files from mutliple runs
    for run in range(l_run):
        # l_run is the number of files that need fixing (number of (partially)
        # successful MINEOS runs).  These are named xxx_0, ..., xxx_[l_run - 1].
        execfile = _write_eig_recover(parameters, save_name, run)
        _run_execfile(execfile)

    # Apply Q correction to velocities
    execfile, qfile = _write_q_correction(parameters, save_name, l_run)
    _run_execfile(execfile)

    phase_vel = _read_qfile(qfile, periods)

    return phase_vel, n_runs



def _write_run_mineos(parameters:RunParameters, save_name:str,
                      l_run:int, l_min:int):
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

    try:
        os.remove(execfile)
    except:
        pass

    fid = open(execfile, 'w')
    fid.write('{}/mineos_nohang << ! > {}\n'.format(
        parameters.bin_path, logfile))
    fid.write('{0}.card\n{0}_{1}.asc\n{0}_{1}.eig\n{0}_{1}.mode\n!'.format(
                save_name, l_run))
    fid.write
    fid.close()

    return execfile


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



def _run_execfile(execfile:str):
    """
    """
    subprocess.run(['chmod', 'u+x', './{}'.format(execfile)])
    subprocess.run(['timeout', '120', './{}'.format(execfile)])


def _check_mineos_run(save_name:str, l_run:int, l_min:int,
    parameters:RunParameters):
    """ Load in MINEOS output file and work out parameters to rerun if needed.

    Read in ascfile to give the maximum achieved angular order, l (and
    equivalently, the minimum achieved period, T) in the calculation before
    it broke or timed out.

    Return the minimum calculated period (to check if the calculations go to
    high enough frequency) and, if the calculation does need to restart to go
    to higher frequency, an updated starting angular order, l_min.
    """

    ascfile = '{0}_{1}.asc'.format(save_name, l_run)
    modes = _read_ascfiles([ascfile])

    if modes.empty:
        l_last = l_min + parameters.l_increment_failed
        os.remove(eigfile)
        os.remove(ascfile)
    else:
        l_run += 1
        last_fundamental_l = max(modes[(modes['n'] == 0)]['l'])
        lowest_fundamental_period = min(modes[(modes['n'] == 0)]['T_sec'])


    new_l_min = last_fundamental_l + parameters.l_increment_standard

    return lowest_fundamental_period, new_l_min, l_run


def _read_ascfiles(ascfiles:list):
    """
    n = [] # mode - number of nodes in radius
    l = [] # angular order - number of nodes in latitude
    w_rad_per_s = [] # anglar frequency in rad/s
    w_mHz = [] # frequency in mHz
    T_sec = [] # period (s)
    grV_km_per_s = [] # group velocity
    Q = [] # quality factor
    """

    output = pd.DataFrame()

    for ascfile in ascfiles:
        # Find number of lines to skip - interesting output starts after
        # line labelled 'MODE'
        n_lines = 0
        with open(ascfile, 'r') as fid:
            for line in fid:
                n_lines += 1
                if 'MODE' in line:
                    break

        try:
            output = output.append(pd.read_csv(ascfile, sep='\s+',
                                   skiprows=n_lines, header=None))
        except:
            print(ascfile, ' is empty.')

    output.columns = ['n', 'mode', 'l', 'w_rad_per_s', 'w_mHz', 'T_sec',
                          'grV_km_per_s', 'Q', 'RaylQuo']
    output.drop(['mode', 'RaylQuo'], axis=1, inplace=True)


    return output


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
        print(run)
        fid.write('{}_{}.eig_fix\n'.format(save_name, run))
        if run == 0:
            fid.write('y\n')
    fid.write('\n!\n')#echo "Done velocity calculation, cleaning up..."\n')
    #fid.write('rm {}\n'.format(logfile))
    fid.close()

    return execfile, qfile


def _read_qfile(qfile, periods):
    """
    Note we are returning the Q corrected phase velocity only
    """

    with open(qfile, 'r') as fid:
        n_lines = int(fid.readline())

    qf = pd.read_csv(qfile, header=None, skiprows=n_lines+1, sep='\s+')
    qf.columns = ['n', 'l', 'w_mHz', 'Q', 'phi', 'ph_vel',
                     'gr_vel', 'ph_vel_qcorrected', 'T_qcorrected', 'T_sec']
    qf = qf[::-1]
    qf = qf[qf['n'] == 0] # Fundamental mode only

    ph_vel = np.interp(periods, qf.T_qcorrected, qf.ph_vel_qcorrected)

    return ph_vel

def _write_kernel_files(parameters:RunParameters, periods:np.array,
                        save_name:str, n_runs:int):
    """
    Note this is hardwired to only calculate phase velocity - possible to
    do for group velocity as well.

    Also note that I changed the plot_wk execfile part from Zach's code -
    he had it running from angular order 0 to 3000.  I increased the upper
    limit and also started it at 1 (because my output never has n=0, l=0 -
    and the fact this mode was missing was breaking the table function).
    """

    execfile = '{0}.run_kernels'.format(save_name)

    max_angular_order = {
        'Rayleigh': 5500,
        'Love': 3500,
    }

    eigfiles = (['{}_{}.eig_fix'.format(save_name, run)
                for run in range(1, n_runs)])


    with open(execfile, 'w') as fid:
        fid.write("""#!/bin/bash
#
echo "======================" > {0}.log
echo "Stripping MINEOS" >> {0}.log
#
{1}/mineos_strip <<! >> {0}.log
{0}.strip
{2}
{3}

!
#
echo "======================" > {0}.log
echo "Done stripping, now calculating tables" > {0}.log
#
{1}/mineos_table <<! >> {0}.log
{0}.table
40000
0 {4:.1f}
1 {5:.0f}
{0}.q
{0}.strip

!
#
echo "======================" > {0}.log
echo "Creating branch file" > {0}.log
#
{1}/plot_wk <<! >> {0}.log
table {0}.table_hdr
search
1 0.0 {4:.1f}
99 0 0
branch

quit
!
#
echo "======================" > {0}.log
echo "Making frechet phV kernels binary" > {0}.log
#
rm {0}.cvfrechet
{1}/frechet_cv <<! >> {0}.log
{6}
{0}.table_hdr.branch
{0}.cvfrechet
{2}
0
{3}

!
#
echo "======================" > {0}.log
echo "Writing phV kernel files for each period" > {0}.log
#
                 """.format(
                 save_name,
                 parameters.bin_path,
                 '{}_0.eig_fix'.format(save_name),
                 '\n'.join(eigfiles),
                 1000 / min(periods) + 0.1, # max freq. in mHz
                 max_angular_order[parameters.Rayleigh_or_Love],
                 parameters.qmod_path,
                 ))

    # Need to loop through periods in executable
    for period in periods:
        with open(execfile, 'a') as fid:
            fid.write("""{1}/draw_frechet_gv <<!
{0}.cvfrechet
{0}_cvfrechet_{2:.1f}s
{2:.2f}
!
            """.format(
            save_name,
            parameters.bin_path,
            period,
            ))


    return execfile

def _read_kernels(save_name, periods):
    """
    """
    kernels = pd.DataFrame([])

    for period in periods:
        kernelfile = '{0}_cvfrechet_{1:.1f}s'.format(save_name, period)
        kf = pd.read_csv(kernelfile, sep='\s+', header=None)
        kf.columns = ['r', 'vsv', 'vpv', 'vsh', 'vph', 'eta', 'rho']
        kf['z'] = 6371 - kf['r'] * 1e-3
        kf['period'] = period
        kf = kf[::-1]
        kernels = kernels.append(kf)

    kernels = kernels[['z', 'period', 'vsv', 'vpv', 'vsh', 'vph', 'eta', 'rho']]

    return kernels
