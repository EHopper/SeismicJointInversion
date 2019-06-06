""" Generate synthetic surface waves.

This is an all-Python code to calculate surface wave phase velocities
from a given starting velocity model.  Ultimately, this should perhaps
be replaced with MINEOS.

Classes:
    PhaseVelocity, with fields period and phase velocity
    ObsPhaseVelocity, with fields period, phase velocity, and error

Functions:
    synthesise_surface_wave(model, swd_in) -> PhaseVelocity:
        - calculate surface wave phase velocities given model and period
    _Rayleigh_phase_velocity_in_half_space(vp, vs) -> float:
        - given Vp and Vs of a half space, calculate phase velocity
    _min_value_secular_function(omega, k_lims, n_ksteps,
                                thick, rho, vp, vs, mu) -> float:
        - search over k_lims to find the minimum value of _secular()
    _secular(k, om, thick, mu, rho, vp, vs) -> float:
        - calculate Green's function for velocity structure
    _make_3D(x):
        - transpose vector into third dimension (1 x 1 x size)
"""


#import collections
import typing
import numpy as np

import matlab
import define_earth_model

# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class PhaseVelocity(typing.NamedTuple):
    """ Surface wave phase velocity information at a series of periods

    Fields (all fields are n_periods x 1 numpy arrays):
        - period: dominant period of dispersion measurement, seconds
        - c: phase velocity in km/s
        - std: standard deviation of measurement at each period
    """
    period: np.array
    c: np.array

class ObsPhaseVelocity(typing.NamedTuple):
    """ Surface wave phase velocity information at a series of periods

    Fields (all fields are n_periods x 1 numpy arrays):
        - period: dominant period of dispersion measurement, seconds
        - c: phase velocity in km/s
        - std: standard deviation of measurement at each period
    """
    period: np.array
    c: np.array
    std: np.array

# =============================================================================
#       Synthesise Surface Wave Dispersion measurements - Rix & Lai, 2003
# =============================================================================

def synthesise_surface_wave(model:define_earth_model.EarthModel,
                            periods:np.array) -> PhaseVelocity:
    """ Calculate phase velocity given a velocity model and period range.

    The velocity model needs thickness, vp, vs, rho.  The period range
    is read in from swd_in.period.

    This is based on Matlab code from Bill Menke via Natalie Accardo
         - MATLAB code includes following copyright info
    Copyright 2003 by Glenn J. Rix and Carlo G. Lai
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation

    The algorithms are based on:
            Hisada, Y., (1994). "An Efficient Method for Computing Green's
        Functions for a Layered Half-Space with Sources and Receivers at Close
        Depths," Bulletin on the Seismological Society of America, Vol. 84,
        No. 5, pp. 1456-1472.
            Lai, C.G., (1998). "Simultaneous Inversion of Rayleigh Phase
        Velocity and Attenuation for Near-Surface Site Characterization," Ph.D.
         Dissertation, Georgia Institute of Technology.

    Arguments
        - model (define_earth_model.EarthModel):
            Input velocity and density model.
        - periods (np.array):
            Periods of interest (s).

    Returns:
        - calculated phase velocity:
            .period: same as input periods
            .cr: caluclated phase velocity (km/s)

    """

    freq = 1/periods

    if model.vs.size == 1:
        cr = _Rayleigh_phase_velocity_in_half_space(model.vp[0], model.vs[0])
        return PhaseVelocity(period = periods,
                             c = np.ones(freq.size)*cr,
        )
        # No frequency dependence in homogeneous half space.

    # Set bounds of tested Rayleigh wave phase velocity (c).
    cr_max = np.max(model.vs)
    cr_min = 0.98 * _Rayleigh_phase_velocity_in_half_space(np.min(model.vp),
                                                 np.min(model.vs))
    omega = 2 * np.pi * freq
    n_ksteps = 20 # assume this is finely spaced enough for our purposes
        #  Original code had 200, but this should speed things up
    cr = np.zeros(omega.size)
    mu = model.rho * model.vs**2

    # Look for the wavenumber (i.e. velocity given fixed frequency)
    # with minimum secular function value
    k_lims = np.vstack((omega/cr_max, omega/cr_min))
    for i_om in range(omega.size):
        # Limiting wavenumber search range breaks things, FYI
        cr[i_om] = _min_value_secular_function(omega[i_om], k_lims[:,i_om],
              n_ksteps, model.thickness, model.rho, model.vp, model.vs, mu)

    return PhaseVelocity(period = periods, c = cr)

def _make_3D(x):
    """Transpose a vector into the third dimension (1 x 1 x size)"""

    return np.reshape(x,(1,1,x.size))

def _Rayleigh_phase_velocity_in_half_space(vp, vs):
    """ Calculate the (frequency independent) phase velocity in a half space.

    Phase velocity can be estimated from the Vp and Vs of the half space
    (Achenbach, 1973).

    Note: a more accurate estimate is possible (see commented code,
    which calculates the roots to Rayleigh's equation, then finds the
    one that is closest to the estimated velocity).  However, it is
    5x faster to just use the estimated phase velocity, and it seems to
    give answers within about 0.5% (for Vp/Vs > 1.5).

    Arguments
        - vp (float):
            Vp in the half space (km/s)
        - vs (float):
            Vs in the half space (km/s)

    Returns
        - estimated_phase_vel_rayleigh (float):
            Estimated Rayleigh phase velocity, c, in the half space (km/s)
    """

    vp2=vp * vp
    vs2=vs * vs
    nu = (0.5*vp2 - vs2) / (vp2-vs2) # Poisson's ratio

    # the estimated velocity (Achenbach, 1973)
    estimated_phase_vel_rayleigh = vs * ((0.862 + 1.14*nu) / (1+nu));

    # # # # Using Rayleigh's equation # # # #
    # # Define Coefficients of Rayleigh's Equation
    #a =  1
    #b = -8
    #c =  8 * (3 - 2*(vs*vs) / (vp*vp))
    #d = 16 * ((cvs*cvs)/(cvp*cvp) - 1)
    #
    # # Solution of Rayleigh Equation
    #p   = np.array([a b c d])
    #x   = np.roots(p)
    #cr  = vs * np.sqrt(x)
    #
    # # Determine the correct root using the estimated phase velocity
    #misfit = abs(cr - estimated_phase_vel_rayleigh)
    #phase_vel_rayleigh = cr(misfit == min(misfit));

    return estimated_phase_vel_rayleigh

def _min_value_secular_function(omega, k_lims, n_ksteps,
                                thick, rho, vp, vs, mu):
    """Return the phase velocity for this period (omega).

    The secular function takes wavenumber, angular frequency (omega), and a
    velocity model (vp, vs, rho) as input, and the value returned is closest
    to zero when the wavenumber and angular frequency correspond to the right
    phase velocity for the velocity model.  This function searches for the
    minimum value of the secular function, and returns the corresponding
    wavenumber, kmin.  (Note this is all done at a single frequency (omega).
    Thus, phase velocity is omega/kmin.

    In order for the matlab.findmin() function to work, it must know a bracket
    (k3, k2, k1) where k3 > k2 > k1 and f(k3) > f(k2) & f(k1) > f(k2).
    Furthermore, to assign a phase velocity, the secular function must return a
    value less than the assigned tolerance, tol_s.  If no phase velocity is
    calculated, this function will recursively call itself with a more finely
    sampled search range (increased n_ksteps), i.e. decreased learning rate.

    Arguments
        - omega (float):
            Angular frequency (Hz), omega = 2 * pi * frequency.
        - k_lims (2 x 1 np.array):
            Search range for wavenumber (1 / km) - i.e. phase velocity guesses.
            The search range is from the maximum Vs in the input model to the
            calculated phase velocity in a half space of the minimum Vs in the
            input model (converted to k by k = omega/phase velocity).
        - n_ksteps (int):
            Granularity with which to search between k_lims - if no suitable
            k is found with input n_ksteps, the search is restarted with
            higher n_ksteps.
        - thick (np.array):
            Vector of layer thicknesses (km), length n_layers.
        - rho (np.array):
            Vector of density (g / cm^3), length n_layers.
        - vp (np.array):
            Vector of Vp (km/s), length n_layers.
        - vs (np.array):
            Vector of Vs (km/s), length n_layers.

    Returns
        - c (float):
            Best guess at the phase velocity (km/s), calculated as omega/kmin,
            where kmin corresponds to the k that returned the lowest value of
            the secular function.
    """

    # Cycle through wavenumbers (i.e. velocity)
    tol_s = 0.1 # Maximum permissable value for the secular function

    # Define vector of possible wavenumbers to try
    wavenumbers = np.linspace(k_lims[0], k_lims[1], n_ksteps)

    # Loop through wavenumbers from largest (slowest) to smallest (fastest)
    # Function to find the minimum requires a bracket of values k3 > k2 > k1
    # where f(k1) > f(k2) and f(k3) > f(k2).  These values are assinged as
    # we loop through wavenumber, but in the meantime, we assign arbitarily
    # small values to f1, f2, k2
    f1 = 1e-10
    f2 = 1e-9
    k2 = 0
    c = 0
    for i_k in range(-1, -(wavenumbers.size+1), -1):
        k3 = wavenumbers[i_k]

        f3 = _secular(k3, omega, thick, mu, rho, vp, vs)

        if f2 < f1 and f2 < f3: # if f2 has minimum of the 3 values
            # Find minimum more finely
            # Search over range k3 to k1 (given f(k2) is the smallest)
            # using additional arguments, args
            # Returns kmin, the optimum point, and fmin, the optimum value
            kmin, fmin = matlab.findmin(_secular, brack = (k3, k2, k1),
                                 args = (omega, thick, mu, rho, vp, vs))
            if fmin < tol_s:
                c = omega/kmin
                break

        else:
             f1 = f2  # cycle through wavenumber values
             f2 = f3
             k1 = k2
             k2 = k3

    # If we couldn't find an appropriate value of the wavenumber, try sampling
    # the k range more finely (increase n_ksteps).
    if c == 0 and n_ksteps <= 250:
        print(n_ksteps)
        c = _min_value_secular_function(omega, k_lims, n_ksteps+100, thick,
                                         rho, vp, vs, mu)

    return c


def _secular(k:float, om:float, thick:np.array, mu:np.array,
             rho:np.array, vp:np.array, vs:np.array) -> float:
    """ Calculate the secular function.

    The smaller the returned value, the closer the wavenumber is to the
    appropriate one of the phase velocity.

    The Green's function is based on modified generalised reflection
    and transmission coefficients (see Hisada, 1994, BSSA; Appendix A)

    Arguments:
        - k (float):
            Trial wavenumber (effectively phase velocity, given fixed angular
            frequency; increasing k decreases phase velocity).
            Wavenumber = 2 * pi / wavelength
        - om (float):
            Fixed angular frequency, omega = 2 * pi * frequency
        - thick (np.array):
            Vector of layer thicknesses from velocity model, length n_layers.
        - mu (np.array):
            Vector of mu = rho * vs**2, length n_layers.
        - rho (np.array):
            Vector of density, length n_layers.
        - vp (np.array):
            Vector of Vp, length n_layers.
        - vs (np.array):
            Vector of Vs, length n_layers.

    Returns:
        - d (float):
            The absolute value of the secular function.  The smaller this
            value, the closer the guessed wavenumber to the real phase velocity.
    """

    # Check to see if the trial phase velocity is equal to the shear wave
    # velocity or compression wave velocity of one of the layers
    # (k = 2*pi/wavelength, so increasing k for given omega decreases velocity)
    epsilon = 0.0001;
    while (np.any(np.abs(om/k - vs) < epsilon)
          or np.any(np.abs(om/k - vp) < epsilon)):
        k = k * (1+epsilon)


    # First, calculate some commonly used variables
    k = k + 0j # make k complex
    n_layers = mu.size
    nu_s = np.sqrt(k**2 - om**2/vs**2) # 1 x n_layers
    inds = np.imag(-1j * nu_s) > 0
    nu_s[inds] = -nu_s[inds]
    gamma_s = _make_3D(nu_s / k) # 1 x 1 x n_layers
    nu_p = np.sqrt(k**2 - om**2/vp**2) # 1 x n_layers
    inds = np.imag(-1j * nu_p)>0
    nu_p[inds] = -nu_p[inds]
    gamma_p = _make_3D(nu_p / k) # 1 x 1 x n_layers
    chi = 2*k - (om**2/vs**2)/k  # nk x n_layers
    thick = _make_3D(thick)

    # Calculate the E and Lambda matrices (up-going and down-going matrices)
    # for the P-SV case.
    # These are used to calculate the dynamic displacement stresses
    #       dynamic displacement stresss =
    #                   E matrix * Lambda matrix * coefficient matrix
    #   (see Hisada, 1994: Appendix A)
    vector_ones = np.ones((1,1,n_layers))
    vector_zeros = np.zeros((1,1,n_layers))
    mu_chi = _make_3D(mu*chi)
    two_mu_nu_p = _make_3D(2*mu*nu_p)
    two_mu_nu_s = _make_3D(2*mu*nu_s)

    # E matrices are 2 x 2 x n_layers, where the third dimension is f(depth)
    # (as gamma_* etc are from the depth dependent velocity model)
    E11 = np.vstack((np.hstack((-vector_ones,gamma_s)),
                     np.hstack((-gamma_p, vector_ones))))
    E12 = np.vstack((np.hstack((-vector_ones, gamma_s)),
                     np.hstack((gamma_p, -vector_ones))))
    E21 = np.vstack((np.hstack((two_mu_nu_p, -mu_chi)),
                     np.hstack((mu_chi, -two_mu_nu_s))))
    E22 = np.vstack((np.hstack((-two_mu_nu_p, mu_chi)),
                     np.hstack((mu_chi, -two_mu_nu_s))))
    # Lambda matrix
    du  = np.vstack((np.hstack((np.exp(-nu_p*thick), vector_zeros)),
                     np.hstack((vector_zeros, np.exp(-nu_s*thick)))))

    # Initialise X as a 4 x 4 x (n_layers-1) complex matrix
    # X will contain the upgoing and downgoing R/T coefficients:
    #      X[:2, :2, iv] is the downgoing transmission coefficient.
    #      X[2:, :2, iv] is the downgoing reflection coefficient.
    #      X[:2, 2:, iv] is the upgoing reflection coefficient.
    #      X[2:, 2:, iv] is the upgoing transmission coefficient.
    # (see EQ A18, Hisada, 1994, BSSA)
    X = np.zeros((4,4,n_layers-1))+0j

    # Loop through the first N-1 layers (i.e. not touching the halfspace).
    for iv in range(n_layers - 2):
        # Generate some layer/boundary specific 4 x 4 matrices
        # Layer below Vs and current layer Vp (ish)
        A = np.vstack((np.hstack((E11[:,:,iv+1],-E12[:,:,iv])),
                       np.hstack((E21[:,:,iv+1], -E22[:,:,iv]))))
        # Current layer Vs and layer below Vs (ish)
        B = np.vstack((np.hstack((E11[:,:,iv],-E12[:,:,iv+1])),
                       np.hstack((E21[:,:,iv], -E22[:,:,iv+1]))))
        L = np.vstack((np.hstack((du[:,:,iv],np.zeros((2,2)))),
                       np.hstack((np.zeros((2,2)),du[:,:,iv+1]))))
        X[:,:,iv] = matlab.mldivide(A,np.matmul(B,L))

    # And the deepest layer (above the halfspace)
    iv = n_layers - 2
    A = np.vstack((np.hstack((E11[:,:,iv+1],-E12[:,:,iv])),
                   np.hstack((E21[:,:,iv+1], -E22[:,:,iv]))))
    B = np.vstack((E11[:,:,iv],E21[:,:,iv]))
    L = du[:,:,iv]
    X[:,:2,-1] = matlab.mldivide(A, np.matmul(B,L))

    #  Calculate the modified Reflection/Transmission coefficients.
    Td = np.zeros((2,2,mu.size-1)) + 0j
    Rd = np.zeros((2,2,mu.size-1)) + 0j

    # Coefficients for the deepest layer above halfspace.
    Td[:, :, n_layers-2] = X[:2, :2, n_layers-2]
    Rd[:, :, n_layers-2] = X[2:, :2, n_layers-2]
    # Coefficients for other layers.
    for iv in range(mu.size-3,-1,-1):
        Td[:,:,iv] = matlab.mldivide(
                        (np.identity(2) - np.matmul(X[:2,2:,iv],Rd[:,:,iv+1])),
                        X[:2,:2,iv])
        Rd[:,:,iv] = X[2:,:2,iv] + np.matmul(np.matmul(X[2:,2:,iv],Rd[:,:,iv+1]),
                                              Td[:,:,iv])


    # And finally, calculate the absolute value of the secular function.
    # The smaller this value, the closer the wavenumber to the phase velocity.
    d = (np.abs(np.linalg.det(E21[:,:,0] +
                      np.matmul(np.matmul(E22[:,:,0],du[:,:,0]), Rd[:,:,0]))
                    /(nu_s[0]*nu_p[0]*mu[0]**2)))



    return d
