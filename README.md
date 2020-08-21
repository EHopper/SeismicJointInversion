# Description
Here, we (Hopper & Gaherty, in prep) present a new joint inversion of two seismic datasets. Surface-wave phase velocities provide volumetric constraints on absolute velocities in the upper mantle. Common-conversion-point  (CCP) images of S-to-P converted phases (receiver functions) place these velocities in the context of a layered lithosphere-asthenosphere system, quantifying the change in physical characteristics across the boundary.  

Our philosophy in this inversion is to capitalise on our geological intuition that, to first order, the shallow velocity structure of the Earth can be described by three layers coinciding with the crust, lithospheric mantle, and asthenospheric mantle. The boundaries between these layers are clearly observed in receiver function studies.  By modelling such observations, we constrain the travel time to and velocity contrast across these boundaries.

Furthermore, we use the lack of significant converted wave signals from depths other than these boundaries. We infer that there are no sharp changes in velocity (i.e. velocity gradients tend to zero) within each layer. This allows us to produce stable inversions without relying on a complex web of ad hoc regularisation.

We follow the inversion framework of Russell et al. (2019) and Menke (2012).

# Installation
This codebase is written in Python 3. (So remember to install libraries with pip3!)
Python 3.6.9

## Required libraries:
- numpy (1.18.2)
- pandas (0.24.2)
- urllib.request
- xarray (0.12.1)
- random
- unittest
- parameterized.parameterized
- pandas
- pathlib
- xarray
- os
- itertools
- spectrum
- scipy


Plan for inversion:

1. Get it working with surface wave data only first.

    a. Set up starting velocity model
    b. Set up forward model for surface waves  (d = Gm)
          - go from velocity model to predicted phase velocities
    c. Set up the inversion (m = (G'G)^-1 G'd)
          - go from misfit to observations to updated model (see below)
    d. Repeat b & c with updated starting model



      More on that inversion...
          - m = (G' * We * G  + ε^2 * Wm )^-1  (G' * We * d + ε^2 * Wm * <m>)
          - G matrix is the Frechet kernels for each model parameter as rows,
            with each row representing a different period
            (n_periods x n_depth_points*n_model_parameters matrix)
          - Model is a column vector with the model parameters as f(depth)
          - Actually we're damping towards a model, m, instead of dm
                G(m-m0) = dobs - d0
                Gm = G*m0 + dobs - d0
                Gm = Ddobs
          - Data: dCL = observed_phase_vel(i_period) - MINEOS_phase_vel(i_period)
                  dCR - as dCL but for Rayleigh not Love
                  dobs = [dCL, dCR]'*1000; (misfit in m/s)

                  Dd = G*m0
                  DdCL = Dd(Love_periods)' + dCL
                  DdCR = Dd(Rayleigh periods)' + dCR
                  Ddobs = [DdCL, DdCR]';
          - Inversion formulated as...
                  F * m_est  = f
                  F  	=   [    sqrt(We) * G ;    εD   ]
                  f 	=   [    sqrt(We) * d ;    εD<m>   ]
                  (Wm 	=   D' * D  (model damping weighting matrix))
                  Where D = H, D<m> = h (constraint equations & damping)

                  H is (n_constraints x n_depth_points*n_model_parameters)
                    H = stack of all individual constraints (e.g. second order
                        smoothing, a priori info)
                  h is (n_constraints x 1) vector
                    h = zero, often
                  Both H and h are multiplied by layer specific damping parameters
                  (ε in above equation), often called damp_XXXXX - a preset constant
                  These are then further damped by 'epsilon_HhD' (= 1), so can control
                  overall significance of a priori constraints vs. data constraints

                  We is (n_periods x n_periods) matrix for weighting the data
                    We = diagonal matrix, where the value is 1/std(data at that period)
                  Data are further damped by 'epsilon_Gmd_vec' (set to about 0.1,
                  but dependent on period here).

                  F = [epsilon_Gmd_vec * sqrt(We) * GG;     epsilon_HhD * H]
                  f = [epsilon_Gmd_vec * sqrt(We) * Ddobs;  epsilon_HhD * h]

                  Finv = (F'*F + epsilon_0norm * eye(NF, NF)) \ F';
                      This is damped least squares: m = (G'G + ε^2 I) G' d
                      epsilon_0norm = 1e-11, hardwired.

                  So in addition to all of the data quality, layer, and
                  data vs. a priori specific damping, there is general damping
                  applied to the final inversion.
