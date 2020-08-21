# Description
Here, we (Hopper & Gaherty, in prep) present a new joint inversion of two seismic datasets. Surface-wave phase velocities provide volumetric constraints on absolute velocities in the upper mantle. Common-conversion-point  (CCP) images of S-to-P converted phases (receiver functions) place these velocities in the context of a layered lithosphere-asthenosphere system, quantifying the change in physical characteristics across the boundary.  

Our philosophy in this inversion is to capitalise on our geological intuition that, to first order, the shallow velocity structure of the Earth can be described by three layers coinciding with the crust, lithospheric mantle, and asthenospheric mantle. The boundaries between these layers are clearly observed in receiver function studies.  By modelling such observations, we constrain the travel time to and velocity contrast across these boundaries.

Furthermore, we use the lack of significant converted wave signals from depths other than these boundaries. We infer that there are no sharp changes in velocity (i.e. velocity gradients tend to zero) within each layer. This allows us to produce stable inversions without relying on a complex web of ad hoc regularisation.

We follow the inversion framework of Russell et al. (2019) and Menke (2012).

# Installation
This codebase is written in Python 3. (So remember to install libraries with pip3!)
- Python (3.6.9)

## Required libraries:
- Libraries that should come with Python
  - re
  - os
  - unittest
  - typing
  - shutil
  - glob
  - subprocess
  - random
- numpy (1.18.2)
- pandas (0.24.2)
- matplotlib (3.1.0)
- scipy (1.4.1)
- sklearn (0.22.1)
- xarray (0.12.1)
- parameterized (0.7.0)

# Usage
## Overview
This is an iterative linearised least-squares inversion.  That is, a linear regression that we solve using a gradient descent algorithm using an L2 loss function.

The observations (target):
- Surface wave tomography
  - Phase velocities at some number of periods
  - In this study, we use Rayleigh wave observations from Jin & Gaherty (2015) and Babikoff & Dalton (2018); ambient noise observations from Ekstrom (2017)
- Receiver functions
  - Vertical travel time lag (e.g. P wave - Ps wave; Sp wave - S wave) from centre of boundary
  - Velocity contrast across boundary given assumed boundary thickness
    - A discontinuous velocity change will give a higher amplitude converted phase than a velocity contrast that is spread over > 10 km vertically
  - In this study, we use Ps observations of the Moho from Shen & Ritzwoller (2013) and Sp observations from Hopper & Fischer (2018)

The model parameters we are solving for (weights in regression):
- Shear wave velocities at given depths
  - Assume velocities are linearly interpreted between nodes
- Depth of boundaries constrained by receiver function observations, e.g. Moho and LAB

The inversion is described more fully (including equations) in this Google Doc: https://docs.google.com/document/d/1xAM5oSZ7ZpLm0aam-XU4YlqknUbjJGVrNO-GHMUYPa8/edit?usp=sharing

## The process, to first order
1. Define your basic model parameters
   - define_models.SetupModel() - see docstring
   - Includes things like the Vp/Vs ratio (default = 1.75); depth range of model (default = 0 to 400 km); the width of the boundary layers (default = 3 km for Moho; 10 km for LAB)
2. Extract your observations (target variables) from saved data
   - constraints.extract_observations() - see docstring
   - Given a location, will pull RF data and surface wave data from previously saved files
   - Details of the data that you need to download is saved in data/obs_dispersion/README and data/RFconstraints/README
3. Define your starting model
   - define_models.InversionModel() - see docstring
   - This defines the shear velocity as a function of depth by listing velocities at the edges of layers, layer thicknesses, and the indices of your RF constrained boundaries
   - Note that as this inversion has no regularisation of model length (i.e. no ridge regularisation), the result is very insensitive to the actual starting model. It is just important to initialise it with the correct format.
4. Iteratively perform the inversion
   - inversion.\_inversion_iteration() - see docstring
   - Can either iterate some set number of times, or break out of loop when the new model is sufficiently similar to the starting model at that iteration
     - Note that you should always set an upper limit to number of iterations, as there are some locations (normally with very slow phase velocities at the shortest periods) which will not converge
  - Steps at this stage
    - Pass the starting model to MINEOS to calculate surface wave kernels and predicted phase velocities
    - Calculate the partial derivatives for gradient descent using the surface wave kernels and the starting model
    - Calculate regularisation matrices (see step 5)
    - Perform the damped least-squares inversion (gradient descent step)
    - Build a new model from the output and reformat it
5. Set regularisation
   - weights.build_weighting_damping() - see docstring
   - Various regularisation options are actually coded up here, but the only ones with non-zero coefficients are from weights.\_build_constraint_damp_zero_gradient()
     - That is, we are only regularising by damping to zero gradient outside of the boundary layers
     - The strength of this regularisation can be adjusted for the crust, lithosphere, and asthenosphere via weights.\_set_layer_values()
6. There are various plotting codes to look at the intermediate and final outputs in plots.py

\
Note: working.py contains functions that I have been using to play around with this code.  In particular, working.try_run() is basically the series of steps listed above, but with a lot of other stuff in there to make useful plots along the way.
