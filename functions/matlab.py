# -*- coding: utf-8 -*-
"""
Matlab compatibility layer. Do things the way matlab arbitrarily chooses to do.
Created on Mon Feb 19 16:15:45 2018
@author: emily
"""

from scipy import signal
import numpy as np
from scipy.signal import butter, detrend
from scipy.optimize import brent
from spectrum.mtm import dpss
# install as conda config --add channels conda-forge; conda install spectrum

def findmin(func, args, brack):
    return brent(func, args, brack, tol=1e-3,
                   full_output = True, maxiter=500)[0:2]

def slepian(n_samp, num_tapers):
    return dpss(N = n_samp, NW=int((num_tapers+1)/2), k=None)[0]

def filtfilt(b, a, data):
    return signal.filtfilt(b, a, data, padlen=3*(max(len(a), len(b)) - 1))

def mldivide(a, b):
    # This works as long as the rank is the same as the number of num vars
    #if np.linalg.matrix_rank(a) == a.shape[1]:
    return np.linalg.lstsq(a, b)[0]
    # from https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
#    else:
#        from itertools import combinations
#        for nz in combinations(range(num_vars), rank):    # the variables not set to zero
#            try:
#                sol = np.zeros((num_vars, 1))
#                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
#            except np.linalg.LinAlgError:
#                pass


# Butterworth bandpass filter edited slightly from
#   http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass

def ButterBandpass(short_T, long_T, dt, order=2):
     nyq = 0.5 / dt
     low = 1 / (nyq*long_T)
     high = 1 / (nyq*short_T)
     b, a = butter(order, [low, high], btype='bandpass')
     return b, a

def BpFilt(data, short_T, long_T, dt, order=2):
     b, a = ButterBandpass(short_T, long_T, dt, order=order)
     data = data - np.mean(data)
     y = filtfilt(b, a, data) # Note: filtfilt rather than lfilt applies a
                               # forward and backward filter, so no phase shift
     y = detrend(y)
     return y

def Taper(data, i_taper_width, i_taper_start, i_taper_end):
    # This will apply a (modified) Hann window to data (a np.array), such that
    # there is a cosine taper (i_taper_width points long) that is equal to 1
    # between i_taper_start (index) and i_taper_end (index)
    taper = np.concatenate([np.zeros(i_taper_start - i_taper_width),
                            np.insert(np.hanning(2*i_taper_width),i_taper_width,
                                      np.ones(i_taper_end - i_taper_start)),
                            np.zeros(data.size - i_taper_end - i_taper_width)])

    return data * taper
