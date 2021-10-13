import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

def resampleWave(wave,step=5):
    #-----------------------------------------------------
    # resample the wavelengths by a factor step
    #-----------------------------------------------------
    w = np.cumsum(np.log(wave))
    b = list(range(1,wave.shape[0],step))
    db = np.diff(w[b])
    dd = (db/step)
    return np.exp(dd)
def resampleFlux_i(flux, step=5):
    #-----------------------------------------------------
    # resample the spectrum by a factor step
    #-----------------------------------------------------
    c = np.cumsum(flux)
    b = list(range(1,flux.shape[0],step))
    db = np.diff(c[b])
    dd = (db/step)
    return dd
def resampleFlux(fluxs, L,step=5):
    out = np.zeros((len(fluxs), L))
    for ii, flux in enumerate(fluxs):
        out[ii] = resampleFlux_i(flux, step=step)
    return out

def resample(wave, flux, step=10, verbose=1):
    waveL= resampleWave(wave, step=step)
    L = len(waveL)
    fluxL = resampleFlux(flux, L, step=step)
    if verbose:
        print(L, fluxL.shape, end=" ")
        print_res(waveL)
    return waveL, fluxL

def print_res(wave):
    dw = np.mean(np.diff(np.log(wave)))
    
