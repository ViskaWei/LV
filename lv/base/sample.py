import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

def resampleWave(wave,step=5, verbose=1):
    #-----------------------------------------------------
    # resample the wavelengths by a factor step
    #-----------------------------------------------------
    w = np.cumsum(np.log(wave))
    b = list(range(1,wave.shape[0],step))
    db = np.diff(w[b])
    dd = (db/step)
    wave1 = np.exp(dd) 
    if verbose: print_res(wave1)
    return wave1

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

def resample(wave, fluxs, step=10, verbose=1):
    waveL= resampleWave(wave, step=step, verbose=verbose)
    L = len(waveL)
    fluxL = resampleFlux(fluxs, L, step=step)
    return waveL, fluxL

def print_res(wave):
    dw = np.mean(np.diff(np.log(wave)))
    print(f"#{len(wave)} R={1/dw:.2f}")
    
