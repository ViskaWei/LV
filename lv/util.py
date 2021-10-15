import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

class Util():
    def __init__(self):
    
    
        pass
#--------------------
    @staticmethod
    def get_para_index(Ps, dfp, Pnms=None):
        if Pnms is None: Pnms = dfp.columns
        for ii, P in enumerate(Pnms):
            dfp = dfp[dfp[P]==Ps[ii]]
        return dfp.index
# ----------------------------------------------------------
    @staticmethod
    def get_flux_in_Wrange(wave, flux, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return wave[start:end], flux[:, start:end]

    @staticmethod
    def get_flux_in_Prange(dfpara, pval):
        Fs, Ts, Ls, _,_ = pval
        maskF = (dfpara["F"] >= Fs[0]) & (dfpara["F"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["L"] >= Ls[0]) & (dfpara["L"] <= Ls[1]) 
        mask = maskF & maskT & maskL
        dfpara = dfpara[mask]
        para = np.array(dfpara.values, dtype=np.float16)
        return dfpara.index, para

# sample ------------------------------------------------------------------------------

    @staticmethod
    def resampleWave(wave,step=5, verbose=1):
        #-----------------------------------------------------
        # resample the wavelengths by a factor step
        #-----------------------------------------------------
        w = np.cumsum(np.log(wave))
        b = list(range(1,wave.shape[0],step))
        db = np.diff(w[b])
        dd = (db/step)
        wave1 = np.exp(dd) 
        if verbose: Util.print_res(wave1)
        return wave1

    @staticmethod
    def resampleFlux_i(flux, step=5):
        #-----------------------------------------------------
        # resample the spectrum by a factor step
        #-----------------------------------------------------
        c = np.cumsum(flux)
        b = list(range(1,flux.shape[0],step))
        db = np.diff(c[b])
        dd = (db/step)
        return dd

    @staticmethod
    def resampleFlux(fluxs, L,step=5):
        out = np.zeros((len(fluxs), L))
        for ii, flux in enumerate(fluxs):
            out[ii] = Util.resampleFlux_i(flux, step=step)
        return out

    @staticmethod
    def resample(wave, fluxs, step=10, verbose=1):
        waveL= Util.resampleWave(wave, step=step, verbose=verbose)
        L = len(waveL)
        fluxL =Util.resampleFlux(fluxs, L, step=step)
        return waveL, fluxL

    @staticmethod
    def print_res(wave):
        dw = np.mean(np.diff(np.log(wave)))
        print(f"#{len(wave)} R={1/dw:.2f}")

# load/save ----------------------------------------------------------------------------------------------------------------------

    
    @staticmethod
    def save(wave, flux, para, SAVE_PATH):
        with h5py.File(SAVE_PATH, "w") as f:
            f.create_dataset(f"flux", data=flux, shape=flux.shape)
            f.create_dataset(f"para", data=para, shape=para.shape)
            f.create_dataset(f"wave", data=wave, shape=wave.shape)   
    @staticmethod
    def load(SAVE_PATH):
        with h5py.File(SAVE_PATH, "r") as f:
            flux = f["flux"][:]
            para = f["para"][:]
            wave = f["wave"][:]
        return wave, flux, para