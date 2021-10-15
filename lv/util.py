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
    def resample_ns(wave, fluxs, errs, step=10, verbose=1):
        waveL= Util.resampleWave(wave, step=step, verbose=verbose)
        L = len(waveL)
        fluxL =Util.resampleFlux(fluxs, L, step=step)
        errL = Util.resampleFlux(errs, L, step=step)
        return waveL, fluxL, errL

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




# Alex ----------------------------------------------------------------------------------------------------------------------   

    @staticmethod
    def convolveSpec(flux,step=5):
        #---------------------------------------------------------
        # apply a line spread function to the h-pixel spectrum
        # with a width of 1.3 m-pixels. This is using the log(lambda) 
        # binning in hirez.
        #---------------------------------------------------------
        sg = 1.3
        #-----------------------------------
        # create resampled kernel by step
        # precompute the kernel, if there are many convolutions
        #-----------------------------------
        xx = np.linspace(-7*step, 7*step, 14*step+1)
        yy = np.exp(-0.5*xx**2/sg**2)/np.sqrt(2*np.pi*sg**2)
        fspec = np.convolve(flux,yy,'same')
        return fspec

    @staticmethod
    def makeNLArray(ss,skym):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------
        nla = [2,5,10,20,50,100,200,500]
        sna = [11,22,33,55,110]
        
        ssm   = getModel(ss,0)
        varm  = getVar(ssm,skym)
        noise = getNoise(varm)  
        NL = []
        SN = []
        for nl in nla:
            ssobs = ssm + nl*noise
            sn    = getSN(ssobs)
            NL.append(nl)
            SN.append(sn)
        f = sp.interpolate.interp1d(SN,NL, fill_value=0)
        nlarray = f(sna)
        
        return nlarray

    @staticmethod
    def getVar(ssm, skym):
        #--------------------------------------------
        # Get the total variance
        # BETA is the scaling for the sky
        # VREAD is the variance of the white noise
        # This variance is still scaled with an additional
        # factor when we simuate an observation.
        #--------------------------------------------
        BETA  = 10.0
        VREAD = 16000
        varm  = ssm + BETA*skym + VREAD
        return varm

    @staticmethod
    def getModel(sconv,rv):
        #-----------------------------------------------------
        # Generate a spectrum shifted by rv. sconv is a high rez
        # spectrum already convolved with the LSF, rv is the 
        # radial velocity in km/s. Here we convolve once for speed, 
        # then apply different shifts and resample.
        #-----------------------------------------------------
        ss1 = shiftSpec(sconv,rv)
        ss1 = resampleSpec(ss1)    
        return ss1
    
    @staticmethod
    def getNoise(varm):
        #--------------------------------------------------------
        # given the noise variance, create a noise realization
        # using a Gaussian approximation
        # Input
        #  varm: the variance in m-pixel resolution
        # Output
        #  noise: nosie realization in m-pixels
        #--------------------------------------------------------
        noise = np.random.normal(0, 1, len(varm))*np.sqrt(varm)
        return noise

    @staticmethod
    def getObs(sconv,skym,rv,NL):
        #----------------------------------------------------
        # get a noisy spectrum for a simulated observation
        #----------------------------------------------------
        # inputs
        #   sconv: the rest-frame spectrum in h-pixels, convolved
        #   skym: the sky in m-pixels
        #   rv  : the radial velocity in km/s
        #   NL  : the noise amplitude
        # outputs
        #   ssm : the shifted, resampled sepectrum in m-pix
        #   varm: the variance in m-pixels
        #-----------------------------------------------
        # get shifted spec and the variance
        #-------------------------------------
        ssm   = getModel(sconv,rv)
        varm  = getVar(ssm,skym)
        noise = getNoise(varm)  
        #---------------------------------------
        # add the scaled noise to the spectrum
        #---------------------------------------
        ssm = ssm + NL*noise
        return ssm