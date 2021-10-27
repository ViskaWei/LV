import os
import getpass
import numpy as np
import scipy as sp
import pandas as pd
import h5py
from tqdm import tqdm
from scipy.stats import chi2 as chi2
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse


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


    @staticmethod
    def lognorm_flux(fluxs):
        fluxs = np.where(fluxs>0, fluxs, 1e-20)
        norm_flux = np.divide(fluxs, fluxs.mean(1)[:,None])
        # norm_flux = np.where(norm_flux <= 0, 0, norm_flux)
        LNflux = np.log(norm_flux)
        return LNflux

    # @staticmethod
    # def safe_log(x):
    #         a = np.exp(args[0]) if args is not None else 1e-10
    #         return np.log(np.where(x < a, a, x))

    @staticmethod
    def lognorm_flux_i(flux):
        return np.log(np.divide(flux, flux.mean()))
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

# Plot ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_correlated_dataset(n=1000, cov=[[1, -2],[0.3, 1]], mu=(2,4), scale=(1,10)):
        latent = np.random.randn(n, len(mu))
        dependent = latent.dot(cov)
        scaled = dependent * scale
        scaled_with_offset = scaled + mu
        # return x and y of the new, correlated dataset
        return scaled_with_offset
    
    def plot3(self, fns=[], data=None, lbl=["MH","Teff","Logg"]):
        f, axs = plt.subplots(1, 3 ,  figsize=(16, 4), facecolor="w")
        for ii, ax in enumerate(axs):
            jj = 0 if ii == 2 else ii + 1
            if data is not None:
                x, y = data[:,jj], data[:,ii]
                ax.scatter(x, y, s=1, alpha=0.5, color="k")
            for fn in fns:
                fn(ii,jj,ax,[])            
            ax.set_xlabel(lbl[ii])            
            ax.set_ylabel(lbl[jj])           

    @staticmethod
    def set_unique_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
     






    @staticmethod
    def get_ellipse_fn_2d(x,y,df):
        x0,y0=x.mean(0),y.mean(0)
        # x1,y1=x.std(0),y.std(0)
        # _, s, v = np.linalg.svd(np.cov(x-x0,y-y0))
        _, s, v = np.linalg.svd(np.cov(x,y))
        s05 = s**0.5
        degree = Util.get_angle_from_v(v) 
        def add_ellipse(ratio, c="k", ax=None):
            chi2_val = chi2.ppf(ratio, df)
            co = 2 * chi2_val**0.5
            e = Ellipse(xy=(0,0),width=co*s05[0], height=co*s05[1], facecolor="none",edgecolor=c,label=f"Chi2_{100*ratio:.0f}%")
            transf = transforms.Affine2D().rotate_deg(degree).translate(x0,y0) + ax.transData        
            e.set_transform(transf)
            ax.add_patch(e)
            ax.plot(x0,y0,"go")
        return add_ellipse

    @staticmethod
    def get_angle_from_v(v, idx=0):
        radian =np.arctan(v[idx][1] / v[idx][0])
        degree = radian / np.pi * 180    
        return degree

    # def add_ellipse(ratio, c="k", ax=None):
    #     chi2_val = chi2.ppf(ratio, df)
    #     co = 2 * chi2_val**0.5
    #     e = Ellipse(xy=(0,0),width=co*s05[0], height=co*s05[1], facecolor="none",edgecolor=c,label=f"Chi2_{100*ratio:.0f}%")
    #     transf = transforms.Affine2D().rotate_deg(degree).translate(x0,y0) + ax.transData        
    #     e.set_transform(transf)
    #     ax.add_patch(e)
    #     ax.plot(x0,y0,"go")

    def get_ellipse_fn_ij(mean,s,v):

        x0,y0=x.mean(0),y.mean(0)
        def add_ellipse(ratio, c="k", ax=None):
            chi2_val = chi2.ppf(ratio, df)
            co = 2 * chi2_val**0.5
            e = Ellipse(xy=(0,0),width=co*s05[0], height=co*s05[1], facecolor="none",edgecolor=c,label=f"Chi2_{100*ratio:.0f}%")
            transf = transforms.Affine2D().rotate_deg(degree).translate(x0,y0) + ax.transData        
            e.set_transform(transf)
            ax.add_patch(e)
            ax.plot(x0,y0,"go")
        return add_ellipse

    @staticmethod
    def plot_correlated(x,y, chis=[0.95,0.99]):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(x,y)
        x0,y0=x.mean(0),y.mean(0)
        ax.plot(x0,y0,"ro")
        add_e = Util.get_ellipse_fn_2d(x,y,2)
        for ratio in chis:
            add_e(ratio, ax=ax)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_title("Correlated data")
        ax.legend()


# Alex ----------------------------------------------------------------------------------------------------------------------   
    @staticmethod
    def getSN(spec):
        #--------------------------------------------------
        # estimate the S/N using Stoehr et al ADASS 2008
        #    signal = median(flux(i))
        #    noise = 1.482602 / sqrt(6.0) *
        #    median(abs(2 * flux(i) - flux(i-2) - flux(i+2)))
        #    DER_SNR = signal / noise
        #--------------------------------------------------
        s1 = np.median(spec)
        s2 = np.abs(2*spec-sp.ndimage.shift(spec,2)-sp.ndimage.shift(spec,-2))
        n1 = 1.482602/np.sqrt(6.0)*np.median(s2)
        sn = s1/n1
        return sn


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