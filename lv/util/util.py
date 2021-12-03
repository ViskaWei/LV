import os
import sys
# import getpass
import numpy as np
import scipy as sp
import pandas as pd
import h5py
from tqdm import tqdm
from scipy.stats import chi2 as chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Patch
import matplotlib.transforms as transforms

from matplotlib.lines import Line2D
from matplotlib import collections  as mc

class Util():
    def __init__(self):
        
    
    
        pass
#--------------------
    @staticmethod
    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def get_arg(name, old_value, args):
        if name in args and args[name] is not None:
            return args[name]
        else:
            return old_value

    @staticmethod
    def is_arg(name, args):
        return name in args and args[name] is not None

    @staticmethod
    def func_fullname(f):
        m = sys.modules[f.__module__]
        return '{}.{}'.format(m.__name__, f.__qualname__)


    @staticmethod
    def get_para_index(Ps, dfp, Pnms=None):
        if Pnms is None: Pnms = dfp.columns
        for ii, P in enumerate(Pnms):
            dfp = dfp[dfp[P]==Ps[ii]]
        return dfp.index

    @staticmethod
    def safe_log(x):
        return np.log(np.where(x <= 1, 1, x))

    @staticmethod
    def normlog_flux(fluxs):
        logflux = np.log(np.where(fluxs <= 1, 1, fluxs))
        normlogflux = logflux - logflux.mean(1)[:,None]
        return normlogflux

    @staticmethod
    def norm_flux(fluxs, log=1):
        if len(fluxs.shape) == 1: 
            fluxs = fluxs[:,None]
        if log:
            normflux = fluxs - fluxs.mean(1)[:,None]
        else:
            normflux = fluxs / fluxs.mean(1)[:,None]
        if len(fluxs.shape) == 1: 
            normflux = normflux[0]
        return normflux

    @staticmethod
    def normlog_flux_i(flux):
        logflux = np.log(np.where(flux <= 1, 1, flux))
        normlogflux = logflux - logflux.mean()
        return normlogflux

    @staticmethod
    def lognorm_flux(fluxs):
        fluxs = np.where(fluxs>0, fluxs, 1e-20)
        norm_flux = np.divide(fluxs, fluxs.mean(1)[:,None])
        # norm_flux = np.where(norm_flux <= 0, 0, norm_flux)
        lognormflux = np.log(norm_flux)
        return lognormflux

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
        if step==0:
            wave1 = wave
        else:
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
        if step==0: return flux
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
        if step == 0: return fluxs
        out = np.zeros((len(fluxs), L))
        for ii, flux in enumerate(fluxs):
            out[ii] = Util.resampleFlux_i(flux, step=step)
        return out

    @staticmethod
    def resampleSky(sky, wave_grid, step=5):
        #-----------------------------------------------------
        # resample the sky matching the spectrum
        #-----------------------------------------------------
        ws = sky[:,0]
        cs = np.cumsum(sky[:,1])
        f = sp.interpolate.interp1d(ws,cs, fill_value=0)
        #----------------------------------------------------
        # get the breakpoints in lambda
        #--------------------------------

        #---------------------------------------------------
        # interpolate the cumulative sky to the breakpoints
        #---------------------------------------------------
        if step==0: 
            sky_new = np.diff(f(wave_grid))
            sky_new = np.insert(sky_new, 0, f(wave_grid[0]))
        else:
            b = list(range(1,wave_grid.shape[0],step))
            sky_new = np.diff(f(wave_grid[b]))
        # assert sky_new.shape == wave_grid.shape
        return sky_new

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


    def flow_fn(self,paras, center, legend=0):
        fpara=self.scatter_fn(paras, c="r",s=10)
        fmean=self.scatter_fn(center, c="g",s=10)
        ftraj=self.traj_fn(paras, center, c="r",lw=2)
        return [fpara,fmean, ftraj]

    def flow_fn_i(self, pred, center, snr=None, legend=0):
        mu, sigma = pred.mean(0), pred.std(0)
        center = np.array([center])
        MU = np.array([mu])
        lgd =f"SNR={snr}" if legend else None
        
        fpred=self.scatter_fn(pred, c="gray",s=10, lgd=lgd)
        fmean=self.scatter_fn(MU,  c="r",s=10)
        ftarget=self.scatter_fn(center,  c="g",s=10)
        ftraj=self.traj_fn(MU, center, c="r",lw=2)
        add_ellipse = self.get_ellipse_fn(pred,c='b', legend=legend)


        return [fpred,fmean, ftarget,ftraj, add_ellipse]


    @staticmethod
    def traj_fn(strts, ends, c=None, lw=2):
        nTest=strts.shape[0]
        def fn(i, j , ax, handles=[]):
            flowList=[]
            for ii in range(nTest):
                strt=strts[ii]
                end= ends[ii]
                flowList.append([(strt[i],strt[j]), (end[i],end[j])])
            lc = mc.LineCollection(flowList, colors=c, linewidths=lw)
            ax.add_collection(lc)
            return handles
        return fn


    @staticmethod
    def scatter_fn(data, c=None, s=1, lgd=None):
        def fn(i, j, ax, handles=[]):
            ax.scatter(data[:,i], data[:,j],s=s, c=c)
            if lgd is not None: 
                handles.append(Line2D([0], [0], marker='o',color='w', label=lgd, markerfacecolor=c, markersize=10))
            return handles
        return fn

    @staticmethod
    def get_ellipse_params(pred, npdx=None):
        if npdx is None: npdx = pred.shape[1]
        x0s,y0s,s05s,degs = [],[],[],[]
        for ii in range(npdx):
            jj = 0 if ii ==npdx-1 else ii + 1
            x0, y0, s05, degree = Util.get_ellipse_param(pred[:,ii], pred[:,jj])
            x0s.append(x0)
            y0s.append(y0)
            s05s.append(s05)
            degs.append(degree)
        return x0s,y0s,s05s,degs

    @staticmethod
    def get_ellipse_param(x, y):
        x0,y0=x.mean(0),y.mean(0)
        _, s, v = np.linalg.svd(np.cov(x,y))
        s05 = s**0.5
        degree = Util.get_angle_from_v(v)
        return x0, y0, s05, degree

    @staticmethod
    def get_ellipse_fn(data, c=None, ratio=0.95, legend=1):
        x0s,y0s,s05s,degrees = Util.get_ellipse_params(data)
        chi2_val = chi2.ppf(ratio, 2)
        co = 2 * chi2_val**0.5
        if c is None: c = "r"
        def add_ellipse(i, j, ax, handles):
            x0, y0, s05, degree = x0s[i], y0s[i], s05s[i], degrees[i]
            e = Ellipse(xy=(0,0),width=co*s05[0], height=co*s05[1], facecolor="none",edgecolor=c,)
            transf = transforms.Affine2D().rotate_deg(degree).translate(x0,y0) + ax.transData        
            e.set_transform(transf)
            ax.add_patch(e)
            if legend:
                handles.append(Ellipse(xy=(0,0),width=2, height=1, facecolor="none",edgecolor=c,label=f"Chi2_{100*ratio:.0f}%"))
            return handles
        return add_ellipse

    @staticmethod
    def box_fn(pRange, pMin, pMax,  n_box=None, c="k"):
        def fn(i, j , ax, handles=[]):
            if n_box is not None:
                ax.set_xlim(pMin[i]-n_box*pRange[i], pMax[i]+n_box*pRange[i])
                ax.set_ylim(pMin[j]-n_box*pRange[j], pMax[j]+n_box*pRange[j])
            ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor=c,lw=2, facecolor="none"))
            handles.append(Patch(facecolor='none', edgecolor=c, label=f"Box")) 
            return handles
        return fn

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
    def fmn(x):    
        return '{:02d}'.format(np.floor(x).astype(np.int32))

    @staticmethod
    def fmt(x):
        y = np.round(np.abs(10*x)+0.2).astype(np.int32)
        z = '{:+03.0f}'.format(y).replace('+','p')
        if (np.sign(x)<0):
            z = z.replace('p','m')
        return z

    @staticmethod
    def getname(m,t,g,c,a):
        #----------------------------------
        # get short name for the spectrum
        #----------------------------------
        fname = 'T'+ Util.fmn(t)+'G'+Util.fmn(10*g)+'M'+Util.fmt(m)+'A'+Util.fmt(a)+'C'+Util.fmt(c)
        return fname
    
    @staticmethod
    def getSN(flux):
        #--------------------------------------------------
        # estimate the S/N using Stoehr et al ADASS 2008
        #    signal = median(flux(i))
        #    noise = 1.482602 / sqrt(6.0) *
        #    median(abs(2 * flux(i) - flux(i-2) - flux(i+2)))
        #    DER_SNR = signal / noise
        #--------------------------------------------------
        s1 = np.median(flux)
        s2 = np.abs(2*flux-sp.ndimage.shift(flux,2)-sp.ndimage.shift(flux,-2))
        n1 = 1.482602/np.sqrt(6.0)*np.median(s2)
        sn = s1/n1
        return sn

    @staticmethod
    def shiftSpec(flux, rv):
        #--------------------------------------------------
        # The radial velocity rv is given in km/sec units
        # This must be done at the hirez pixel resolution
        #--------------------------------------------------
        return sp.ndimage.shift(flux, rv/3.0)

    @staticmethod
    def convolveSpec(flux,step=5, sg=1.3):
        #---------------------------------------------------------
        # apply a line spread function to the h-pixel spectrum
        # with a width of 1.3 m-pixels. This is using the log(lambda) 
        # binning in hirez.
        #-----------------------------------
        # create resampled kernel by step
        # precompute the kernel, if there are many convolutions
        #-----------------------------------
        xx = np.linspace(-7*step, 7*step, 14*step+1)
        yy = np.exp(-0.5*xx**2/sg**2)/np.sqrt(2*np.pi*sg**2)
        fspec = np.convolve(flux,yy,'same')
        return fspec

    @staticmethod
    def makeNLArray(ss, skym, step=5):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------
        noise_level_grid = [2,5,10,20,50,100,200,500]
        snrList = [11,22,33,55,110]
        
        # ssm   = Util.getModel(ss,0)
        ssm   = Util.resampleFlux_i(ss, step                        )       
        varm  = Util.getVar(ssm,skym)
        noise = Util.getNoise(varm)  

        SN = []
        for noise_level in noise_level_grid:
            ssobs = ssm + noise_level * noise
            sn    = Util.getSN(ssobs)
            SN.append(sn)
        f = sp.interpolate.interp1d(SN, noise_level_grid, fill_value=0)
        
        noise_level_interpd = f(snrList)  
        return noise_level_interpd


    
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
    def get_noise_N(varm, N):
        out = np.zeros((N,varm.shape[0]))
        for i in range(N):
            out[i] = np.random.normal(0, np.sqrt(varm), len(varm))
        return out

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
        # np.random.seed(42)
        noise = np.random.normal(0, np.sqrt(varm), len(varm))
        return noise

    @staticmethod
    def getObs(sconv,skym,rv, noise_level, step=5):
        #----------------------------------------------------
        # get a noisy spectrum for a simulated observation
        #----------------------------------------------------
        # inputs
        #   sconv: the rest-frame spectrum in h-pixels, convolved
        #   skym: the sky in m-pixels
        #   rv  : the radial velocity in km/s
        #   noise_level  : the noise amplitude
        # outputs
        #   ssm : the shifted, resampled sepectrum in m-pix
        #   varm: the variance in m-pixels
        #-----------------------------------------------
        # get shifted spec and the variance
        #-------------------------------------
        ssm   = Util.getModel(sconv, rv, step=step)
        varm  = Util.getVar(ssm,skym)
        noise = Util.getNoise(varm)  
        #---------------------------------------
        # add the scaled noise to the spectrum
        #---------------------------------------
        ssm = ssm + noise_level * noise
        return ssm
    





