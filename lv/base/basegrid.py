import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from scipy.interpolate import RBFInterpolator

from lv.util.constants import Constants as C
from lv.util.util import Util
from lv.util.obs import Obs

class BaseGrid(object):
    def __init__(self, R):
        self.GRID_DIR = C.GRID_DIR
        self.wave = None
        self.flux = None
        self.Res = 5000
        self.R = R
        self.RR = C.dRR[R]
        self.Util = Util()
        self.step=10
        self.Ws = [7100, 8850]
        self.Obs = Obs()
        self.bnds = None
        self.test = {"pmt": np.array([-2.0, 8000, 2.5, 0.0, 0.25]), "noise_level": 0.1,
            "obsflux": None, "obsvar": None}

    def init_bnds(self, R):
        Rs = C.dRs[R]
        diff = np.diff(Rs).T[0]
        bnds=[]
        for ii in range(5):
            lb, ub = Rs[ii][0], Rs[ii][1]
            bnds.append([lb, ub, diff[ii], C.dDX[ii], 0.5*(lb+ub)])
        self.bnds = np.array(bnds)
        self.scaler = self.get_scaler_fn(self.bnds[:,0], C.dDX)

    def get_scaler_fn(self, mins, ds ):
        def fn(x):
            return np.divide((x-mins) ,ds)
        return fn

    def load_grid(self, PATH=None):
        if PATH is None: PATH = os.path.join(self.GRID_DIR, f"bosz_{self.Res}_{self.RR}.h5")
        with h5py.File(PATH, "r") as f:
            self.wave0 = f["wave"][()]
            self.flux0 = f["flux"][()]
            self.para = f["para"][()]
            self.pdx = f["pdx"][()]
        print(self.wave0.shape, self.flux0.shape, self.para.shape)
        self.pdx0 = self.pdx - self.pdx[0]
        
    def prepro(self, wave, flux):
        wave, flux = self.Util.get_flux_in_Wrange(wave, flux, self.Ws)
        wave, flux = self.Util.resample(wave, flux, self.step)
        return wave, flux

    def get_idx_from_pdx(self, pdx_i):
        mask = True
        for ii, p in enumerate(pdx_i):
            mask = mask & (self.pdx0[:,ii] == p)
        idx = np.where(mask)[0][0]
        return idx

    def build_rbf(self, flux):
        print(f"Building RBF on flux shape {flux.shape}")
        self.rbf = RBFInterpolator(self.pdx0, flux, kernel='gaussian', epsilon=0.5)

    def prepare(self):
        self.init_bnds(self.R)
        wave, flux = self.prepro(self.wave0, self.flux0)
        self.wave = wave
        self.flux = flux
        self.getSky()
        self.build_rbf(flux)
        self.NLflux = self.Util.normlog_flux(flux)
        self.pca(self.NLflux)

    def get_idx_from_pdx(self, pdx_i):
        mask = True
        for ii, p in enumerate(pdx_i):
            mask = mask & (self.pdx0[:,ii] == p)
        idx = np.where(mask)[0][0]
        return idx


    def get_fdx_from_pmt(self, pmt):
        mask = True
        for ii, p in enumerate(pmt):
            mask = mask & (self.para[:,ii] == p)
        try:
            idx = np.where(mask)[0][0]
            return idx
        except:
            raise("No such pmt")


    def pca(self, flux, top=10):
        _,s,v = np.linalg.svd(flux, full_matrices=False)
        self.eigv0 = v
        if top is not None: 
            v = v[:top]
            s=s[:top]
        print(s[:10].round(2))
        assert abs(np.mean(np.sum(v.dot(v.T), axis=0)) -1) < 1e-5
        assert abs(np.sum(v, axis=1).mean()) < 0.1
        self.eigv = v
        return v


    def init_para(self, para):
        return pd.DataFrame(data=para, columns=C.pshort)

    def load_para(self):
        dfpara = pd.read_csv(self.GRID_DIR + "para.csv")
        return dfpara

    def get_flux_in_Prange(self, R=None, para=None, fix_CA=False):
        if para is None: 
            dfpara = self.load_para()
        else:
            dfpara = self.init_para(para)
        if R is None: 
            R = self.R
            
        Fs, Ts, Ls,_,_ = C.dRs[R]
        if fix_CA:
            dfpara = dfpara[(dfpara["A"] == 0.0)]
            # dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["A"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        maskM = (dfpara["M"] >= Fs[0]) & (dfpara["M"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["G"] >= Ls[0]) & (dfpara["G"] <= Ls[1]) 
        mask = maskM & maskT & maskL
        self.dfpara = dfpara[mask]
        self.para = np.array(self.dfpara.values)
        return self.dfpara.index

    def save_dataset(self, wave, flux, para, SAVE_PATH=None):
        if SAVE_PATH is None: SAVE_PATH = os.path.join(self.GRID_DIR, f"bosz_{self.Res}_{self.RR}.h5")
        with h5py.File(SAVE_PATH, "w") as f:
            f.create_dataset(f"flux", data=flux, shape=flux.shape)
            f.create_dataset(f"para", data=para, shape=para.shape)
            f.create_dataset(f"wave", data=wave, shape=wave.shape)  


    def getSky(self):
        self.Obs.initSky()
        self.Obs.getSky(self.wave0, self.step)

    
    def getModel(self, pmt, normlog=True):
        rbf_pdx = self.scaler(pmt)
        flux = self.rbf([rbf_pdx])[0]
        if normlog: flux = self.Util.normlog_flux_i(flux)
        return flux

    #likelihood ---------------------------------------------------------------------------------
    def getLogLik_pmt(self, temp_pmt, obsflux, obsvar, sky_mask0=None, nu_only=True):
        tempflux_in_res = self.getModel(temp_pmt, normlog=self.normlog)
        if self.normlog:
            tempflux_in_res = np.exp(tempflux_in_res)
            obsflux = np.exp(obsflux)

        return self.Obs.getLogLik(tempflux_in_res, obsflux, obsvar, nu_only=nu_only)   

    def get_LLH_fn(self, pdx, temp_pmt, obsflux, obsvar, sky_mask0=None):
        pmt = np.copy(temp_pmt)
        def fn(x, nu_only=True):
            pmt[pdx] = x
            return self.getLogLik_pmt(pmt, obsflux, obsvar, 
                                    sky_mask0=sky_mask0, nu_only=nu_only)
        return fn

    def make_obs_from_pmt(self, pmt, noise_level, normlog=1):
        flux = self.getModel(pmt, normlog=False)
        obsflux, obsvar = self.Obs.add_obs_to_flux(flux, noise_level)
        if normlog:  obsflux = self.Util.normlog_flux_i(obsflux)
        return obsflux, obsvar



    def getGridModel(self, pmt, normlog=True):
        flux_idx = self.get_fdx_from_pmt(pmt)
        flux = self.flux[flux_idx]
        if normlog: flux = self.Util.normlog_flux_i(flux)
        return flux




    
    def eval_pmt_on_axis(self, temp_pmt, x, obsflux, obsvar, axis="T", sky_mask0=None, plot=1):
        pdx = C.pshort.index(axis)
        name = self.Util.getname(*temp_pmt)
        print(f"Fitting with Template {name}")
        fn = self.get_LLH_fn(pdx, temp_pmt, obsflux, obsvar, sky_mask0=sky_mask0)
        self.fn = fn
        X = self.Obs.estimate(fn, x0=self.bnds[pdx][4], bnds=None)
        print("estimate", X)
        if plot: 
            SN = self.Util.getSN(obsflux)
            sigz2 = 0
            self.plot_pmt_on_axis(fn, x, X, SN, sigz2, pdx)
        return X

    def plot_pmt_on_axis(self, fn, x, X, SN, sigz2, pdx):
        x_large = np.linspace(self.bnds[pdx][0], self.bnds[pdx][1], 101)
        x_small = np.linspace(x - self.bnds[pdx][3], x + self.bnds[pdx][3], 25)
        y1 = []
        y2 = []
        for xi in x_large:
            y1.append(-1 * fn(xi))
        for xj in x_small:
            y2.append(-1 * fn(xj))

        MLE_x = -1 * fn(x)
        MLE_X = -1 * fn(X)
        plt.figure(figsize=(15,6), facecolor="w")
        plt.plot(x_large, y1,'g.-',markersize=7, label = f"llh")    
        plt.plot(x, MLE_x, 'ro', label=f"Truth {MLE_x:.2f}")
        plt.plot(X, MLE_X, 'ko', label=f"Estimate {MLE_X:.2f}")
        xname = C.Pnms[pdx]
        ts = f'{xname} Truth={x:.2f}K, {xname} Estimate={X:.2f}K, S/N={SN:3.1f}'
        # ts = ts +  'sigz={:6.4f} km/s,  '.format(np.sqrt(sigz2))
        plt.title(ts)
        plt.xlabel(f"{xname}")
        plt.ylabel("Log likelihood")
        plt.grid()
        plt.ylim((min(y1),min(y1)+(max(y1)-min(y1))*1.5))
        plt.legend()
        ax = plt.gca()
        ins = ax.inset_axes([0.1,0.45,0.4,0.5])
        ins.plot(x_small,y2,'g.-',markersize=7)
        ins.plot(x, MLE_x, 'ro')
        ins.plot(X, MLE_X, 'ko')
        ins.grid()
        
    def RBF_generator(self, n_sample, noise_level, pvals=None):
        if pvals is None: pvals = self.get_rand_pmt(n_sample)
        fluxs = np.zeros((n_sample, self.wave.shape[0]))
        obsfluxs = np.zeros((n_sample, self.wave.shape[0]))
        obsvars = np.zeros((n_sample, self.wave.shape[0]))
        for ii, pval in enumerate(pvals):
            flux = self.getModel(pval, normlog=False)
            obsflux, obsvars[ii] = self.Obs.add_obs_to_flux(flux, noise_level)
            obsfluxs[ii] = self.Util.normlog_flux_i(obsflux)
            fluxs[ii] = flux
        return pvals, fluxs, obsfluxs, obsvars

    def gen_noise_from_flux(self, fluxs, noise_level):
        n_sample = fluxs.shape[0]
        obsfluxs = np.zeros_like(fluxs)
        for ii, flux in enumerate(fluxs):
            obsflux, _ = self.Obs.add_obs_to_flux(flux, noise_level)
            obsfluxs[ii] = self.Util.normlog_flux_i(obsflux)
        return obsfluxs

    def get_rand_pmt(self, n_sample):
        pvals = np.zeros((n_sample, 5))
        for ii, bnd in enumerate(self.bnds):
            lb, ub = bnd[0], bnd[1]
            print(lb, ub)
            pvals[:,ii] = np.random.uniform(lb, ub, n_sample) 
        return pvals 

    