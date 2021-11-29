import os
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from lv.util.util import Util
from lv.grid.rbfloader import RbfLoader
from .fisher import Fisher



class PC(Fisher):
    def __init__(self, RR="BHB", step=0, W="Rtest", Res=5000, topk=20):
        super().__init__(W, Res)
        self.step = step
        self.RBF = None

        self.rbf_wave_mask = None
        self.wave_mask     = None
        self.wave_in_res   = None
        self.wave          = None

        self.sky0       = None
        self.sky_in_res = None
        self.sky        = None

        self.eigv       = None
        self.topk       = topk

        self.test = {"pmt": np.array([-2.0, 8000, 2.5, 0.0, 0.25]), "noise_level": 0.1,
                    "obsflux": None, "obsvar": None}
        self.init(RR)


# test--------------------------------------------------------------------------------
    def test_makeObs(self):
        obsflux, obsvar = self.makeObs(self.test["pmt"], self.test["noise_level"], plot=1)
        self.test["obsflux"] = obsflux
        self.test["obsvar"] = obsvar
        return obsflux, obsvar

    def test_eval_pmt_on_axis(self, test_X = None, x=None, axis="T"):
        pdx = self.params.index(axis)
        if x is None: 
            if self.test["obsflux"] is None: 
                obsflux, obsvar = self.test_makeObs()
            else:
                obsflux = self.test["obsflux"]
                obsvar = self.test["obsvar"]
            x = self.test["pmt"][pdx]
        else:
            truth_pmt = np.copy(self.test["pmt"])
            truth_pmt[pdx] = x
            obsflux, obsvar = self.makeObs(truth_pmt, self.test["noise_level"])
        temp_pmt = np.copy(self.test["pmt"])

        if test_X is not None:
            temp_pmt[pdx] = test_X 
        X = self.eval_pmt_on_axis(temp_pmt, x, obsflux, obsvar, axis=axis)
        return X

# init --------------------------------------------------------------------------------------------------------------------

    def init(self, RR):
        self.initPara()
        self.initRBF(RR)
        self.initSky()

    def initRBF(self, RR):
        RBF = RbfLoader(RR)
        self.bnds = RBF.bnds
        self.initWave(RBF.wave)
        self.initPC(RBF.eigv)
        self.RBF = RBF

    def initWave(self, W):
        self.rbf_wave_mask = (W>=self.lb) & (W<=self.ub)
        wave_in_approx_rng = W[self.rbf_wave_mask]
        self.wave_in_res = self.Util.resampleWave(wave_in_approx_rng, step=self.step)
        self.wave_mask = (self.wave_in_res>=self.Ws[0]) & (self.wave_in_res<=self.Ws[1])
        self.wave = self.wave_in_res[self.wave_mask]
    
    def initPC(self, rbf_eigv):
        # rbf_wave_mask_in_rng = (self.RBF.wave >= self.Ws[0]) & (self.RBF.wave <= self.Ws[1])
        self.eigv = rbf_eigv


    def initSky(self):
        super().initSky()
        self.sky_in_res = self.Util.resampleSky(self.sky0, self.wave_in_res, step=self.step)
        self.sky =self.sky_in_res[self.wave_mask]
        
# flux -------------------------------------------------------------------------

    def getModel_coeff(self, coeff):
        # coeff = self.RBF.get_coeff_from_pmt(pmt) 
        # flux0 = np.exp(self.eigv.dot(coeff))
        # flux_in_approx_rng = flux[self.wave_mask]
        flux_in_res = self.Util.resampleFlux_i(flux_in_approx_rng, step=step)
        return flux_in_res

    def getModel(self, pmt, step=0):
        flux0= self.RBF.get_flux_from_pmt(pmt)
        flux_in_approx_rng = flux0[self.rbf_wave_mask]
        flux_in_res = self.Util.resampleFlux_i(flux_in_approx_rng, step=step)
        return flux_in_res

    def makeFluxObs(self, flux_in_res, noise_level):
        var_in_res = Util.getVar(flux_in_res, self.sky_in_res)
        noise = Util.getNoise(var_in_res)
        obsflux_in_res = flux_in_res + noise_level * noise
        obsvar_in_res = var_in_res * noise_level**2
        return obsflux_in_res, obsvar_in_res
    
    def makeObs(self, pmt, noise_level, plot=1):
        flux_in_res = self.getModel(pmt, self.step)
        obsflux_in_res, obsvar_in_res = self.makeFluxObs(flux_in_res, noise_level)
        if plot: self.plotSpec(flux_in_res, obsflux_in_res, pmt)
        obsflux_in_rng = obsflux_in_res[self.wave_mask]
        obsvar_in_rng = obsvar_in_res[self.wave_mask]
        return obsflux_in_rng, obsvar_in_rng

    def plotSpec(self, flux_in_res, obsflux_in_res, pmt0):
        plt.figure(figsize=(9,3), facecolor='w')
        SN = self.Util.getSN(obsflux_in_res)
        plt.plot(self.wave_in_res, obsflux_in_res, lw=0.2, label=f"SNR={SN:.1f}")
        plt.plot(self.wave_in_res, flux_in_res)
        if pmt0 is None: 
            name = self.name
        else:
            name = self.Util.getname(*pmt0)
        plt.title(f"{name}")
        plt.legend()
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux [erg/s/cm2/A]")

    
#likelihood ---------------------------------------------------------------------------------
    def getLogLik_pmt(self, temp_pmt, obsflux, obsvar, sky_mask0=None, nu_only=True):
        tempflux_in_res = self.getModel(temp_pmt, step=self.step)
        tempflux_in_rng = tempflux_in_res[self.wave_mask]
        if sky_mask0 is not None: 
            tempflux_in_rng[sky_mask0] = 0.0
        return self.getLogLik(tempflux_in_rng, obsflux, obsvar, nu_only=nu_only)   

    def get_LLH_fn(self, pdx, temp_pmt, obsflux, obsvar, sky_mask0=None):
        pmt = np.copy(temp_pmt)
        def fn(x, nu_only=True):
            pmt[pdx] = x
            return self.getLogLik_pmt(pmt, obsflux, obsvar, 
                                    sky_mask0=sky_mask0, nu_only=nu_only)
        return fn

    def guessEstimation(self, fn):
        super().guessEstimation(fn)
        return 0

    def eval_pmt_on_axis(self, temp_pmt, x, obsflux, obsvar, axis="T", sky_mask0=None, plot=1):
        pdx = self.params.index(axis)
        name = self.Util.getname(*temp_pmt)
        print(f"Fitting with Template {name}")
        fn = self.get_LLH_fn(pdx, temp_pmt, obsflux, obsvar, sky_mask0=sky_mask0)
        X = self.estimate(fn, x0=self.bnds[pdx][2], bnds=self.bnds[pdx][:2])
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
        plt.figure(figsize=(15,6))
        plt.plot(x_large, y1,'g.-',markersize=7, label = f"llh")    
        plt.plot(x, MLE_x, 'ro', label=f"Truth {MLE_x:.2f}")
        plt.plot(X, MLE_X, 'ko', label=f"Estimate {MLE_X:.2f}")
        xname = self.pnames[pdx]
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
        