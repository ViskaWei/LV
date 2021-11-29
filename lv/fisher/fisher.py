import numpy as np
import scipy as sp
import pandas as pd

import os
import collections

import matplotlib.pyplot as plt
from lv.base.specloader import getSpectrum
from lv.util.util import Util
from .doppler import Doppler    

class Fisher(object):
    def __init__(self, W="RedM", Res=50000):
        self.Res = Res
        self.dSteps = {"RedM":5, "RL":200, "Rtest":0}
        self.dWs = {"RedM":[7100,8850], "RL":[7100,8850], "Rtest":[7100,8850]}
        self.DATADIR = '../data/fisher/'

        self.Util=Util()
        self.Doppler= None

        self.lb = 6250
        self.ub = 9750

        self.step=self.dSteps[W]
        self.Ws = self.dWs[W]
        self.Template = collections.namedtuple('Template',['name','sst','wwm','ssm','skym','pmt','iwm'])



#init ---------------------------------------------------------------------------------

    def initPara(self):
        dfpara=pd.read_csv(self.DATADIR +"para.csv")
        self.MHs = dfpara["FeH"].values
        self.TEs = dfpara["Teff"].values
        self.LGs = dfpara["Logg"].values
        self.CHs = dfpara["C_M"].values
        self.AHs = dfpara["O_M"].values
        self.uM  = dfpara["FeH"].unique()
        self.uT  = dfpara["Teff"].unique()
        self.uL  = dfpara["Logg"].unique()
        self.uC  = dfpara["C_M"].unique()
        self.uA  = dfpara["O_M"].unique()
        self.params = ["M", "T", "L", "C", "A"]
        self.pnames = ["[M/H]", "Teff","logG", "Carbon","Alpha"]
        
    def initSky(self):
        sky = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        sky[:, 0] = 10 * sky[:, 0]
        self.sky0 = sky

#Get Spectrum---------------------------------------------------------------------------------
    def getSpectrum(self, MH, TE, LG, CH=0.0, AH=0.0, R=50000, lb=6250, ub=9750):
            # first check if the values are a valid grid location
    #------------------------------------------------------
        if (~self.isValidGrid(MH, TE, LG, CH, AH)):
            print('Parameters are not on the grid')
            return np.zeros((1,3))
        else:
            spec = getSpectrum(MH, TE, LG, CH, AH, R, lb, ub)
            return spec

    def isValidGrid(self, MH, TE, LG, CH, AH):
        #----------------------------------------------------------
        # determine if the parameters are at a valid grid point
        #----------------------------------------------------------
        ix = (self.MHs==MH) & (self.CHs==CH) & (self.AHs==AH) & (self.TEs==TE) & (self.LGs==LG)
        return ix.any()

#Get Template---------------------------------------------------------------------------------

    def makeTemplate_spec(self, spec, pmt):
        #-----------------------------------------------
        # get the spectrum (n) and build the template
        #-----------------------------------------------
        flux  = spec[:,1]
        flux  = self.Util.convolveSpec(flux)
        flux_m  = self.Util.resampleFlux_i(flux, step=self.step)
        assert abs(self.wave0 - spec[:,0]).sum() < 1e-6            
        name = self.Util.getname(*pmt)
        temp = self.Template(name, flux, self.wave_m, flux_m, self.sky_m, pmt, self.wave_mask)
        return temp

    def testTemplate(self,pmt = [-2.0,8000,2.5,0.0,0.25], axis=["T","L"]):
        nearby_pmt=self.get_nearby_grid_nd(pmt,axis=axis, step=1)
        specs = []
        for p in nearby_pmt:
            spec = self.getSpectrum(*p, R=self.Res, lb=self.lb, ub=self.ub)
            specs.append(spec)
        self.specs = specs
        temps = []
        for ii, spec in enumerate(specs):
            p = nearby_pmt[ii]
            temp = self.makeTemplate_spec(spec, p)
            temps.append(temp)
        self.temps = temps
        self.pmts = nearby_pmt
        self.pmt = pmt
        self.flux = self.Util.convolveSpec(self.specs[0][:,1])
        self.name=self.Util.getname(*pmt)

# likelihood---------------------------------------------------------------------------------
    @staticmethod
    def getLogLik(model, obsflux, var, nu_only=True):
        phi = np.sum(np.divide(np.multiply(obsflux, model), var))
        chi = np.sum(np.divide(np.multiply(model  , model), var))
        nu  = phi / np.sqrt(chi)    
        if nu_only: 
            return -nu
        else:
            return nu, phi, chi

    @staticmethod
    def lorentz(x, a,b,c,d):
        return a/(1+(x-b)**2/c**2) + d

    def estimate(self, fn, x0=None, bnds=None):
        if x0 is None: x0 = self.guessEstimation(fn)
        out = sp.optimize.minimize(fn, x0, bounds = bnds, method="Nelder-Mead")
        if (out.success==True):
            X = out.x[0]
        else:
            X = np.nan
        return X

    def guessEstimation(self, fn):
        pass

#Get Grid---------------------------------------------------------------------------------
    def get_nearby_grid_1d(self, pmt, axis="T", step=1, out=[]):
        pdx = self.params.index(axis)
        x = pmt[pdx]
        uX = eval(f'self.u{axis}')
        iX = np.where(uX==x)[0][0]
        for step_i in range(1, step+1):
            if iX >= step_i:
                p1=pmt.copy()
                p1[pdx] = uX[iX-step_i]
                if (self.isValidGrid(*p1)): out.append(p1)
            if iX + step_i < len(uX):
                p2=pmt.copy()
                p2[pdx] = uX[iX+step_i]
                if (self.isValidGrid(*p2)): out.append(p2)
            # print(step_i, uX[iX-step_i], p1)
        return out

    def get_nearby_grid_nd(self, pmt, axis=["T","L"], step=1):
        #TODO: check if work for nd
        nearby_pmts = self.get_nearby_grid_1d(pmt, axis[0], step, out=[pmt])
        outs=[]
        for axis_i in axis[1:]:
            for nearby_pmt in nearby_pmts:
                out = self.get_nearby_grid_1d(nearby_pmt, axis=axis_i, step=step, out=[nearby_pmt])
                outs = outs + out
            nearby_pmts = outs
        return nearby_pmts

#Get RV---------------------------------------------------------------------------------
    def makeTempObs_rv(self, rv, flux_h, noise_level, pmt0=None, plot=0):
        flux_m, obsflux_m, obsvar_m = self.Doppler.makeObs(flux_h, self.sky_m, rv, 
                                                            noise_level, step=self.step)
        if plot: self.plotSpec(flux_m, obsflux_m, rv, pmt0)
        return obsflux_m[self.wave_mask], obsvar_m[self.wave_mask]

    def testOneRV1(self, flux_h, temp, rv, noise_level, pmt0=None, sky_mask0=None, plot=1):
        obsflux_m0, obsvar_m0 = self.makeTempObs_rv(rv, flux_h, noise_level, pmt0=pmt0, plot=plot)
        RV, F = self.evalRV(temp, obsflux_m0, obsvar_m0, rv, sky_mask0=sky_mask0, plot=plot)
        return RV, F

    def evalRV(self, temp, obsflux_m0, obsvar_m0, rv, sky_mask0 = None, plot=1):
        tempflux = temp.sst
        SN = self.Util.getSN(obsflux_m0)
        if sky_mask0 is not None:
            obsflux_m0[sky_mask0] = 0.0
            # obsvar_m0[sky_mask0]  = 10 * obsvar_m0[sky_mask0]
        print(f"Fitting with Template {temp.name}")
        fn = self.Doppler.get_LLH_fn(tempflux, obsflux_m0, obsvar_m0, sky_mask0=sky_mask0)
        RV = self.Doppler.getRV(fn)  
        if np.isnan(RV): 
            print('getRV error in '+ temp.name)
        else:
            error = np.abs(RV-rv) / rv *100
            print(f"RV err={error:.02f}%")
        F  = self.Doppler.getFisherMatrix(RV,fn)
        det   = F[0][0]*F[1][1]-F[1][0]**2
        print(f'sigma_z={np.sqrt(F[0][0]/det):.5f}')
        if plot:
            sigz2 = self.Doppler.getFisher1(rv, tempflux, obsflux_m0, obsvar_m0)
            self.plotRV(fn, rv, RV, SN, sigz2)
        return RV, F

    def getSkyMask(self, ratio=0.8):
        sky_cut = np.quantile(self.sky_m0, ratio)
        sky_mask0 = self.sky_m0 > sky_cut
        return sky_mask0

#Sigma---------------------------------------------------------------------------------


#plot---------------------------------------------------------------------------------
    

    def plotRV(self, fn, rv, RV, SN, sigz2):
        rv_large  = np.linspace(-300  , 300   , 101)
        rv_small  = np.linspace(rv - 6, rv + 6, 25)
        
        y1 = []
        y2 = []
        for rv_i in rv_large:
            y1.append(-1 * fn(rv_i))
        for rv_j in rv_small:
            y2.append(-1 * fn(rv_j))

        
        MLE_rv = -1 * fn(rv)
        MLE_RV = -1 * fn(RV)

        plt.figure(figsize=(15,6))
        plt.plot(rv_large, y1,'g.-',markersize=7, label = "llh")    
        plt.plot(rv, MLE_rv, 'ro', label=f"rv {MLE_rv:.2f}")
        plt.plot(RV, MLE_RV, 'ko', label=f"RV{MLE_RV:.2f}")
        ts = 'rv={:6.4f} km/s,  '.format(rv)+ 'RV={:6.4f} km/s,  '.format(RV)
        ts = ts + 'S/N={:3.1f},  '.format(SN) + 'sigz={:6.4f} km/s,  '.format(np.sqrt(sigz2))
        plt.title(ts)
        plt.xlabel("rv [km/s]")
        plt.ylabel("Log likelihood")
        plt.grid()
        plt.ylim((min(y1),min(y1)+(max(y1)-min(y1))*1.5))
        plt.legend()
        ax = plt.gca()
        ins = ax.inset_axes([0.1,0.45,0.4,0.5])
        ins.plot(rv_small,y2,'g.-',markersize=7)
        ins.plot(rv, MLE_rv, 'ro')
        ins.plot(RV, MLE_RV, 'ko')
        ins.grid()
        
        # plt.savefig('figs/F-rvfit.png');
        # plt.show()
