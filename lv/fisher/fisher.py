import numpy as np
import scipy as sp
import pandas as pd

import os
import collections
import copy
import time
import urllib
import bz2
import h5py

import matplotlib.pyplot as plt
from scipy import stats
from numpy import linalg
from scipy.optimize import curve_fit
from lv import flux
from lv.base.specloader import getSpectrum
from lv.util import Util
from .doppler import Doppler    

class Fisher(object):
    def __init__(self, W="RedM", Res=50000):
        self.Res = Res
        self.dSteps = {"RedM":5}
        self.dWs = {"RedM":[7100,8850]}
        self.DATADIR = '../data/fisher/'

        self.Util=Util()
        self.Doppler= None
        self.wave   = None
        self.wave_m = None
        self.wave_m0 = None
        self.sky    = None
        self.sky_m  = None
        self.sky_m0 = None
        self.lb = 6250
        self.ub = 9750

        self.step=self.dSteps[W]
        self.Ws = self.dWs[W]
        self.Template = collections.namedtuple('Template',['name','sst','wwm','ssm','skym','pmt','iwm'])


#init ---------------------------------------------------------------------------------

    def init(self):
        self.initPara()
        self.initDoppler()
        self.initSky()


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
        
    def initDoppler(self):
        spec = self.getSpectrum(-2.0, 8000, 2.5, R=self.Res)
        self.wave = spec[:,0]
        self.wave_m = self.Util.resampleWave(self.wave, step=self.step)
        self.wave_mask = (self.wave_m>=self.Ws[0]) & (self.wave_m<=self.Ws[1])
        self.wave_m0 = self.wave_m[self.wave_mask]
        self.Doppler= Doppler(self.wave_mask, self.step)
    
    def initSky(self):
        sky = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        sky[:, 0] = 10 * sky[:, 0]
        self.sky = sky
        self.sky_m = self.Util.resampleSky(self.sky, self.wave, step=self.step)
        self.sky_m0 =self.sky_m[self.wave_mask]

#Get Spectrum---------------------------------------------------------------------------------
    def getSpectrum(self, MH, TE, LG, CH=0.0, AH=0.25, R=50000, lb=6250, ub=9750):
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
    def makeTemplate(self, m,t,g,c,a):
        #-----------------------------------------------
        # get the spectrum (n) and build the template
        #-----------------------------------------------
        name = self.Util.getname(m,t,g,c,a)
        pmt  = (m,t,g,c,a)
        o    = self.getSpectrum(m,t,g,c,a)
        ww   = o[:,0]
        sst  = o[:,1]
        sst  = self.Util.convolveSpec(sst)
        
        wwm  = self.Util.resampleWave(ww, step=self.step)
        ssm  = self.Util.resampleFlux_i(sst, step=self.step)
        skym = self.Util.resampleSky(self.sky, ww, step=self.step)
        iwm  = (wwm>=self.Ws[0]) & (wwm<=self.Ws[1])
        temp = self.Template(name,sst,wwm,ssm,skym,pmt,iwm)
        return temp

    def makeTemplate_spec(self, spec, pmt):
        #-----------------------------------------------
        # get the spectrum (n) and build the template
        #-----------------------------------------------
        flux  = spec[:,1]
        flux  = self.Util.convolveSpec(flux)
        flux_m  = self.Util.resampleFlux_i(flux, step=self.step)
        assert abs(self.wave - spec[:,0]).sum() < 1e-6            
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
    def makeTempObs(self, flux_h, rv, noise_level, pmt0=None, plot=0):
        flux_m, obsflux_m, obsvar_m = self.Doppler.makeObs(flux_h, self.sky_m, rv, 
                                                            noise_level, step=self.step)
        if plot: self.plotSpec(flux_m, obsflux_m, rv, pmt0)
        return obsflux_m[self.wave_mask], obsvar_m[self.wave_mask]

    def testOneRV1(self, flux_h, temp, rv, noise_level, pmt0=None, sky_mask0=None, plot=1):
        obsflux_m0, obsvar_m0 = self.makeTempObs(flux_h, rv, noise_level, pmt0=pmt0, plot=plot)
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
    
    def plotSpec(self, flux_m, obsflux_m, rv, pmt0=None):
        plt.figure(figsize=(9,3), facecolor='w')
        SN = self.Util.getSN(obsflux_m)
        plt.plot(self.wave_m, obsflux_m, lw=0.2, label=f"SNR={SN:.1f}")
        plt.plot(self.wave_m, flux_m, label=f"rv={rv:.1f}")
        if pmt0 is None: 
            name = self.name
        else:
            name = self.Util.getname(*pmt0)
        plt.title(f"{name}")
        plt.legend()
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux [erg/s/cm2/A]")

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
