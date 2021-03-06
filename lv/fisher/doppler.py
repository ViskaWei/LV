import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

from lv.util.util import Util

class Doppler(object):
    def __init__(self, wave_mask,step):
        self.wave_mask = wave_mask
        self.step = step
        self.sky_mask0 = None
        self.wave = None

# RV --------------------------------------------------
    def get_LLH_fn(self, tempflux_m, obsflux_m0, obsvar_m0, sky_mask0=None):
        def fn(x, nu_only=True):
            return self.getLogLik_rv(x, tempflux_m, obsflux_m0, obsvar_m0, 
                                    sky_mask0=sky_mask0, nu_only=nu_only)
        return fn
        
    def getRV(self, fn):
        rv0 = self.guessRV(fn)
        out = sp.optimize.minimize(fn, rv0, method="Nelder-Mead")

        if (out.success==True):
            RV = out.x[0]
        else:
            RV = np.nan
        return RV

    def guessRV(self, fn):
        v0  = np.linspace(-500, 500, 31)
        y0 = []
        for rv_i in v0:
            y0.append(-1 * fn(rv_i, nu_only=True))
        pp = [200,0,100,0.5*(y0[0]+y0[-1])]
        bb=((0,-300,100,min(y0)),(5000,300,300,min(y0)+4*(max(y0)-min(y0))))
        pp, _ = curve_fit(Doppler.lorentz, v0, y0, pp, bounds=bb)
        return pp[1]

    def getLogLik_rv(self, rv, tempflux, obsflux_m0, obsvar_m0, sky_mask0=None, nu_only=True):
        tempflux_m = self.getModel(tempflux, rv, step=self.step)
        tempflux_m0   = tempflux_m[self.wave_mask]
        if sky_mask0 is not None: 
            tempflux_m0[sky_mask0] = 0.0
        return Doppler.getLogLik(tempflux_m0, obsflux_m0, obsvar_m0, nu_only=nu_only)

#fisher --------------------------------------------------
    def getFisherMatrix(self, rv, fn):
        #---------------------------
        # compute the Fisher matrix
        #---------------------------
        nu0, phi0, chi0 = fn(rv    , nu_only=False)
        num, phim, chim = fn(rv - 1, nu_only=False)
        nup, phip, chip = fn(rv + 1, nu_only=False)
        f11 = chi0
        f12 = 0.5*(phip - phim)
        f22 = - nu0 * (nup + num - 2 * nu0) + f12 ** 2
        F = [[f11, f12],[f12, f22]]
        return F

    def getFisher1(self, rv, flux, obsflux_m0, obsvar_m0, drv=1):
        #---------------------------
        # compute the Fisher matrix
        #---------------------------
        m0 = self.getModel(flux, rv    , step=self.step)
        m2 = self.getModel(flux, rv + drv, step=self.step)
        m1 = self.getModel(flux, rv - drv, step=self.step)
        #---------------------------------
        # clip to the spectrograph range
        #---------------------------------
        t0  = m0[self.wave_mask]
        #-----------------------------
        # get the centered difference
        #-----------------------------
        t1 = (m2[self.wave_mask] - m1[self.wave_mask]) / (2.0 * drv)
        ob = obsflux_m0
        vm = obsvar_m0 
        
        # ob = obsflux_m[self.wave_mask]
        # vm = obsvar_m [self.wave_mask]
        #----------------------------------
        # build the different terms
        #----------------------------------
        psi00 = np.sum(t0 * t0 / vm)
        psi01 = np.sum(t0 * t1 / vm)
        psi11 = np.sum(t1 * t1 / vm)
        phi   = np.sum(ob * t0 / vm)
        chi   = np.sum(t0 * t0 / vm)    
        a0    = phi / chi
        dpsi  = psi00 * psi11 - psi01 ** 2
        sigz2 = psi00 / a0 ** 2 / dpsi
        return sigz2
#sigma --------------------------------------------------
    # RVSim = collections.namedtuple('RVSim',['name','N','NT','rv','RV','SN','S2','X2'])

    def getSigmaSim(self, NL,NV,N,NT,ss,T):
        #--------------------------------------------
        # create simulations of the analysis
        # estimate sigma from the data, and 
        # return a normalized variate (rv-RV)/sig
        # NL:  noise level
        # NV:  number of random velocities
        # N :  number of independent realizations
        # NT:  index of template from the stencil
        # ss:  the high resolution spectrum
        # T :  the template array over the stencil
        # Returns:
        # S : list of simulation outputs, one item for each rv
        #--------------------------------------------
        rvarr  = 200 * (np.random.rand(NV)-0.5)
        temp  = T[NT]
        #
        S = []
        for rv in rvarr:
            ssm   = self.getModel(ss,rv)
            wwm   = temp.wwm
            varm  = Util.getVar(ssm, temp.skym)
            vmobs = NL**2 * varm

            RV = []
            SN = []
            A2 = []
            S2 = []
            X2 = []

            for n in range(N):
                ssobs = ssm + getNoise(vmobs)    
                sn  = getSN(ssobs) 
                rvo = getRV1(ssobs,vmobs,temp)
                F2  = getFisherMatrix2(rvo,ssobs,vmobs,temp)
                iF2 = sp.linalg.inv(F2)
                sa2 = iF2[0,0]
                sg2 = iF2[1,1]

                RV.append(rvo)
                SN.append(sn)
                A2.append(sa2)
                S2.append(sg2)
                X2.append((rvo-rv)/np.sqrt(sg2))

            sim = RVSim(name,N,NT,rv,np.array(RV)-rv,SN,S2,X2)
            S.append(sim)

        return S
#static --------------------------------------------------
    @staticmethod
    def getModel(sconv, rv, step=5):
        #-----------------------------------------------------
        # Generate a spectrum shifted by rv. sconv is a high rez
        # spectrum already convolved with the LSF, rv is the 
        # radial velocity in km/s. Here we convolve once for speed, 
        # then apply different shifts and resample.
        #-----------------------------------------------------
        ss1 = Util.shiftSpec(sconv,rv)
        ss1 = Util.resampleFlux_i(ss1, step=step)    
        return ss1

    @staticmethod
    def makeObs(sconv, sky_m, rv, noise_level, step=5):
        flux_m = Doppler.getModel(sconv, rv, step=step)
        var_m = Util.getVar(flux_m, sky_m)
        noise = Util.getNoise(var_m)
        obsflux_m = flux_m + noise_level * noise
        obsvar_m = var_m * noise_level**2
        return flux_m, obsflux_m, obsvar_m

    @staticmethod
    def getLogLik(model, obsflux, var, nu_only=True):
        phi = np.sum(obsflux * model / var)
        chi = np.sum(model * model / var)
        nu  = phi / np.sqrt(chi)    
        if nu_only: 
            return -nu
        else:
            return nu, phi, chi

    @staticmethod
    def lorentz(x, a,b,c,d):
        return a/(1+(x-b)**2/c**2) + d
