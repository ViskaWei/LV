import numpy as np
import scipy as sp
from .util import Util

class Obs(object):
    def __init__(self):
        self.DATADIR = '/home/swei20/LV/data/fisher/'
        self.sky = None


    def initSky(self):
        sky = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        sky[:, 0] = 10 * sky[:, 0]
        self.sky0 = sky

    def getSky(self, wave, step):
        self.sky_in_res = Util.resampleSky(self.sky0, wave, step)

    def add_obs_to_flux(self, flux_in_res, noise_level):
        var_in_res = Util.getVar(flux_in_res, self.sky_in_res)
        noise = Util.getNoise(var_in_res)
        obsflux_in_res = flux_in_res + noise_level * noise
        obsvar_in_res = var_in_res * noise_level**2
        return obsflux_in_res, obsvar_in_res


    def get_obsflux_N(self, flux_in_res, var_in_res, noise_level, N):
        fluxs = np.tile(flux_in_res, (N, 1)) 
        noise = Util.get_noise_N(var_in_res, N)
        obsfluxs = fluxs + noise_level * noise
        return obsfluxs

    def add_obs_to_flux_N(self, flux_in_res, noise_level, N):
        var_in_res = Util.getVar(flux_in_res, self.sky_in_res)
        obsvar_in_res = var_in_res * noise_level**2
        obsflux_in_res = self.get_obsflux_N(flux_in_res, var_in_res, noise_level, N)
        return obsflux_in_res, obsvar_in_res

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
        # print(f"x0 = {x0}")
        # print(f"bnds = {bnds}")
        out = sp.optimize.minimize(fn, x0, bounds = bnds, method="Nelder-Mead")
        if (out.success==True):
            X = out.x[0]
        else:
            X = np.nan
        return X

    def guessEstimation(self, fn):
        pass