import time
# import fbpca
import logging
import numpy as np
import cupy as cp
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lv.pcp.pcpc import pcp_cupy




class DataLoader(object):
    def __init__(self):
        ################################ Flux Wave ###############################
        self.Ws = {"L": [3800, 5000], "M": [5000, 8000],
                   "H": [8000, 13000]}
        self.Ts = {"L": [4000, 6500], "H": [6500, 30000],
                   "T": [8000, 9000], "F": [4000, 30000]}
        self.flux = None
        self.wave = None
        self.mean = None
        self.size = None
        self.center = False
        self.prod = None 
        self.nf = None
        self.nw = None

        self.cmap="YlGnBu"

    
################################ Flux Wave #####################################
    def prepare_data(self, flux, wave, para, T, W, fix_CO=False):
        #cpu only
        flux       = self.get_flux_in_Prange(flux, para, self.Ts[T], fix_CO=fix_CO)
        flux, wave = self.get_flux_in_Wrange(flux, wave, self.Ws[W])

        #gpu only
        self.nwave = wave        
        flux = cp.asarray(flux, dtype=cp.float32)
        wave = cp.asarray(wave, dtype=cp.float16)
        self.flux = cp.clip(-flux, 0.0, None)
        self.wave = wave
        self.size = self.flux.shape
        self.nf, self.nw = self.size
        print(f"Cupy flux: {self.nf}, wave: {self.nw}")
    
    def get_flux_in_Prange(self, flux, para, Ts, fix_CO=True):
        dfpara = self.init_para(para)
        if fix_CO:
            dfpara = dfpara[(dfpara["O"] == 0.0)]
            # dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["O"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        self.dfpara = dfpara[(dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1])]
        return flux[self.dfpara.index]

    def get_flux_in_Wrange(self, flux, wave, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]

    def init_para(self, para):
        return pd.DataFrame(data=para, columns=["F","T","L","C","O"])

    def resampleSpec(self, flux, step):
        c = cp.cumsum(flux,axis=1)
        b = list(range(1,flux.shape[1],step))
        db = cp.diff(c[:,b],axis=1)
        dd = (db/step)
        return dd

    def resampleWave(self, wave,step):
        w = cp.cumsum(cp.log(wave))
        b = list(range(1,wave.shape[0],step))
        db = cp.diff(w[b])
        dd = (db/step)
        return cp.exp(dd)


####################################### SVD #######################################

    def _svd(self, X):
        return cp.linalg.svd(X, full_matrices=0)

    def get_eigv(self, X, top=5):
        _,_,v = self._svd(X)
        return v[:top] 

    def plot_eigv(self, v, top=5, mask=None, step=0.3, mask_c="r", name=None, ax=None):
        ax = ax or plt.gca()
        nv = cp.asnumpy(v[:top])
        for i in range(min(len(nv),top)):
            ax.plot(self.nwave, nv[i] + step*(i+1))
        ax.set_ylabel(f"Top {top} Eigvs of {name}")
        ax.set_xlim(self.nwave[0]-1, self.nwave[-1]+2)
        ax.xaxis.grid(True)
        if mask is not None: self.plot_mask(mask, ax=ax, c=mask_c)


####################################### Mask #######################################

    def trim_mask(self, mask0, niter=2):
        mask = cp.copy(mask0)
        for i in range(niter):
            mask[1:-1] = (mask[1:-1] & mask[:-2]) & mask[2:]
        mask[0] = False
        mask[-1] = False
        return mask

    def plot_mask(self, mask, ymax=0.3, c='r', ax=None):
        ax = ax or plt.gca()
        ax.vlines(self.nwave[mask], ymin=0, ymax=ymax, color=c)

    def plot_eroded_mask(self, mask0):
        f, axs = plt.subplots(2,1,figsize=(16,2), sharex=True)
        self.plot_mask(cp.asnumpy(mask0), ax=axs[0],ymax=1, c='r')
        self.plot_mask(self.nmask, ax=axs[1], ymax=1, c='r')

    def get_mask_from_v(self, v, k=5, q=0.8):
        vv = cp.sum(cp.abs(v[:k]), axis=0)
        cut = cp.quantile(vv, q)
        mask = vv > cut
        return mask

####################################### M N #######################################

    def get_major(self, v=None, k=5, q=0.6, niter=2):
        if v is None: _,_,v = self._svd(self.flux)
        mask0 = self.get_mask_from_v(v, k=k, q=q)    
        self.mask = self.trim_mask(mask0, niter=niter)   
        self.nmask = cp.asnumpy(self.mask) 
        return mask0
    
    def get_MN(self, mask, top=5):
        M = cp.zeros_like(self.flux)
        N = cp.zeros_like(self.flux)
        M[:,  mask] = self.flux[:,  mask]
        N[:, ~mask] = self.flux[:, ~mask]    
        self.M = M
        self.N = N    
        self.Mv =self.get_eigv(M, top=top)
        self.Nv =self.get_eigv(N, top=top)

    def plot_MN(self, step=0.3, axs=None):
        if axs is None:
            axs = plt.subplots(2,1,figsize=(16,10))[1]
        self.plot_eigv(self.Mv, mask=self.nmask, name="M", step=step, ax=axs[0])
        self.plot_eigv(self.Nv, mask = ~self.nmask, name="N", mask_c="lightgreen", step=step, ax=axs[1])

####################################### PCP #######################################
    def _pcp(self, X, delta=1e-6, mu=None, lam=None, norm=None, maxiter=50):
        XL, XS, (_,_,XLv) = pcp_cupy(X, delta=delta, mu=mu, lam=lam, norm=norm, maxiter=maxiter)
        XSv = self.get_eigv(XS, top = 5)
        return XL, XS, XLv, XSv


####################################### M LS #######################################
    


    # def get_flux_stats(self, flux=None):
    #     flux = flux or self.flux
    #     # return cp.max(cp.sum(cp.abs(self.flux), axis=1)), cp.sum(self.flux**2)**0.5
    #     l1_inf = cp.max(cp.abs(self.flux))
    #     l2 = cp.sum(self.flux ** 2) ** 0.5
    #     print(f"l1_inf: {l1_inf}, l2: {l2}")
    #     if self.center:
    #         self.GCs[self.g] = [l1_inf, l2]
    #     else:
    #         self.Gs[self.g] = [l1_inf, l2]


    def plot_pcp(self, M, L, S, u, s, v, cmap="hot"):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(cp.abs(M), aspect="auto", cmap=cmap,)
        plt.title("Original Flux")
        plt.subplot(2, 2, 2)
        plt.imshow(cp.abs(L), aspect="auto", cmap=cmap)
        plt.title("Low rank matrix")
        plt.subplot(2, 2, 3)
        plt.imshow(cp.abs(S), cmap=cmap, aspect="auto", )
        plt.title("Sparse matrix")
        plt.subplot(2, 2, 4)
        for i in range(min(len(v),5)):
            plt.plot(self.wave, v[i] + 0.3*(1 + i))
        plt.plot(self.nwave, cp.mean(cp.abs(S), axis=0), c="k")
        # plt.imshow(cp.dot(u, cp.dot(cp.diag(s), v)), cmap="gray")
        plt.title("L & S")
        plt.show()

   
