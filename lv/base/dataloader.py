import time
# import fbpca
import logging
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm




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
    def prepare_data(self, flux, wave, T, W, fix_CO=True, para=None, cleaned=False, center=False, save=False):
        if cleaned: 
            self.flux, self.mean, self.wave = flux, flux.mean(), wave
        else:
            self.center = center
            self.flux, self.mean, self.wave = self.init_flux_wave(flux, wave, para, T, W, fix_CO=fix_CO)
        self.size = self.flux.shape
        self.nf, self.nw = self.size
        # self.mu = 0.25 * self.nf * self.nw 
        self.g = T + W + str(int(fix_CO))
        print(f"center {self.center} {self.g} flux: {self.nf}, wave: {self.nw}")
        self.save = save

    def init_flux_wave(self, flux, wave, para, T, W, fix_CO):

        flux, wave = self.get_flux_in_Wrange(flux, wave, self.Ws[W])
        flux       = self.get_flux_in_Prange(flux, para, self.Ts[T], fix_CO=fix_CO)
        flux = np.clip(-flux, 0.0, None)
        if self.center:
            mean = flux.mean(0)
            return flux - mean, mean, wave
        else:
            return flux, flux.mean(), wave

    def init_para(self, para):
        return pd.DataFrame(data=para, columns=["F","T","L","C","O"])

    # def downsample_flux(self, flux, wave, ds):
    #     return flux[:, ::ds], wave[::ds]

    def resampleSpec(self, flux, step):
        c = np.cumsum(flux,axis=1)
        b = list(range(1,flux.shape[1],step))
        db = np.diff(c[:,b],axis=1)
        dd = (db/step)
        return dd

    def resampleWave(self, wave,step):
        w = np.cumsum(np.log(wave))
        b = list(range(1,wave.shape[0],step))
        db = np.diff(w[b])
        dd = (db/step)
        return np.exp(dd)


    def get_flux_in_Wrange(self, flux, wave, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]

    def get_flux_in_Prange(self, flux, para, Ts, fix_CO=True):
        dfpara = self.init_para(para)
        if fix_CO:
            dfpara = dfpara[(dfpara["O"] == 0.0)]
            # dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["O"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        self.dfpara = dfpara[(dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1])]
        return flux[self.dfpara.index]

    def get_flux_stats(self, flux=None):
        flux = flux or self.flux
        # return np.max(np.sum(np.abs(self.flux), axis=1)), np.sum(self.flux**2)**0.5
        l1_inf = np.max(np.abs(self.flux))
        l2 = np.sum(self.flux ** 2) ** 0.5
        print(f"l1_inf: {l1_inf}, l2: {l2}")
        if self.center:
            self.GCs[self.g] = [l1_inf, l2]
        else:
            self.Gs[self.g] = [l1_inf, l2]


    def _svd(self, X, rank=40, tol=1e-2):
        u, s, v = svds(X, k=rank, tol=tol)
        return  u[:,::-1], s[::-1], v[::-1, :]

    def plot_pcp(self, M, L, S, u, s, v, cmap="hot"):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(M), aspect="auto", cmap=cmap,)
        plt.title("Original Flux")
        plt.subplot(2, 2, 2)
        plt.imshow(np.abs(L), aspect="auto", cmap=cmap)
        plt.title("Low rank matrix")
        plt.subplot(2, 2, 3)
        plt.imshow(np.abs(S), cmap=cmap, aspect="auto", )
        plt.title("Sparse matrix")
        plt.subplot(2, 2, 4)
        for i in range(min(len(v),5)):
            plt.plot(self.wave, v[i] + 0.3*(1 + i))
        plt.plot(self.wave, np.mean(np.abs(S), axis=0), c="k")
        # plt.imshow(np.dot(u, np.dot(np.diag(s), v)), cmap="gray")
        plt.title("L & S")
        plt.show()

    def plot_mask(self, mask, ax=None, ymax=0.3, c='r'):
        ax = ax or plt.gca()
        ax.vlines(self.wave[mask], ymin=0, ymax=ymax, color=c)

    def plot_eigv(self, v, mask=None, gap=0.3, ax=None, mask_c="r"):
        ax = ax or plt.gca()

        for i in range(min(len(v),5)):
            ax.plot(self.wave, v[i] + gap*(i+1))
        if mask is not None: self.plot_mask(mask, ax=ax, c=mask_c)