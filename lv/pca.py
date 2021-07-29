import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from matplotlib.colors import LogNorm

# NORM_PATH = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/norm/spectra.h5'

class PCA(object):
    def __init__(self, flux=None, mask=None, n_trnc=40, n_pc=32):
        self.flux = flux
        self.mask = mask
        self.X = {}
        self.S = {}
        self.C = {}
        self.V = {}
        self.n_trnc = n_trnc
        self.n_pc = n_pc
        self.wave = None
        self.wvln = None
        self.vT = None
        self.Ts = [[3000, 10000], [7000, 15000], [12000, 35000]]

        self.init()

    def init(self, NORM_PATH=None):
        if NORM_PATH is None:
            NORM_PATH = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/norm/spectra.h5'
        with h5py.File(NORM_PATH, 'r') as f:
            self.vT = f['T_eff'][()]
            self.wave = f['wave'][()]
            if self.flux is None:
                self.flux = f['flux'][()]
                self.mask = f['flux_idx'][()]
        self.wvln = len(self.wave)
        self.get_vs()


    def get_flux_in_Trange(self, Ts):
        start = np.digitize(Ts[0], self.vT)
        end = np.digitize(Ts[1], self.vT)
        flux_T = self.flux[:, start:end, ...]
        mask_T = self.mask[:, start:end, ...]
        flux_mat = flux_T[mask_T]
        return flux_mat

    def get_vs(self):
        for ii, ts in tqdm(enumerate(self.Ts)):
            x = self.get_flux_in_Trange(ts)
            c = x.T.dot(x)
            assert c.shape == (self.wvln, self.wvln)            
            self.X[ii] = x
            self.C[ii] =c

            self.get_pca(ii)


    def get_pca(self, ii):
        svd = TruncatedSVD(n_components=self.n_trnc)
        svd.fit(self.C[ii])
        self.S[ii] = svd.singular_values_            # shape: (truncate,)
        self.V[ii] = svd.components_.transpose() [:, :self.n_pc] # shape: (wvln, n_pc)]
            
    def plot_vs(self, ii, step=0.01, ax=None):
        ax = ax or plt.gca()
        for n in range(12):
            ax.plot(self.wave, self.V[ii][:, n]**2 + n*step)
            ax.plot([self.wave[0], self.wave[-1]], [n*step, n*step], 'k:')
            ax.set_ylabel(f"T {self.Ts[ii][0]}K - {self.Ts[ii][0]}K | # {self.X[ii].shape}")

    def plot_all_vs(self, step=0.01):
        n_region = len(self.Ts)
        f, axs = plt.subplots(1, n_region, figsize=(18,10), sharey="row", facecolor='w', edgecolor='k')
        for ii, ax in enumerate(axs):
            self.plot_vs(ii, step, ax)    