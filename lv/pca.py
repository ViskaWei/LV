import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from matplotlib.colors import LogNorm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# NORM_PATH = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/norm/spectra.h5'

class PCA(object):
    def __init__(self, flux=None, mask=None, n_trnc=40, n_pc=5, Wcut=False):
        self.flux = flux
        self.mask = mask
        self.X = {}
        self.S = {}
        self.C = {}
        self.V = {}
        self.U = {}
        self.n_trnc = n_trnc
        self.n_pc = n_pc
        self.wave = None
        self.wvln = None
        self.vT = None
        self.Ts = [[3500, 7000], [7000, 12000], [12000, 30000]]
        # self.Fs = 
        # self.Ws = [[8000, 13000]]
        self.Ws = [[8000, 13000]]
        self.Wcut= Wcut
        self.n_spec = {}
        
        self.df_para = None
        self.df_vs = {}
        self.n_eval = 5
        self.Tname = ["LOW", "MID", "HIGH"]


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
        self.get_all_para()
        self.get_pcs()

    def get_df_vs(self, ii):
        ts = self.Ts[ii]
        df_para = self.df_para
        df_paraT = df_para[(df_para["Teff"] > ts[0])& (df_para["Teff"] <= ts[1])]
        assert len(df_paraT) == self.n_spec[ii]
        df_paraT.index = range(df_paraT.shape[0])
        for jj in range(self.n_eval):
            V = self.V[ii][:, jj]
            df_paraT[f'p{jj}'] = V
        self.df_vs[ii] = df_paraT
            

    def get_all_para(self):
        df_para_all = pd.read_csv("/home/swei20/AE/data/para.csv")
        self.df_para = df_para_all[['FeH', 'Teff', 'Logg', 'C_M', 'O_M']]


    def get_flux_in_Wrange(self, Ws):
        start = np.digitize(Ws[0], self.wave)
        end = np.digitize(Ws[1], self.wave)
        self.flux = self.flux[..., start:end]
        self.wave = self.wave[start:end]
        # self.wvln = len(self.wave)


    def get_flux_in_Trange(self, Ts):
        start = np.digitize(Ts[0], self.vT)
        end = np.digitize(Ts[1], self.vT)
        flux_T = self.flux[:, start:end, ...]
        mask_T = self.mask[:, start:end, ...]
        flux_mat = flux_T[mask_T]
        return flux_mat

    def get_pcs(self):
        if self.Wcut:
            self.get_flux_in_Wrange(self.Ws[0])
        self.wvln = len(self.wave)
        for ii, ts in tqdm(enumerate(self.Ts)):
            x = self.get_flux_in_Trange(ts)
            c = x.T.dot(x)
            assert c.shape == (self.wvln, self.wvln)            

            self.n_spec[ii] = x.shape[0]
            self.X[ii] = x
            self.C[ii] =c

            self.get_pca(ii)
            self.get_df_vs(ii)


    def get_pca(self, ii):
        svd = TruncatedSVD(n_components=self.n_trnc)
        svd.fit(self.C[ii])
        self.S[ii] = svd.singular_values_            # shape: (truncate,)
        self.U[ii] = svd.components_.transpose() [:, :self.n_pc] # shape: (wvln, n_pc)]
        self.V[ii] = self.X[ii].dot(self.U[ii]) # shape: (n_spec, n_pc)
        
            
    def plot_us(self, ii, step=0.01, n_eigv=5, ax=None):
        ax = ax or plt.gca()
        for n in range(n_eigv):
            ax.plot(self.wave, self.U[ii][:, n]**2 + n*step)
            ax.plot([self.wave[0], self.wave[-1]], [n*step, n*step], 'k:')
            ax.set_ylabel(f"T {self.Ts[ii][0]}K - {self.Ts[ii][1]}K | # {self.X[ii].shape}")

    def plot_all_us(self, step=0.01, ub=None, lb=None, n_eigv=5):
        n_region = len(self.Ts)
        f, axs = plt.subplots(1, n_region, figsize=(18, n_eigv*4), sharey="row", facecolor='w', edgecolor='k')
        for ii, ax in enumerate(axs):
            self.plot_us(ii, step, n_eigv, ax)  
            ax.set_xlim([lb,ub] or [self.wave[0], self.wave[-1]])  
            ax.set_ylim([0, n_eigv*step])
        
    
    def plot_df_vs(self, ii):
        sns.pairplot(
            self.df_vs[ii],
            x_vars=[f"p{i}" for i in range(self.n_eval)],
            y_vars=[f"p{i}" for i in range(self.n_eval)],
            hue="FeH",
            plot_kws=dict(marker="o", s=2, edgecolor="none"),
            diag_kws=dict(fill=False),
            palette="Spectral",
            corner=True
        )

            
    # def plot_ 
    #     plt.figure(figsize=(10,10))
    #     plt.scatter(U[:,0], U[:,1], c=df_para0["Logg"])
    #     plt.grid()
    #     plt.colorbar()