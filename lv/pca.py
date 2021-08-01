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
    def __init__(self, flux=None, mask=None, n_trnc=5, n_pc=5, Wcut=False):
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

        e = 1e-3
        self.Ts = [[3500 - e, 5000], [7000, 12000], [12000, 30000]]
        self.Fs = [[-2.5 - e, -1.25], [-1.25, 0.], [0., 0.75]]
        self.Ls = [[0 - e, 2], [2, 4], [4, 5.0]]

        # self.Ws = [[8000, 13000]]
        self.Ws = [[8000, 13000]]
        self.Wcut= Wcut
        self.n_spec = {}
        
        self.df_para = None
        self.df_vs = {}
        self.n_eval = 5
        self.Tname = ["LOW", "MID", "HIGH"]
        self.cbars = {}
        self.cmap = "Spectral"


        self.init()
        self.init_plots()

    def init(self, NORM_PATH=None):
        if NORM_PATH is None:
            NORM_PATH = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/norm/spectra.h5'
        with h5py.File(NORM_PATH, 'r') as f:
            self.vT = f['T_eff'][()]
            self.wave = f['wave'][()]
            if self.flux is None:
                self.flux = f['flux'][()]
                self.mask = f['flux_idx'][()]
        self.load_para()
        self.get_pcs()

    def get_level(self, para):
        for ii, ts in enumerate(self.Ts):
            idx = df_para[(df_para[para] > ts[0])& (df_para[para] <= ts[1])].index
            df_para["L3"][idx] = self.Tname[ii]

    def get_df_vs(self, ii):
        ts = self.Ts[ii]
        df_para = self.df_para
        df_paraT = df_para[(df_para["Teff"] > ts[0])& (df_para["Teff"] <= ts[1])]
        assert len(df_paraT) == self.n_spec[ii]
        df_paraT.index = range(df_paraT.shape[0])
        for jj in range(self.n_eval):
            U = self.U[ii][:, jj]
            df_paraT[f'p{jj}'] = U
        self.df_vs[ii] = df_paraT
            

    def load_para(self):
        PARA_PATH = "/home/swei20/LV/data/p5T3.csv"
        self.df_para = pd.read_csv(PARA_PATH)
        # print(self.df_para)

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
        svd = TruncatedSVD(n_components=self.n_pc + self.n_trnc)
        svd.fit(self.C[ii])
        self.S[ii] = svd.singular_values_            # shape: (truncate,)
        self.V[ii] = svd.components_.transpose() [:, :self.n_pc] # shape: (wvln, n_pc)]
        self.U[ii] = self.X[ii].dot(self.V[ii]) # shape: (n_spec, n_pc)
        
            
    def plot_eigv_at_T(self, ii, step=0.01, n_eigv=5, ax=None):
        ax = ax or plt.gca()
        for n in range(n_eigv):
            ax.plot(self.wave, self.V[ii][:, n]**2 + n*step)
            ax.plot([self.wave[0], self.wave[-1]], [n*step, n*step], 'k:')
            ax.set_ylabel(f"T {int(self.Ts[ii][0])}K - {self.Ts[ii][1]}K | # {self.X[ii].shape}")

    def plot_eigv(self, step=0.01, ub=None, lb=None, n_eigv=None):
        n_eigv = n_eigv or self.n_eval
        n_region = len(self.Ts)
        f, axs = plt.subplots(1, n_region, figsize=(18, n_eigv*4), sharey="row", facecolor='w', edgecolor='k')
        for ii, ax in enumerate(axs):
            self.plot_eigv_at_T(ii, step, n_eigv, ax)  
            ax.set_xlim([lb,ub] or [self.wave[0], self.wave[-1]])  
            ax.set_ylim([0, n_eigv*step])
        
    
    def plot_df_vs(self, ii, c="FeH"):
        g = sns.pairplot(
            self.df_vs[ii],
            x_vars=[f"p{i}" for i in range(self.n_eval)],
            y_vars=[f"p{i}" for i in range(self.n_eval)],
            hue=c,
            plot_kws=dict(marker="o", s=2, edgecolor="none"),
            diag_kws=dict(fill=False),
            palette=self.cmap,
            corner=True
        )
        g.fig.text(0.5, 0.8, f"{self.Tname[ii]} T", ha ='center', fontsize = 15)

        # handles = g._legend_data.values()
        # labels = g._legend_data.keys()
        # g.fig.legend(handles=handles, labels=labels, loc='upper center',facecolor = 'w')


    def init_plots(self):
        self.get_cbars()
        step = 0.01 if self.Wcut else 0.0001
        self.plot_eigv(step=step, lb=8000, ub=10000, n_eigv=5)


    def plot_all(self):
        self.zoom_eigv(0, 0, c="b")
        self.zoom_eigv(0, 1, c = "orange")
        self.plot_loggs(0,1)
        self.plot_df_vs(0, c="FeH")
        self.plot_df_vs(0, c="Logg")


    def get_cbars(self):
        for ii, ts in enumerate(self.Ts):
            norm = plt.Normalize(*ts)
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])
            self.cbars[ii] = sm

        # cmap = sns.cubehelix_palette(8, start=0.5, rot=-.75, as_cmap=True)

    def plot_logg(self, ii, p1, p2):
        sns.set(rc={'figure.figsize':(20, 8)})
        fg=sns.FacetGrid(self.df_vs[ii], col="GG")
        fg.map_dataframe(sns.scatterplot, f"p{p1}", f"p{p2}", hue="Teff", palette=self.cmap, s=5)
        plt.colorbar(self.cbars[ii])

    def plot_loggs(self, p1, p2):
        for ii in range(3):
            self.plot_logg(ii, p1, p2)

    def zoom_eigv(self, ii, pdx, c=None):
        c = c or "k"
        bnds =[8000, 10000]
        s, e = np.digitize(bnds, self.wave) 
        plt.figure(figsize=(20,5))
        plt.plot(self.wave[s:e], self.V[ii][s:e,pdx], c=c, label = "eigv {pdx}")
        plt.ylabel(f"{self.Tname[ii]} T")
        plt.legend()
        plt.xlim(*bnds)
            
    # def plot_ 
    #     plt.figure(figsize=(10,10))
    #     plt.scatter(U[:,0], U[:,1], c=df_para0["Logg"])
    #     plt.grid()
    #     plt.colorbar()