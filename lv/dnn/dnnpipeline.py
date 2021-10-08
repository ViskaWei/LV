import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .dnn import DNN 
from tqdm import tqdm
from lv.constants import Constants as c
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)



class DNNPipeline(object):
    def __init__(self, top=20, pdx=[0,1,2]):
        self.pdx=pdx
        self.npdx=len(pdx)
        self.top=top
        self.n_ftr = top * self.npdx
        self.dnn = None
        self.x_trains = {}
        self.y_trains = {}
        # self.p_trains= {}
        self.f_tests = {}
        self.p_tests= {}
        self.x_tests = {}
        self.y_tests = {}
        self.p_preds = {}
        self.dp_preds = {}
        self.pdx = None
        self.pRanges = {}
        self.pMins = {}
        self.pMaxs = {}
        self.PCs = {}
        self.Wnms = ["BL","RML","NL"]
        self.Rnms = c.Rnms
        self.nR = len(self.Rnms) 
        self.RRnms = c.RRnms
        self.Pnms = c.Pnms
        self.dWs = c.dWs
        self.dRs = c.dRs
        self.dR = c.dR
        self.dRC = c.dRC
        self.Ws = None
        self.Rs = None
        


        self.Xs = {}
        self.Ys = {}
        self.YOs = {}
        self.YPs = {}
        self.dYPs = {}
        self.pRanges = {}
        self.pMins = {}
        self.pMaxs = {}
        self.W = None
        self.R = None
        self.Rg = None
        self.Rts = None
        self.pdx = pdx
        self.wave = None
        self.resolution = 1000

        self.init()

############################################ DATA #######################################
    def init(self, N_train=10000, N_test=1000):
        self.load_PCs(top=self.top)
        self.setup_scalers()
        self.load_test_fluxs(N=N_test)
        self.get_train_pcF_nP(N=N_train)
        self.get_test_pcF_nP()

    def load_PCs(self, top=None):
        for W in self.Wnms:
            Ws = self.dWs[W]
            PC_PATH = f"/scratch/ceph/swei20/data/dnn/pc/bosz_{Ws[3]}_R{Ws[2]}.h5"
            PC_W = {}
            with h5py.File(PC_PATH, 'r') as f:
                for R in self.Rnms:
                    PC = f[f'PC_{R}'][()]
                    PC_W[R] = PC[:top]
            self.PCs[W] = PC_W

    def load_RBF_data(self, N, R=None):
        nn= N // 1000
        DATA_PATH = f"/scratch/ceph/swei20/data/dnn/{self.dR[R]}/rbf_R{self.resolution}_{nn}k.h5"
        with h5py.File(DATA_PATH, 'r') as f:
            wave = f['wave'][()]
            flux = f['normflux'][()]
            pval = f['pval'][()]
        # print(wave.shape, flux.shape, pval.shape)
        fluxs = {}
        for W in self.Wnms:
            Ws = self.dWs[W]
            _, fluxs[W] = self.get_flux_in_Wrange(wave, flux, Ws)   
        return fluxs, pval[:, self.pdx]

    def get_flux_in_Wrange(self, wave, flux, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return wave[start:end], flux[:, start:end]

    def load_test_fluxs(self, N=1000):
        for R in self.Rnms:
            self.f_tests[R], self.p_tests[R] = self.load_RBF_data(N, R=R)

    def get_train_pcF_nP(self, N=10000):
        for R0 in self.Rnms:
            fluxs, pvals = self.load_RBF_data(N, R=R0)
            self.x_trains[R0] = self.transform_R(fluxs, R0)
            self.y_trains[R0] = self.scale(pvals,  R0)

    def get_test_pcF_nP(self):
        for R0 in self.Rnms:
            self.get_test_pcF_nP_R0(R0)

    def get_test_pcF_nP_R0(self, R0):
        x_tests_R0 = {}
        y_tests_R0 = {}
        for R, fluxs in self.f_tests.items():
            pvals = self.p_tests[R]
            x_tests_R0[R] = self.transform_R(fluxs, R0)
            y_tests_R0[R] = self.scale(pvals, R)

        self.x_tests[R0] = x_tests_R0
        self.y_tests[R0] = y_tests_R0

    def transform_R(self, fluxs, R):
        xs=[]
        for W in self.Wnms:
            Ws = self.dWs[W]
            flux = fluxs[W]
            PC_WR = self.PCs[W][R]
            xs.append(flux.dot(PC_WR.T))
        x = np.hstack(xs)
        return x

############################################ SCALER #######################################
    def setup_scalers(self):
        for R, Rs in self.dRs.items():
            self.pRanges[R], self.pMins[R], self.pMaxs[R] = self.get_scaler(Rs)

    def get_scaler(self, Rs0):
        Rs = np.array(Rs0).T
        Rs = Rs[:, self.pdx]
        return np.diff(Rs, axis=0)[0], Rs[0], Rs[1]

    def scale(self, pval, R):
        pnorm = (pval - self.pMins[R]) / self.pRanges[R]        
        return pnorm

    def rescale(self, pnorm, R):
        pval = pnorm * self.pRanges[R] + self.pMins[R]
        return pval

############################################ DNN ##########################################
    def prepare_DNN(self, lr=0.01, dp=0.0):
        dnn = DNN()
        dnn.set_model_shape(self.n_ftr, len(self.pdx))
        dnn.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        dnn.build_model()
        return dnn

    def run_R0_v1(self, R0, ep=1, verbose=0):
        dnn = self.prepare_DNN()
        dnn.fit(self.x_trains[R0], self.y_trains[R0], ep=ep, verbose=verbose)
        p_preds_R0= {}
        dp_preds_R0 = {}
        for R, x_test in self.x_tests[R0].items():
            p_preds_R0[R] = self.predict(dnn, x_test, R0)
            dp_preds_R0[R] = p_preds_R0[R] - self.p_tests[R]
        self.p_preds[R0] = p_preds_R0
        self.dp_preds[R0] = dp_preds_R0

    def run(self, ep, verbose=0):
        for R0 in tqdm(self.Rnms):
            self.run_R0(R0, ep, verbose)

    def run_R0(self, R0, ep=1, verbose=0):
        dnn = self.prepare_DNN()
        dnn.fit(self.x_trains[R0], self.y_trains[R0], ep=ep, verbose=verbose)
        p_preds_R0= {}
        for R, x_test in self.x_tests[R0].items():
            p_preds_R0[R] = self.predict(dnn, x_test, R0)
        self.p_preds[R0] = p_preds_R0

    def predict(self, dnn, data, R):
        y_preds = dnn.model.predict(data)
        return self.rescale(y_preds, R)

    def plot_box_R0_R1(self, R0, R1, n_box=2, axs=None, ylbl=1):
        if axs is None: axs = plt.subplots(1, self.npdx,  figsize=(16, 4), facecolor="w")[1]
        pRange, pMin, pMax, = self.pRanges[R0], self.pMins[R0], self.pMaxs[R0]
        for i, ax in enumerate(axs):
            j = 0 if i + 1 == 3 else i + 1

            ax.scatter(self.p_preds[R0][R1][:,i],self.p_preds[R0][R1][:,j],s=1, c=self.dRC[R1], label= f"{self.dR[R1]}")
            # ax.annotate(f"{self.dR[R0]}-NN", xy=(0.5,0.8), xycoords="axes fraction",fontsize=15, c=self.dRC[R0])
            handles, labels = ax.get_legend_handles_labels()

            ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor="r",lw=2, facecolor="none"))

            if R0 != R1:
                ax.scatter(self.p_preds[R0][R0][:,i],self.p_preds[R0][R0][:,j],s=1, c=self.dRC[R0], label= f"{self.dR[R0]}")
                ax.add_patch(Rectangle((self.pMins[R1][i],self.pMins[R1][j]),\
                    (self.pRanges[R1][i]),(self.pRanges[R1][j]),edgecolor="k",lw=2, facecolor="none"))
                legend_ele = [Line2D([0], [0], marker='o', color='w',label=f'{self.dR[R0]}_Pred', markerfacecolor=self.dRC[R0], markersize=10),
                            Line2D([0], [0], marker='o',color='w', label=f'{self.dR[R1]}_Pred', markerfacecolor=self.dRC[R1], markersize=10),
                            Patch(facecolor='none', edgecolor='r', label=f"{self.dR[R0]}-NN"), 
                            Patch(facecolor='none', edgecolor='k', label=f"{self.dR[R1]}")]
            else:
                legend_ele = [Line2D([0], [0], marker='o', color='w',label=f'{self.dR[R0]}_Pred', markerfacecolor=self.dRC[R0], markersize=10),
                                Patch(facecolor='none', edgecolor='r', label=f"{self.dR[R0]}-NN")] 

            ax.set_xlim(pMin[i]-n_box*pRange[i], pMax[i]+n_box*pRange[i])
            ax.set_ylim(pMin[j]-n_box*pRange[j], pMax[j]+n_box*pRange[j])
            ax.set_xlabel(self.Pnms[i])

            ax.legend(handles = legend_ele)
            if ylbl: ax.set_ylabel(self.Pnms[j])

    def plot_box_R0(self, R0, n_box=2,  axs=None):
        if axs is None: 
            f, axss = plt.subplots(self.npdx, self.nR,  figsize=(20, 4*self.npdx), sharey="row", sharex="row", facecolor="w")
        for i, axs in enumerate(axss.T):
            R1 = self.Rnms[i]
            self.plot_box_R0_R1(R0, R1, n_box, axs, ylbl = (i==0))
        plt.tight_layout()


    def plot_box(self):
        for rdx, R0 in enumerate(self.Rnms):
            self.plot_box_R0(R0, n_box=2, axs=None)

    # def plot_pred(self, R=None):
    #     l = len(self.pdx)
    #     f, axs = plt.subplots(2, l,figsize=(16,12), facecolor="w")
    #     for ii in range(l):
    #         axs[0][ii].scatter(self.p_tests[R0][:,ii], self.p_preds[R0][:,ii], c="k",s=1, label=f"{self.R}")
    #         if R is not None:
    #             axs[0][ii].scatter(self.YOs[R][:,ii], self.YPs[R][:,ii], c="b",s=1, label=f"{R}")

    #         axs[0][ii].annotate(f"{self.R}-NN\n{c.Pnms[ii]}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=20)
    #         # axs[1][ii].plot(np.array([[self.pMin[ii],self.pMin[ii]], [self.pMax[ii]],self.pMax[ii]]), c="r")

    #         axs[0][ii].legend()
    #         axs[1][ii].scatter(self.y_test_org[:,ii], self.dy_pred[:,ii], c="k",s=1, label=f"{self.R}")
    #         if R is not None:
    #             axs[1][ii].scatter(self.YOs[R][:,ii], self.dYPs[R][:,ii], c="b",s=1, label=f"{R}")

    #         axs[1][ii].annotate(f"{self.R}-NN\n{c.Pnms[ii]}", xy=(0.6,0.2), xycoords="axes fraction", fontsize=20)
    #         axs[1][ii].axhline(0, c='r')
    #         axs[1][ii].legend()
    #     axs[0][0].set_ylabel(f"pred")
    #     axs[1][0].set_ylabel(f"$\Delta$pred")



