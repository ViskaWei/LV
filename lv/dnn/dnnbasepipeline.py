import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .dnn import DNN 
from lv.constants import Constants as c
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle


class DNNBasePipeline(object):
    def __init__(self):
        self.dnn = None
        self.x_train = None
        self.y_train = None
        self.y_train_org = None
        self.x_test = None
        self.y_test = None
        self.y_test_org = None
        self.y_pred = None
        self.dy_pred = None
        self.pdx = None
        self.pRange = None
        self.pMin = None
        self.pMax = None
        self.PC = None
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
        self.pdx = None
        self.wave = None
        self.Ps = c.Ps
        self.resolution = 1000

############################################ DATA #######################################

    def prepare_data(self, R, W, N_train=10000, N_test=1000, top=20, pdx=[0,1,2]):
        self.Ws = c.dWs[W]
        self.Rs = c.dRs[R]
        self.R = c.dR[R]
        self.top = top
        self.pRange, self.pMin, self.pMax = self.get_scaler(self.Rs, pdx=pdx)
        self.load_PC_from_R(self.R, top=top)

        self.x_train, self.y_train, self.y_train_org = self.prepare_RBF_data(N_train)  
        self.x_test, self.y_test, self.y_test_org = self.prepare_RBF_data(N_test)

    def prepare_RBF_data(self,N, R=None):
        wave, flux, pval = self.load_RBF_data(N, R=R)
        x = flux.dot(self.PC.T)
        y_org = pval[:,self.pdx]
        y = self.scale(y_org)
        return x, y, y_org

    def load_RBF_data(self, N, R=None):
        if R is None: 
            R = self.R
        else:
            R = c.dR[R]
        nn= N // 1000
        DATA_PATH = f"/scratch/ceph/swei20/data/dnn/{R}/rbf_R{self.Ws[2]}_{nn}k.h5"
        with h5py.File(DATA_PATH, 'r') as f:
            wave = f['wave'][()]
            flux = f['normflux'][()]
            pval = f['pval'][()]
        print(wave.shape, flux.shape, pval.shape)
        wave, flux = self.get_flux_in_Wrange(wave, flux, self.Ws)   
        return wave, flux, pval

    def get_scaler(self, Rs0, pdx=None):
        l = len(Rs0)
        Rs = np.array(Rs0).T
        if pdx is not None: 
            self.pdx = pdx
            Rs = Rs[:,pdx]
        else:
            self.pdx = np.arange(l)
        return np.diff(Rs, axis=0)[0], Rs[0], Rs[1]


    def scale(self, pval):
        pnorm = (pval - self.pMin) / self.pRange        
        return pnorm

    def rescale(self, pnorm):
        pval = pnorm * self.pRange + self.pMin
        return pval

    def load_PC_from_R(self, R, top=20):
        PC_PATH = f"/scratch/ceph/swei20/data/dnn/pc/bosz_{self.Ws[3]}_R{self.Ws[2]}.h5"
        with h5py.File(PC_PATH, 'r') as f:
            PC = f[f'PC_{R[0]}'][()]
        self.PC = PC[:top]
        print(self.PC.shape)
        
    def get_flux_in_Wrange(self, wave, flux, Ws=None):
        if Ws is None: Ws = self.Ws
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return wave[start:end], flux[:, start:end]

############################################ DNN ##########################################
    def prepare_DNN(self, lr=0.01, dp=0.0):
        self.dnn = DNN()
        self.dnn.set_model_shape(self.top, len(self.pdx))
        self.dnn.set_model_param(lr=lr, dp=dp, mtype="PCA", loss='mse', opt='adam', name='')
        self.dnn.build_model()


    def prepare(self, R, W, N_train=10000, lr=0.01):
        self.prepare_data(R, W, N_train=N_train)
        self.prepare_DNN(lr=lr)

    def run(self, ep=1, verbose=0):
        self.dnn.fit(self.x_train, self.y_train, top=self.top, ep=ep, verbose=verbose)
        self.y_pred = self.predict(self.x_test)
        self.dy_pred = self.y_pred - self.y_test_org
        self.plot_pred()

    def predict(self, data):
        return self.rescale(self.dnn.model.predict(data))

    def plot_pred(self, R=None):
        l = len(self.pdx)
        f, axs = plt.subplots(2, l,figsize=(16,12), facecolor="w")
        for ii in range(l):
            axs[0][ii].scatter(self.y_test_org[:,ii], self.y_pred[:,ii], c="k",s=1, label=f"{self.R}")
            if R is not None:
                axs[0][ii].scatter(self.YOs[R][:,ii], self.YPs[R][:,ii], c="b",s=1, label=f"{R}")

            axs[0][ii].annotate(f"{self.R}-NN\n{self.Ps[ii]}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=20)
            # axs[1][ii].plot(np.array([[self.pMin[ii],self.pMin[ii]], [self.pMax[ii]],self.pMax[ii]]), c="r")

            axs[0][ii].legend()
            axs[1][ii].scatter(self.y_test_org[:,ii], self.dy_pred[:,ii], c="k",s=1, label=f"{self.R}")
            if R is not None:
                axs[1][ii].scatter(self.YOs[R][:,ii], self.dYPs[R][:,ii], c="b",s=1, label=f"{R}")

            axs[1][ii].annotate(f"{self.R}-NN\n{self.Ps[ii]}", xy=(0.6,0.2), xycoords="axes fraction", fontsize=20)
            axs[1][ii].axhline(0, c='r')
            axs[1][ii].legend()
        axs[0][0].set_ylabel(f"pred")
        axs[1][0].set_ylabel(f"$\Delta$pred")

    def eval(self):
        pass


    def run_test_data(self, R, N_test=10000):
        self.Xs[R], self.Ys[R], self.YOs[R] = self.prepare_RBF_data(N_test, R=R)
        self.YPs[R] = self.predict(self.Xs[R])
        self.dYPs[R] = self.YPs[R] - self.YOs[R]
        self.pRanges[R], self.pMins[R], self.pMaxs[R] = self.get_scaler(c.dRs[R], pdx=self.pdx)

        self.plot_pred(R=R)
        self.plot_pspace(R=R)
        


    def plot_pspace(self,R=None, auto=0):
        f, axs = plt.subplots(1,3,figsize=(16,6), facecolor="w")
        for i in range(len(self.pdx)):
            j = 0 if i + 1 == 3 else i+1
            # axs[i].autoscale(1)

            axs[i].add_patch(Rectangle((self.pMin[i],self.pMin[j]),(self.pRange[i]),(self.pRange[j]),edgecolor="r",lw=2, facecolor="none"))

            axs[i].scatter(self.y_pred[:,i],self.y_pred[:,j],s=1, c="k")
            axs[i].set_xlim(self.pMin[i]-2*self.pRange[i], self.pMax[i]+2*self.pRange[i])
            axs[i].set_ylim(self.pMin[j]-2*self.pRange[j], self.pMax[j]+2*self.pRange[j])

            # axs[i].autoscale(auto)

            if R is not None:
                axs[i].scatter(self.YPs[R][:,i],self.YPs[R][:,j],s=1,c="b")
                axs[i].add_patch(Rectangle((self.pMins[R][i],self.pMins[R][j]),(self.pRanges[R][i]),(self.pRanges[R][j]),edgecolor="g",lw=2, facecolor="none"))

                # axs[i].add_patch(Rectangle((self.pMin[i],self.pMin[j]),(self.pRange[i]),(self.pRange[j]),edgecolor="r",lw=1, facecolor="none"))




