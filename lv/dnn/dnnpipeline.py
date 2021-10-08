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


class DNNPipeline(object):
    def __init__(self, top=20, pdx=[0,1,2]):
        self.pdx=pdx
        self.dnn = None
        self.x_trains = {}
        self.y_trains = {}
        self.p_trains= {}
        self.f_tests = {}
        self.p_tests= {}
        self.x_tests = {}
        self.y_tests = {}
        self.y_test_orgs = {}
        self.y_pred = None
        self.dy_pred = None
        self.pdx = None
        self.pRanges = {}
        self.pMins = {}
        self.pMaxs = {}
        self.PCs = {}
        self.Wnms = ["BL","RML","NL"]
        self.Rnms = c.Rnms
        self.RRnms = c.RRnms
        self.Pnms = c.Pnms
        self.dWs = c.dWs
        self.dRs = c.dRs
        self.dR = c.dR
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

        self.init(top)

############################################ DATA #######################################
    def init(self, top, N_train=10000, N_test=1000):
        self.load_PCs(top=top)
        self.setup_scalers()
        self.load_test_fluxs(N=N_test)
        self.get_train_pcF_nP(N=N_train)
        self.get_test_pcF_nP()
        self.prepare_lbl()

    def load_PCs(self, top=None):
        self.top = top
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
        print(wave.shape, flux.shape, pval.shape)
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
            fluxs, self.p_trains[R0] = self.load_RBF_data(N, R=R0)
            self.x_trains[R0] = self.transform_R(fluxs, R0)
            self.y_trains[R0] = self.scale(self.p_trains[R0], R0)

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

    def rescale(self, pnorm):
        pval = pnorm * self.pRanges[R] + self.pMins[R]
        return pval

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

            axs[0][ii].annotate(f"{self.R}-NN\n{c.Pnms[ii]}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=20)
            # axs[1][ii].plot(np.array([[self.pMin[ii],self.pMin[ii]], [self.pMax[ii]],self.pMax[ii]]), c="r")

            axs[0][ii].legend()
            axs[1][ii].scatter(self.y_test_org[:,ii], self.dy_pred[:,ii], c="k",s=1, label=f"{self.R}")
            if R is not None:
                axs[1][ii].scatter(self.YOs[R][:,ii], self.dYPs[R][:,ii], c="b",s=1, label=f"{R}")

            axs[1][ii].annotate(f"{self.R}-NN\n{c.Pnms[ii]}", xy=(0.6,0.2), xycoords="axes fraction", fontsize=20)
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




