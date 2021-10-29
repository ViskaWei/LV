import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .dnn import DNN 
from tqdm import tqdm
from lv.constants import Constants as c
from lv.dnn.baseDNN import BaseDNN

np.random.seed(922)

class DNN_ALL(BaseDNN):
    def __init__(self, arms=["BL"], grid=0, top=100, pdx=[0,1,2], N_test=1000, pc_name=""):
        super().__init__()
        self.pdx=pdx
        self.npdx=len(pdx)
        self.top=top
        self.N_test=N_test
        # self.Wnms = ["BL", "RML"]
        self.arms =arms
        
        self.grid=grid
        # self.Wnms = ["BL","RML","NL"]
        self.n_ftr = top * len(self.arms)
        self.pc_name = pc_name
        self.dPCs = {}


    def prepare(self, Ws=None, Rs=None, N_train=None, grid=0, isNoisy=1):
        if Ws is None: Ws = self.arms
        self.load_PCs(Ws, Rs)
        self.setup_scalers()
        self.prepare_trainset(Ws, Rs, N_train, grid, isNoisy)
        self.prepare_testset(Ws, Rs, self.N_test, grid, isNoisy)


    def load_PCs(self, Ws=None, Rs=None, top=None):
        for W in Ws:
            self.dPC[W], self.dPxl[W] = self.pcloader_W(W=W, Rs=Rs, top=top, name=self.pc_name)
        for R in self.dPC[W].keys():
            PCs = []
            for W in self.arms:
                PCs.append(self.dPC[W][R])
            self.dPCs[R] = np.vstack(PCs)

    
    def run_R0(self, R0, top = None, lr=0.01, dp=0.01, ep=1, verbose=0):
        dnn = self.prepare_DNN(input_dim=top, lr=lr, dp=dp)
        x_train = self.x_trains[R0][:, :top]
        dnn.fit(x_train, self.y_trains[R0], ep=ep, verbose=verbose)
        p_preds_R0= {}
        for R, x_test in self.x_tests[R0].items():
            x_test = x_test[:, :top]
            p_preds_R0[R] = self.predict(x_test, R0, dnn=dnn)
        self.p_preds[R0] = p_preds_R0
        self.dCT[R0] = self.get_overlap_R0(R0)
        self.dnns[R0] = dnn
            
    def run(self, lr=0.01, dp=0.01, ep=1, verbose=0, top=None):
        for R0 in self.x_trains.keys():
            self.run_R0(R0, top=top, lr=lr, dp=dp, ep=ep, verbose=verbose)
        # self.get_contamination_mat(plot=1)

