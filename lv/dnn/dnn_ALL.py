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


class DNN_ALL(BaseDNN):
    def __init__(self, top=100, pdx=[0,1,2], N_test=1000):
        super().__init__()
        self.pdx=pdx
        self.npdx=len(pdx)
        self.top=top
        self.N_test=N_test
        self.Wnms = ["RML"]
        # self.Wnms = ["BL","RML","NL"]
        self.n_ftr = top * len(self.Wnms)


    def prepare(self, N_train=10000):
        self.load_PCs(top=self.top)
        self.setup_scalers(self.pdx)
        for W in self.Wnms:
            self.prepare_trainset_W(W, N_train)
            self.prepare_testset_W(W, self.N_test)

    def load_PCs(self, top=None):
        for W in self.Wnms:
            self.dPC[W], self.dPxl[W] = self.load_PC_W(W=W, Rs=None, top=top)

    
    def run_R0(self, R0, lr=0.01, dp=0.01, ep=1, verbose=0):
        dnn = self.prepare_DNN(lr=lr, dp=dp)
        dnn.fit(self.x_trains[R0], self.y_trains[R0], ep=ep, verbose=verbose)
        p_preds_R0= {}
        for R, x_test in self.x_tests[R0].items():
            p_preds_R0[R] = self.predict(x_test, R0, dnn=dnn)
        self.p_preds[R0] = p_preds_R0
        self.dCT[R0] = self.get_contamination_R0(R0)
        self.dnns[R0] = dnn
            
    def run(self, lr=0.01, dp=0.01, ep=1, verbose=0):
        for R0 in self.Rnms:
            self.run_R0(R0, lr=lr, dp=dp, ep=ep, verbose=verbose)
        # self.get_contamination_mat(plot=1)

