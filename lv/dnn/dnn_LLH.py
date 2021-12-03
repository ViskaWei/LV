import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lv.dnn.baseDNN import BaseDNN
from .dnn import DNN 
from tqdm import tqdm

from lv.util.constants import Constants as C
from lv.util.util import Util
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib.patches import Rectangle, Ellipse, Patch
# import matplotlib.transforms as transforms
# from matplotlib.lines import Line2D
# from matplotlib import collections  as mc
from scipy.stats import chi2 as chi2


from .baseDNN import BaseDNN


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class DNN_LLH(BaseDNN):
    def __init__(self):
        super().__init__()
        self.dataDir = os.path.join(C.PFSSPEC_DIR, "train/pfs_stellar_model/dataset")
        self.x_trains = {}
        self.y_trains = {}
        self.p_trains= {}
        self.gpu=3
        self.pdx=[0,1,2,3,4]
        self.Res= 5000
        self.top=10
        self.N_train = 100000
        self.N_test = 1000
        self.dnn = {}

    def prepare(self,R0, NL):
        self.setup_scalers(R0)
        self.init_gpu(gpu=self.gpu)
        self.transform = self.pcloader(R0)

        x_train, y_train = self.dataloader(R0, self.N_train, NL)
        self.prepare_trainset(R0, x_train, y_train)
        x_test, y_test = self.dataloader(R0, self.N_test, NL)
        self.prepare_testset(R0, x_test, y_test)

        # self.

    def pcloader(self, R0):
        PATH ="/datascope/subaru/user/swei20/data/pfsspec/train/pfs_stellar_model/dataset/LLH/bosz_5000_PC.h5"
        with h5py.File(PATH, "r") as f:
            PC = f[f"PC_{R0}"][:]
        self.PC = PC[:self.top]
        def fn(x):
            return x.dot(self.PC.T)
        return fn

    


    def dataloader(self, R0, N, NL):
        nn = N // 1000
        RR = C.dRR[R0]
        PATH = os.path.join(C.TRAIN_DIR, "LLH", f"bosz_{self.Res}_{RR}_NL{NL}_{nn}k.h5")
        with h5py.File(PATH, "r") as f:
            # fluxs = f["fluxs"][:]
            pvals = f["pvals"][:]
            obsfluxs = f["obsfluxs"][:]
            # obsvars = f["obsvars"][:]
            if self.wave is not None: 
                self.wave = f["wave"][:]
        return obsfluxs, pvals

        
    def run_R0(self, R0, top = None, lr=0.01, dp=0.01, ep=1, verbose=0):
        self.R0 = R0
        dnn = self.prepare_DNN(input_dim=top, lr=lr, dp=dp)
        x_train = self.x_trains[R0][:, :top]
        dnn.fit(x_train, self.y_trains[R0], ep=ep, verbose=verbose)
        self.dnn[R0] = dnn
        p_preds_R0= {}
        x_test = self.x_tests[R0][:, :top]
        p_preds_R0[R0] = self._predict(x_test, R0, dnn=dnn)
        # for R, x_test in self.x_tests[R0].items():
        #     x_test = x_test[:, :top]
        #     p_preds_R0[R] = self.predict(x_test, R0, dnn=dnn)
        self.p_preds[R0] = p_preds_R0


    def prepare_trainset(self, R0, x_train, y_train):
        
        self.x_trains[R0] = self.transform(x_train)
        self.p_trains[R0] = y_train
        self.y_trains[R0] = self.scale(y_train, R0)

    
    def prepare_testset(self, R0, x_test, y_test):
        self.x_tests[R0] = self.transform(x_test)
        self.p_tests[R0] = y_test
        self.y_tests[R0] = self.scale(y_test, R0)

    def predict(self,flux):
        pcData = self.transform(flux)
        if len(pcData.shape) ==1: pcData = np.array([pcData])
        pPred = self._predict(pcData, self.R0, dnn=self.dnn[self.R0])
        return pPred
    



    # def prepare(self, Ws=None, Rs=None, N_train=None, grid=0, isNoisy=1):
    #     if Ws is None: Ws = self.arms
    #     self.load_PCs(Ws, Rs)
    #     self.setup_scalers()
    #     self.prepare_trainset(Ws, Rs, N_train, grid, isNoisy)
    #     self.prepare_testset(Ws, Rs, self.N_test, grid, isNoisy)
    #     self.init_gpu(gpu=self.gpu)


    def init_gpu(self, gpu=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


