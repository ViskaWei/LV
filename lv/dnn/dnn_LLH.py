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

from lv.util.constants import Constants as c
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
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class DNN_LLH(BaseDNN):
    def __init__(self):
        self.dataDir = os.path.join(c.PFSSPEC_DIR, "train/pfs_stellar_model/dataset")
        self.x_trains = {}
        self.y_trains = {}
        self.p_trains= {}

    def prepare_DNN(self, input_dim=None, lr=0.01, dp=0.0):
        dnn = DNN()
        if input_dim is None: input_dim = self.n_ftr
        dnn.set_model_shape(input_dim, len(self.pdx))
        dnn.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        dnn.build_model()
        return dnn

    def test(self, R0, R1, dnn=None):

        
    def run_R0(self, R0, top = None, lr=0.01, dp=0.01, ep=1, verbose=0):
        dnn = self.prepare_DNN(input_dim=top, lr=lr, dp=dp)
        x_train = self.x_trains[R0][:, :top]
        dnn.fit(x_train, self.y_trains[R0], ep=ep, verbose=verbose)
        p_preds_R0= {}
        for R, x_test in self.x_tests[R0].items():
            x_test = x_test[:, :top]
            p_preds_R0[R] = self.predict(x_test, R0, dnn=dnn)
        self.p_preds[R0] = p_preds_R0


    def prepare_trainset(self, R0, x_train, y_train):
        nsflux = self.add_noise(flux, err)
        self.x_trains[R0] = self.transform_R(nsflux, R0)
        self.y_trains[R0] = self.scale(pval, R0)
        self.p_trains[R0] = pval

    
    def prepare_testset(self, R0, N_test):
        for R1 in self.Rnms:
            self.prepare_testset_R0_R1(R0, R1, N_test)


    def prepare(self, Ws=None, Rs=None, N_train=None, grid=0, isNoisy=1):
        if Ws is None: Ws = self.arms
        self.load_PCs(Ws, Rs)
        self.setup_scalers()
        self.prepare_trainset(Ws, Rs, N_train, grid, isNoisy)
        self.prepare_testset(Ws, Rs, self.N_test, grid, isNoisy)
        self.init_gpu(gpu=self.gpu)


    def init_gpu(self, gpu=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


