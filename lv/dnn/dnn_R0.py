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

from lv.dnn.baseDNN import BaseDNN


class DNN_R0(BaseDNN):
    def __init__(self, R0, top=100, pdx=[0,1,2], N_test=1000, step=20):
        super().__init__()
        self.pdx=pdx
        self.npdx=len(pdx)
        self.top=top
        self.N_test=N_test
        self.Wnms = ["RML"]
        self.R0=R0
        self.step=20
        # self.Wnms = ["BL","RML","NL"]

        self.n_ftr = top * len(self.Wnms)

        self.PC = None


    def get_PC_R(self, R, W="RedM", N=1000, step=20):
        PC_PATH=f"/scratch/ceph/swei20/data/dnn/PC/logPC/bosz_{W}_R5000_step{step}.h5"
        with h5py.File(PC_PATH, 'r') as f:
            PC = f[f"PC_{R}"][:]
        return PC

    def init(self):
        self.PC = self.get_PC_R(self.R0, W="RedM", N=1000, step=self.step)

    def prepare_dataset_R_W(self, R, W, N_train, N_test, step=None):
        self.wave, x_train, err_train, pval, dfsnr_train = self.load_R_dataset(R, W=W, N=N, step=step)



