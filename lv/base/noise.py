import os
import sys
import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import h5py
import scipy as sp
from lv.constants import Constants as c
from lv.util import Util as u

class Noise():
    def __init__(self):

        self.dRs=c.dRs
        self.dRR=c.dRR
        self.dWw = {"RedM": "mr", "Blue": "b", "NIR": "n"}
        self.Pnms =c.Pnms
        self.PLs = {}

    def init_para(self, para):
        self.dfpara = pd.DataFrame(para, columns = self.Pnms)

    def get_flux_from_param():
        pass

    def get_idx_from_param(self, Ps):
        dfp = self.dfpara
        for ii, P in enumerate(Pnms):
            dfp = dfp[dfp[P]==Ps[ii]]
        return dfp.index
        