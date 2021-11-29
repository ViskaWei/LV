import os
import numpy as np
import scipy as sp
from lv.grid.bosz import Bosz
from lv.grid.rbfgrid import RbfGrid
from lv.grid.modelgrid import ModelGrid

    
class RbfLoader(object):
    def __init__(self,  RR="BHB"):
        self.RBFDIR="/datascope/subaru/user/swei20/data/pfsspec/import/stellar/rbf/"
        self.RR = RR
        self.bnds = []

        self.init(RR)

    def init(self, RR):
        rbf = self.initRBF(RR)
        self.wave = rbf.wave
        self.eigv = rbf.grid.eigv['flux']
        self.bnds = self.initBnds(rbf)

        

    def initRBF(self, RR):
        RBF_PATH = os.path.join(self.RBFDIR, f"bosz_5000_{RR}/rbf/", 'spectra.h5')     

        rbf = ModelGrid(Bosz(pca=True, normalized=True), RbfGrid)
        rbf.preload_arrays = False
        rbf.load(RBF_PATH, format='h5')

        self.rbf = rbf
        return rbf


    def initBnds(self, rbf):
        bnds = []
        axes = rbf.grid.grid.get_axes()
        for k in axes:
            axes_val = axes[k].values
            print(k, axes[k].values.shape, axes_val)
            lb, ub = axes_val[0], axes_val[-1]
            bnds.append([lb, ub, 0.5*(lb+ub), np.abs(axes_val[1] - lb)])
        return bnds

    def get_coeff_from_pmt(self, pmt):
        coeff= self.rbf.grid.grid.get_value('flux', Fe_H=pmt[0], T_eff=pmt[1], log_g=pmt[2],C_M=pmt[3], O_M=pmt[4])
        return coeff

    def get_flux_from_pmt(self, pmt):
        spec = self.rbf.get_model(denormalize=True, Fe_H=pmt[0], T_eff=pmt[1], log_g=pmt[2],C_M=pmt[3], O_M=pmt[4])
        return spec.flux