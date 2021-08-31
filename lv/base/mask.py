import time
# import fbpca
import logging
import cupy as cp
import pandas as pd
# import cupy.linalg.svd as svd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def get_mask(v, ds, k=5, q=0.6, step=2):
    mask0 = get_mask_from_v(v, k=k, q=q)    
    mask = trim_mask(mask0, step=step)    
    f, axs = plt.subplots(1,2,figsize=(16,2))
    ds.plot_mask(cp.asnumpy(mask0), ax=axs[0])
    ds.plot_mask(cp.asnumpy(mask), ax=axs[1])
    return mask

def trim_mask(mask0, step=2):
    mask = cp.copy(mask0)
    for i in range(step):
        mask[1:-1] = (mask[1:-1] & mask[:-2]) & mask[2:]
    return mask

def get_mask_from_v(v, k=5, q=0.8):
    vv = cp.sum(cp.abs(v[:k]), axis=0)
    cut = cp.quantile(vv, q)
    mask = vv > cut
    return mask

def plot_L_eigv(ds, mask, Mc=None, L=None, vL=None, ax=None):
    if vL is None:
        if L is None:
            L = cp.zeros(Mc.shape)
            L[:,~mask] = Mc[:,~mask]
        _,wL,vL = cp.linalg.svd(L, full_matrices=0) 
    ax = ax or plt.subplots(figsize=(16,6))[1]
    ds.plot_eigv(cp.asnumpy(vL), cp.asnumpy(~mask), gap=0.5, mask_c="lightgreen", ax=ax)

def plot_S_eigv(ds, mask, Mc=None, S=None, vS=None, ax=None):
    if vS is None:
        if S is None:
            S = cp.zeros(Mc.shape)
            S[:,mask] = Mc[:,mask]
        _,wS,vS = cp.linalg.svd(S, full_matrices=0)
    ax = ax or plt.subplots(figsize=(16,6))[1]
    ds.plot_eigv(cp.asnumpy(vS), cp.asnumpy(mask), gap=0.5, mask_c="r", ax=ax)

def plot_SL(ds, mask, Mc):
    f, axs = plt.subplots(2,1,figsize=(16,12))
    plot_S_eigv(ds, mask, Mc, ax=axs[0])
    plot_L_eigv(ds, mask, Mc, ax=axs[1])