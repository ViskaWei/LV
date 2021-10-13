import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from lv.base.sample import resample


class Util():
    def __init__(self):
    
    
        pass
# sample ------------------------------------------------------------------------------

    @staticmethod
    def resample(wave, flux, step, verbose=1):
        return resample(wave, flux, step, verbose)

# load/save ----------------------------------------------------------------------------------------------------------------------

    
    @staticmethod
    def save(wave, flux, para, SAVE_PATH):
        with h5py.File(SAVE_PATH, "w") as f:
            f.create_dataset(f"flux", data=flux, shape=flux.shape)
            f.create_dataset(f"para", data=para, shape=para.shape)
            f.create_dataset(f"wave", data=wave, shape=wave.shape)   
    @staticmethod
    def load(SAVE_PATH):
        with h5py.File(SAVE_PATH, "r") as f:
            flux = f["flux"][:]
            para = f["para"][:]
            wave = f["wave"][:]
        return wave, flux, para