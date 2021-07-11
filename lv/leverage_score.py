import numpy as np
import matplotlib.pyplot as plt

class LeverageScore(object):
    def __init__(self, wave, eigv, k):
        
        self.k = k
        self.eigv = None
        self.lvrg = None
        self.wave = None
        self.wavebin = None
        self.wave_len = None
        self.pidxs = []
        self.init(wave, eigv, R=5000, wave_lb=5000)
        
    def init(self, wave, eigv, R, wave_lb):
        
        w_idx0 = np.digitize(wave_lb, wave)  
        
        self.wave = wave[w_idx0:]
        self.eigv = eigv[w_idx0:, :]
        self.wavebin = self.wave / R


        self.max_wave_idx = len(self.wave)
        self.pc_len = len(self.eigv[0])
        self.get_lvrg()
        self.hist_all = {}

####################################################INIT####################################################
    def get_lvrg(self):
        self.lvrg = np.cumsum(self.eigv**2, axis = 1) / np.arange(1, self.pc_len + 1)
        
    def get_max_p(self):
        pidx = np.argmax(self.lvrg[:, self.k])
        return pidx

    def get_all_roi(self):
        num_iter = 0
        while True:
            self.init_roi(num_iter)
            self.hist_all[num_iter] = self.hist
            num_iter +=1
            break
        
#################################################### GROW ITH ROI ####################################################
            
#     def find_roi(self):
#         self.pidx = self.get_max_p()
#         self.plot_lvrg(self.pidx)
#         return pidx
    
    
    def init_roi(self, num_iter):
        self.hist = {}
        
        self.hist["bnd"] = []
        
    def find_roi(self, pidx):
        lvrg_init, wavebin_init, info_gain_init = self.base_roi()
        while info_gain_init > 0:
            dl, dr, info_gain = self.grow_roi(dl, dr)
            print(dl, dr)
#             break
        return dl, dr
    
    def get_lvrg_sum(self, start_idx, end_idx):
        return np.sum(self.lvrg[start_idx : end_idx, self.k])
    
    def grow_wavebin(self, start_idx, end_idx):
        return self.wavebin[end_idx]- self.wavebin[start_idx]
               
    def base_roi(self):
        self.pidx = self.get_max_p()
        self.pidxs.append(self.pidx)
        self.plot_lvrg(pidx=self.pidx)
        lvrg_init, wavebin_init, info_gain_init = self.grow_roi(0, 0)
        print("base", lvrg_init, wavebin_init, info_gain_init)
        return lvrg_init, wavebin_init, info_gain_init

    def update_roi(self, i, lvrg_sum_i, wavebin_sum_i, info_gain_i):
        left_i, right_i = self.bnd[i]
        lvrg_j, wavebin_j, info_gain_j = self.grow_roi(left_i, right_i)

    def get_roi_gain(self, lvrg_i, lvrg_j, wavebin_i, wavebin_j):
        pass


    def get_region_info(self, left_idx, right_idx):
        start_idx = self.pidx - left_idx
        end_idx   = self.pidx + right_idx

        if (start_idx < 0) or (end_idx > self.max_wave_idx):
            return 0, 0, 0

        lvrg_sum = self.get_lvrg_sum(start_idx, end_idx)
        wavebin_sum = self.get_wavebin_sum(start_idx, end_idx)   
        info_gain = lvrg_sum / wavebin_sum
        return lvrg_sum, wavebin_sum, info_gain
    
    def grow_roi(self, left_i, right_i):
        lvrg_sum_l, wavebin_sum_l, info_gain_l = self.get_region_info(
                                                        left_idx  = left_i + 1, 
                                                        right_idx = right_i
                                                        )        
        lvrg_sum_r, wavebin_sum_r, info_gain_r = self.get_region_info(
                                                        left_idx  = left_i,
                                                        right_idx = right_i + 1
                                                        )
        if info_gain_l > info_gain_r:
            self.bnd.append([left_i + 1, right_i    ])
            return lvrg_sum_l, wavebin_sum_l, info_gain_l
        else:
            self.bnd.append([left_i    , right_i + 1])
            return lvrg_sum_r, wavebin_sum_r, info_gain_r

    
    def plot_lvrg(self, pidx=None):
        f, ax = plt.subplots(figsize=(20, 8))
        pidx = pidx or self.pidx
        if pidx is not None:
            plt.axvline(self.wave[pidx], color='r', label=f"Max P")
        plt.plot(self.wave, self.lvrg[:,self.k], color='k', label=f"Top-{self.k}")
        plt.xlim(self.wave[0], self.wave[-1])
        plt.yscale("log")
        plt.legend()
    
    
            