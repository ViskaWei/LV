import numpy as np
import matplotlib.pyplot as plt

class LeverageScore(object):
    def __init__(self, wave, eigv, k, wv_lb=5000):
        
        self.k = k
        self.eigv = None
        self.lvrg = None
        self.wave = None
        self.wvbn = None
        self.wv_dim = None
        self.pc_dim = None
        self.pidxs = []
        self.T = 1.0
        self.max_ep = 100
        self.max_pidx_iter = 1
        self.init(wave, eigv, R=5000, wave_lb=wv_lb)
        
    def init(self, wave, eigv, R, wave_lb):
        
        w_idx0 = np.digitize(wave_lb, wave)  
        
        self.wave = wave[w_idx0:]
        self.eigv = eigv[w_idx0:, :]
        self.wvbn = self.wave / R


        self.max_wave_idx = len(self.wave)
        self.pc_dim = len(self.eigv[0])
        self.get_lvrg()
        self.hist = {}

####################################################INIT####################################################
    def get_lvrg(self):
        self.lvrg = np.cumsum(self.eigv**2, axis = 1) / np.arange(1, self.pc_dim + 1)
        
    def get_max_p(self):
        pidx = np.argmax(self.lvrg[:, self.k])
        return pidx

    def find_all_roi(self):
        pidx_iter = 0
        while (pidx_iter < self.max_pidx_iter):
            self.find_pidx_roi()
            self.hist[pidx_iter] = self.Ss
            pidx_iter +=1
        
#################################################### GROW ITH ROI ####################################################

    def get_lvrg_sum(self, start_idx, end_idx):
        return np.sum(self.lvrg[start_idx : end_idx, self.k])

    def get_wvbn_sum(self, start_idx, end_idx):
        return self.wvbn[end_idx] - self.wvbn[start_idx]
        # return np.sum(self.wvbn[start_idx : end_idx])

    def find_max_pidx(self):
        #TODO: find max pidx iteratively   
        pidx = self.get_max_p()
        return pidx

    def base_roi(self):
        while True:
            pidx = self.find_max_pidx()
            S = self.get_S_from_policy(pidx, pidx)
            if S is not None:
                self.pidxs.append(pidx)
                print("base", S)
                return pidx, S
            else:
                raise NotImplementedError

    def find_pidx_roi(self):
        ep = 0
        # self.As = []
        self.Ss = []        
        pidx, S = self.base_roi()
        terminate = False
        # self.plot_lvrg(pidx=self.pidx)
        while (not terminate) & (ep <= self.max_ep):
            self.Ss.append(S)
            print(f"================================ EP {ep}==============================")
            S, terminate = self.grow(pidx, S)
            ep += 1
        # left_i, right_i, lvrg_sum_i, wvbn_sum_i, gain_i = S
        # self.plot_lvrg(pidx=self.pidx, roi=[left_i, right_i])
        

    def grow(self, pidx, S):
        start_idx, end_idx, lvrg_sum_i, wvbn_sum_i, gain_i = S
        print(f"ROI: {self.wave[start_idx]} - {self.wave[end_idx]} | P Sum {lvrg_sum_i:.2} | lambda {wvbn_sum_i:.2} | gain {gain_i:.2}")
        S_new = self.get_S_from_policy(start_idx, end_idx)
        terminate = self.is_terminate(lvrg_sum_i, wvbn_sum_i, S_new)
        return S_new, terminate

    def is_terminate(self, lvrg_sum_i, wvbn_sum_i, S_new):
        if S_new is None: return True
        start_idx, end_idx, lvrg_sum_j, wvbn_sum_j, gain_j = S_new
        print(f"ROI: {self.wave[start_idx]} - {self.wave[end_idx]} | P Sum {lvrg_sum_j:.2} | lambda {wvbn_sum_j:.2} | gain {gain_j:.2}")

        gain_rate = np.abs((lvrg_sum_j - lvrg_sum_i) / (wvbn_sum_j - wvbn_sum_i))
        print(f"dP: {(lvrg_sum_j - lvrg_sum_i):.2} | dLambda: {(wvbn_sum_j- wvbn_sum_i):.2}")
        upper_bnd = self.T * gain_j
        # upper_bnd = self.T / wvbn_sum_j

        not_converge = gain_rate > upper_bnd
        print(f"gain_rate: {gain_rate:.2} | upper bnd @ T={self.T}: {upper_bnd:.2} | converge: {not not_converge}")
        return not_converge 

    def get_S_from_A(self, A, start_idx, end_idx):
        if   A == "left":
            start_idx -= 1
        elif A == "right":
            end_idx   += 1

        if (start_idx < 0) or (end_idx > self.max_wave_idx):
            return None

        lvrg_sum = self.get_lvrg_sum(start_idx, end_idx)
        wvbn_sum = self.get_wvbn_sum(start_idx, end_idx)   
        gain = lvrg_sum / wvbn_sum
        return [start_idx, end_idx, lvrg_sum, wvbn_sum, gain]
    
    def get_S_from_policy(self, start_idx, end_idx):        
        S_left  = self.get_S_from_A("left",  start_idx, end_idx)
        S_right = self.get_S_from_A("right", start_idx, end_idx)
        return self.policy(S_left, S_right)
         
    def policy(self, S_left, S_right):
        if S_left is not None and S_right is not None:
            gain_l = S_left[-1]
            gain_r = S_right[-1]
            return S_right if gain_r > gain_l else S_left
        else:
            return S_left or S_right

    def plot_lvrg(self, pidx=None, roi=None):
        f, ax = plt.subplots(figsize=(20, 8))
        pidx = pidx or self.pidx
        if pidx is not None:
            plt.axvline(self.wave[pidx], color='r', label=f"Max P")
        plt.plot(self.wave, self.lvrg[:,self.k], color='k', label=f"Top-{self.k}")
        if roi is not None:
            plt.fill_betweenx(self.wave[pidx - roi[0]], self.wave[pidx + roi[1]])
        plt.xlim(self.wave[0], self.wave[-1])
        plt.yscale("log")
        plt.legend()
    
    
            