import numpy as np
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class LeverageScore(object):
    def __init__(self, wave, eigv, K, T=1.0, N=2, wv_lb=5000, debug=0):
        
        self.K = K
        self.T = T
        self.max_roi_iter = N

        self.wave = None
        self.wvbn = None
        self.eigv = None
        self.mask = None

        self.lvrg = None
        self.lvrg_kmax = None
        self.wv_dim = None
        self.pc_dim = None
        self.pidxs = []
        self.rois = []
        self.hist = {}

        self.max_ep = 1000

        self.setup_logging(debug)
        self.init(wave, eigv, R=5000, wave_lb=wv_lb)
        
    def init(self, wave, eigv, R, wave_lb):
        self.w0 = np.digitize(wave_lb, wave)  
        
        self.wave = wave[self.w0:]
        self.eigv = eigv[self.w0:, :]
        self.wvbn = self.wave / R

        # self.max_wave_idx = len(self.wave)
        self.wv_dim, self.pc_dim = self.eigv.shape
        self.mask = np.zeros(self.wv_dim, dtype=bool)
        # self.wv_dim = len(self.wave)
        # self.pc_dim = len(self.eigv[0])
        self.get_lvrg()

    def setup_logging(self, debug):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=level)
        # logging.basicConfig(filename=f'.log', encoding='utf-8', level=logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(level)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


####################################################INIT####################################################
    def get_lvrg(self):
        self.lvrg = np.cumsum(self.eigv**2, axis = 1) / np.arange(1, self.pc_dim + 1)
        self.lvrg_kmax = self.lvrg[:, self.K].sum()
        
    def get_lvrg_sum(self, start_idx, end_idx):
        return np.sum(self.lvrg[start_idx : end_idx, self.K])

    def get_wvbn_sum(self, start_idx, end_idx):
        return self.wave[end_idx] - self.wave[start_idx]
        # return np.sum(self.wvbn[start_idx : end_idx])

    def find_max_pidx(self):
        pidx = self.get_max_p()
        # self.mask[pidx] = False
        return pidx

    def get_max_p(self):
        masked = np.ma.masked_array(self.lvrg[:, self.K], mask = self.mask)
        pidx = np.argmax(masked)
        return pidx

    def find_all_rois(self):
        roi_iter = 0
        while (roi_iter < self.max_roi_iter):
            self.find_roi()
            self.hist[roi_iter] = self.Ss
            logging.info(f"P: {self.wave[self.pidxs[roi_iter]]} | ROI: {self.rois[roi_iter]}")
            roi_iter +=1
        
#################################################### GROW ITH ROI ####################################################


#################################################### GROW ITH ROI ####################################################
    def find_roi(self, pidx=None):
        pidx, S = self.base_roi(pidx)
        ep = 0
        stop = False
        self.Ss = []        
        while (not stop) & (ep <= self.max_ep):
            logging.debug(f"================================ EP {ep}==============================")
            S, stop = self.grow_roi(ep, pidx, S)
            ep += 1

        self.mask[S[0]:S[1]] = True
        # logging.info(f"P: {pidx} | S, E {S[0], S[1]}")
        self.rois.append([self.wave[S[0]], self.wave[S[1]]])

    def base_roi(self, pidx=None):
        pidx = pidx or self.find_max_pidx()
        S = self.get_S_from_policy(pidx, pidx)
        if S is not None:
            self.pidxs.append(pidx)
            self.mask[pidx] = True
            return pidx, S
        else:
            raise NotImplementedError

    def grow_roi(self, ep, pidx, S):
        self.Ss.append(S)
        start_idx, end_idx, lvrg_sum, wvbn_sum, R = S
        logging.debug(f"EP{ep} | ROI: {self.wave[start_idx]} - {self.wave[end_idx]} | P Sum {lvrg_sum:.2} | lambda {wvbn_sum:.2} | R {R:.2}")

        new_S = self.get_S_from_policy(start_idx, end_idx)
        stop = self.is_stop(lvrg_sum, wvbn_sum, new_S)
        # if stop:
            # logging.debug(f"EP{ep} | ROI: {self.wave[start_idx]} - {self.wave[end_idx]} | P Sum {lvrg_sum:.2} | lambda {wvbn_sum:.2} | R {R:.2}")
        return new_S, stop

    def is_stop(self, old_lvrg_sum, old_wvbn_sum, new_S):
        if new_S is None: return True
        _, _, new_lvrg_sum, new_wvbn_sum, new_R = new_S
        # print(f"ROI: {self.wave[start_idx]} - {self.wave[end_idx]} | P Sum {lvrg_sum_j:.2} | lambda {new_wvbn_sum:.2} | R {R_j:.2}")
        gain_rate = np.abs((new_lvrg_sum - old_lvrg_sum) / (new_wvbn_sum - old_wvbn_sum))
        # print(f"dP: {(new_lvrg_sum - lvrg_sum_i):.2} | dLambda: {(new_wvbn_sum- wvbn_sum_i):.2}")
        # upper_bnd = self.T * new_R
        upper_bnd = self.T * self.lvrg_kmax / new_wvbn_sum

        not_converge = gain_rate > upper_bnd
        if not_converge:
            logging.debug(f"gain_rate: {gain_rate:.2} | upper bnd @ T={self.T}: {upper_bnd:.2} | converge: {not not_converge}")
        return not_converge 
    
    def get_S_from_A(self, A, start_idx, end_idx):
        if   A == "left":
            start_idx -= 1
        elif A == "right":
            end_idx   += 1

        if (start_idx < 0) or (end_idx >= self.wv_dim):
            return None

        lvrg_sum = self.get_lvrg_sum(start_idx, end_idx)
        wvbn_sum = self.get_wvbn_sum(start_idx, end_idx)   
        R = lvrg_sum / wvbn_sum
        return [start_idx, end_idx, lvrg_sum, wvbn_sum, R]
    
    def get_S_from_policy(self, start_idx, end_idx):        
        S_left  = self.get_S_from_A("left",  start_idx, end_idx)
        S_right = self.get_S_from_A("right", start_idx, end_idx)
        return self.policy(S_left, S_right)
         

    def policy(self, S_left, S_right):
        if S_left is not None and S_right is not None:
            lvrg_sum_l = S_left[2]
            lvrg_sum_r = S_right[2]
            return S_right if lvrg_sum_r > lvrg_sum_l else S_left
        else:
            return S_left or S_right
    # def policy(self, S_left, S_right):
    #     if S_left is not None and S_right is not None:
    #         R_l = S_left[-1]
    #         R_r = S_right[-1]
    #         return S_right if R_r > R_l else S_left
    #     else:
    #         return S_left or S_right

    def plot_lvrg_i(self, pidx=None, roi=None):
        f, ax = plt.subplots(figsize=(20, 8))
        pidx = pidx or self.pidx
        if pidx is not None:
            plt.axvline(self.wave[pidx], color='r', label=f"Max P")
        plt.plot(self.wave, self.lvrg[:,self.K], color='k', label=f"K = {self.K}")
        if roi is not None:
            plt.axvspan(roi[0], roi[1],  label = f"ROI_{ii + 1}")
            # plt.fill_betweenx(self.wave[pidx - roi[0]], self.wave[pidx + roi[1]])
        plt.xlim(self.wave[0], self.wave[-1])
        plt.yscale("log")
        plt.legend()
    
    def plot_rois(self, flux=None, ax=None, log=1):
        if ax is None:
            f, ax = plt.subplots(figsize=(30, 6))
        colors = cm.get_cmap('Spectral', self.max_roi_iter)
        for ii, roi in enumerate(self.rois):
            pidx = self.pidxs[ii]
            ax.axvspan(roi[0], roi[1], label = f"R{ii + 1}", color=colors(ii), alpha=0.3)
            ax.axvline(self.wave[pidx], color='r', alpha=0.3)
        if flux is None:
            ax.plot(self.wave, self.lvrg[:,self.K], color='k', alpha=1., lw=1.)
            ax.set_ylabel(f"K{self.K} LS")
        else:
            ax.plot(self.wave, flux[self.w0:], color='k', alpha=1., lw=1.)
            ax.set_ylabel(f"Flux")
            # ax.legend(ncol=int(self.max_roi_iter//2))
        ax.set_xlim(self.wave[0], self.wave[-1])
        if log: ax.set_yscale("log")
            
    def plot_max_ps(self, ax=None, pidxs=None, log=1):
        if ax is None:
            f, ax = plt.subplots(figsize=(30, 6))
        for ii, pidx in enumerate(pidxs):
            # ax.axvspan(roi[0], roi[1], label = f"ROI_{ii + 1}")
            ax.axvline(self.wave[pidx], color='k')
        ax.plot(self.wave, self.lvrg[:,self.K], color='r', alpha=0.8, lw=0.3, label=f"K = {self.K}")
        
        ax.set_xlim(self.wave[0], self.wave[-1])
        if log: ax.set_yscale("log")
        ax.set_ylabel(f"Leverage Score")
        ax.legend()

    def plot_flux_rois(self, flux, legend=0, title=None):
        f, axs = plt.subplots(2,1, figsize=(10,4), sharex="all")
        self.plot_rois(flux=flux, log=0, ax=axs[0])
        self.plot_rois(flux=None, log=1, ax=axs[1])
        # plt.xlim(8300, 8800)
        if legend: 
            axs[0].legend(ncol=int(self.max_roi_iter//2))
        f.suptitle(title)

    # def plot_Lvrg(self):

    # def correct_wave_grid(wlim):
    #     RESOLU = 5000
    #     WLBEG = wlim[0]  # nm
    #     WLEND = wlim[1]  # nm
    #     RATIO = 1. + 1. / RESOLU
    #     RATIOLG = np.log10(RATIO)
    #     IXWLBEG = int(np.log10(WLBEG) / RATIOLG)
    #     WBEGIN = 10 ** (IXWLBEG * RATIOLG)

    #     if WBEGIN < WLBEG:
    #         IXWLBEG = IXWLBEG + 1
    #         WBEGIN = 10 ** (IXWLBEG * RATIOLG)
    #     IXWLEND = int(np.log10(WLEND) / RATIOLG)
    #     WLLAST = 10 ** (IXWLEND * RATIOLG)
    #     if WLLAST > WLEND:
    #         IXWLEND = IXWLEND - 1
    #         WLLAST = 10 ** (IXWLEND * RATIOLG)
    #     LENGTH = IXWLEND - IXWLBEG + 1
    #     DWLBEG = WBEGIN * RATIO - WBEGIN
    #     DWLLAST = WLLAST - WLLAST / RATIO

    #     a = np.linspace(np.log10(10 * WBEGIN), np.log10(10 * WLLAST), LENGTH)
    #     cwave = 10 ** a

    #     return cwave
