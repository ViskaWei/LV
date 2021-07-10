
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


        self.wave_len = len(self.wave)
        self.pc_len = len(self.eigv[0])
        
    def prepare(self):
        self.get_lvrg()
        pidx = self.get_max_p()
        self.plot_lvrg(pidx)
        return pidx
        
    def find_roi(self, pidx):
        dl, dr, info_gain = 0, 0, np.inf
        while info_gain > 0:
            dl, dr, info_gain = self.grow_roi(dl, dr)
            print(dl, dr)
#             break
        return dl, dr
        
        
    def get_lvrg(self):
        self.lvrg = np.cumsum(self.eigv**2, axis = 1) / np.arange(1, self.pc_len + 1)
        
    def get_max_p(self):
        pidx = np.argmax(self.lvrg[:, self.k])
        return pidx
    
    def get_info_gain(self, pidx, dl, dr):
        pidx_l = pidx - dl
        pidx_r = pidx + dr
        if pidx_l < 0 or pidx_r > self.wave_len:
            return 0, 0

        lvrg_sum = np.sum(self.lvrg[(pidx - dl):(pidx + dr), self.k])
        dl = np.sum(self.wavebin[(pidx - dl):(pidx + dr)])    
        info_gain = lvrg_sum / dl
        return lvrg_sum, dl, info_gain
    
    def grow_roi(self, dl, dr,):
        lvrg_sum_l, dl_l, info_gain_l = self.get_info_gain(pidx, dl=dl + 1, dr=dr)        
        lvrg_sum_r, dl_r, info_gain_r = self.get_info_gain(pidx, dl=dl, dr=dr + 1)
        
        if info_gain_l > info_gain_r:
            dl += 1
            lvrg_sum_new = lvrg_sum_l
            dl_new = dl_l
            info_gain_new = info_gain_l
            
        else:

            lvrg_sum_new = lvrg_sum_l
             info_gain_r
            
            
        return dl, dr, info_gain
    
    def get 
     lvrg_sum, dl, info_gain
    
    def plot_lvrg(self, pidx=None):
        f, ax = plt.subplots(figsize=(20, 8))
        if pidx is not None:
            plt.axvline(self.wave[pidx], color='r', label=f"Max P")
        plt.plot(self.wave, self.lvrg[:,self.k], color='k', label=f"Top-{self.k}")
        plt.xlim(self.wave[0], self.wave[-1])
        plt.yscale("log")
        plt.legend()
    
    
            