import numpy as np
import h5py 


bnds=[]
for val in axes.values():
    bnds.append([val.values[0], val.values[-1]])
bnds

bnds0 =[[-2.0, -0.3], [5000.0, 6000.0], [4,5.0], [-0.75, 0.5], [-0.25, 0.5]]

for ii, val in enumerate(axes.values()):
    lb, ub = val.values[0], val.values[-1]

N=100000
index = np.zeros((N, 5))
for ii, val in enumerate(bnds0):
    lb, ub = val[0], val[-1]
    index[:,ii] = np.random.uniform(lb, ub, N)
    
train_pc = np.zeros((N, 1000))

for i in tqdm(range(N)):
    p = index[i]
    train_pc[i] = rbf.grid.grid.get_value('flux', 
                                         Fe_H=p[0], 
                                         T_eff=p[1], 
                                         log_g=p[2], 
                                         C_M=p[3], 
                                         O_M=p[4])
    
flux = np.dot(train_pc[2], rbf.grid.eigv['flux'].T)

plt.plot(rbf.wave, flux)

# SAVE_DIR= '/scratch/ceph/swei20/data/pfsspec/train/ae/dataset/bosz/bosz_5000' 
SAVE_DIR = "/home/swei20/LV/data/"
SAVE_PATH = os.path.join(SAVE_DIR, "PCP_100k_solar.h5")

with h5py.File(SAVE_PATH, 'w') as f:
    f.create_dataset('PC', data = train_pc, shape=train_pc.shape)
    f.create_dataset('RBF_PATH', data = RBF_PATH)
    f.create_dataset('eigv', data = rbf.grid.eigv['flux'])
    f.create_dataset('eigs', data = rbf.grid.eigs['flux'])
    f.create_dataset('wave', data = rbf.wave)
    f.create_dataset('pval', data = index)
    

def get_flux_in_Wrange(flux, wave, Ws):
    start = np.digitize(Ws[0], wave)
    end = np.digitize(Ws[1], wave)
    return flux[:, start:end], wave[start:end]
    
def save(SAVE_PATH, flux, wave, para):
    with h5py.File(SAVE_PATH, 'w') as f:
        f.create_dataset('flux', data = flux, shape=flux.shape)
        f.create_dataset('wave', data = wave, shape=wave.shape)
        f.create_dataset('pval', data = para, shape=para.shape)
        f.create_dataset("rbf_path", data = SAVE_ALL_PATH)