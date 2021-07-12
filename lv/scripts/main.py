import sys
import h5py
from lv.leverage_score import LeverageScore

def main():
    # cmd = "main --config /home/swei20/AE/configs/ae/train/pca_config.json"
    # sys.argv = cmd.split()
    # print(sys.argv)
    
    PCA_PATH = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/pca/spectra.h5'
    with h5py.File(PCA_PATH, 'r') as f:
        wave = f['wave'][()]
        eigv = f['flux_eigv'][()]

    k = 200
    lv=LeverageScore(wave, eigv, k)
    # lv.find_all_roi()
    lv.find_pidx_roi()
    # print("merp")

if __name__ == "__main__":
    main()
