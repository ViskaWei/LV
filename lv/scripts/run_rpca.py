import sys
import h5py
from lv.rpca import RPCA

def main():
    # cmd = "main --config /home/swei20/AE/configs/ae/train/pca_config.json"
    # sys.argv = cmd.split()
    # print(sys.argv)
    # DATA_PATH = "/home/swei20/LV/data/full.h5"
    # SAVE_PATH = "/home/swei20/LV/data/full_pca.h5"
    DATA_PATH = "/home/swei20/LV/data/w45_98.h5"
    SAVE_PATH = "/home/swei20/LV/data/w45_98_pca.h5"
    # PCA_PATH = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/pca/spectra.h5'
    print("LOADING")
    with h5py.File(DATA_PATH, 'r') as f:
        flux = f['flux'][()]
        # wave = f['wave'][()]
        # eigv = f['flux_eigv'][()]
    print("FINISHED LOADING")
    p = RPCA(flux, parallel=1)
    p.pcp()
    p.save(SAVE_PATH)


if __name__ == "__main__":
    main()
