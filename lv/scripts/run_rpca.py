import sys
import h5py
from lv.rpca import RPCA

def main():
    # cmd = "main --config /home/swei20/AE/configs/ae/train/pca_config.json"
    # sys.argv = cmd.split()
    # print(sys.argv)
    DATA_PATH = "/home/swei20/LV/data/w8_95.h5"
    SAVE_PATH = "/home/swei20/LV/data/w8_95_pca.h5"
    # PCA_PATH = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/pca/spectra.h5'
    with h5py.File(DATA_PATH, 'r') as f:
        flux = f['flux'][()]
        wave = f['wave'][()]
        # eigv = f['flux_eigv'][()]

    p = RPCA(flux)
    p.pcp(parallel=1)
    p.save(SAVE_PATH)


if __name__ == "__main__":
    main()
