import numpy as np
import sys
import h5py

def main():
    GRID_PATH = "/datascope/subaru/data/pfsspec/import/stellar/grid/bosz/bosz_5000/spectra.h5"
    with h5py.File(GRID_PATH, 'r') as f:
        flux = f['flux'][()]
        mask = f['flux_idx'][()]
        wave = f['wave'][()]

    idx_s, idx_e = np.digitize([3000, 14000], wave)
    wave0 = wave[idx_s:idx_e]
    flux0 = flux[mask, idx_s:idx_e]

    SAVE_PATH = "/datascope/subaru/user/swei20/data/pfsspec/import/stellar/grid/bosz_5000/spectra.h5"
    with h5py.File(SAVE_PATH, 'w') as f:
        f.create_dataset(f"flux", data=flux0, shape=flux0.shape)
        f.create_dataset(f"wave", data=wave0, shape=wave0.shape)


if __name__ == "__main__":
    main()
