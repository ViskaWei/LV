import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pfsspec.stellarmod.alexcontinuummodel import AlexContinuumModel, AlexContinuumModelTrace
from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.bosz import Bosz

import numpy as np
import pandas as pd
import copy

xBalmer, xPaschen = np.log([3647.04, 8205.96])
x1Balmer, x1Paschen = np.log([4200, 12000])
xmin, xmax = np.log(3000), np.log(14000)


r_pix = 14
c_pix = 2.8

# FIT_PATH = '/scratch/ceph/swei20/spec_fit/all_1029_t2'

def pad_grid(fit, orig_values=None, s=None, interpolation = 'ijk'):
    orig_axes = fit.get_axes(squeeze=False)
    if orig_values is None:
        orig_values = fit.grid.get_value('params', s=s, squeeze=False)
        orig_values[orig_values == -1e3] = 0.0
        orig_values[~fit.grid.value_indexes['params']] = np.nan    
    padded_params, padded_axes = ArrayGrid.pad_array(orig_axes, orig_values, interpolation=interpolation)
    padded_params[padded_params < 0] = 0
    return padded_params, padded_axes

def get_grid(FIT_PATH):
    fn = os.path.join(os.environ['PFSSPEC_DATA'], FIT_PATH, 'spectra.h5')
    fit = ModelGrid(Bosz(), ArrayGrid)
    fit.preload_arrays = False
    fit.load(fn, format='h5')
    for k in fit.grid.axes:
        print(k, fit.grid.axes[k].values)
    for k in fit.grid.values:
        print(k, sep = '')
    params = fit.grid.get_value('params') 
    print(params.shape)
    return fit, params

def pad_fill_grid(grid, pad_val=None, n_pad = 1, fill_val=0.0, interpolation = 'ijk'):
    orig_axes = grid.grid.get_axes(squeeze=False)
    orig_value = grid.grid.get_value('params', s=None, squeeze=False)
    mask = grid.grid.get_value_index('params')
    if pad_val is not None:
        orig_value = np.where(orig_value==pad_val, fill_val, orig_value)
    orig_value[~mask] = np.nan
    fill_value, fill_axes = grid.grid.fill_holes(orig_axes, orig_value, mask)
    padded_params, padded_axes = ArrayGrid.pad_array(fill_axes, fill_value, interpolation=interpolation)
    padded_params[padded_params < 0] = 0

    for i in range(n_pad - 1):
        padded_params, padded_axes = ArrayGrid.pad_array(padded_axes, padded_params, interpolation=interpolation)
        padded_params[padded_params < 0] = 0
    print(padded_params.shape, n_pad, padded_params[n_pad:-n_pad, n_pad:-n_pad, n_pad:-n_pad].shape)
    return padded_params, padded_axes

def plot_fit_smth(fit_params, smth_params, grid=None, log_idx=10, pid=26, vmin=0.0, vmax=None, cbar=True):
    nPlot = 3
    r_pix = 3
    c_pix = 2
    f, axs = plt.subplots(1, nPlot, sharex='all', sharey='all', facecolor='w', \
        figsize=(r_pix * nPlot, c_pix) )
    im0 = fit_params[:, :, log_idx, 0, 0, pid]
    im1 = smth_params[:, :, log_idx, 0, 0, pid]
    images = [im0, im1, im0-im1]

    if vmax is None: vmax = np.nanmax(images[0])
    for i in range(2):
        ss = axs[i].imshow(images[i],\
                    aspect='auto', cmap='Spectral', vmin=vmin, vmax=vmax)
    s_diff = axs[2].imshow(images[2],\
            aspect='auto', cmap='viridis')
    log_g = grid.grid.axes['log_g'].values[log_idx]
    axs[0].set_title(f'Balmer Jump Amplitude @ log g = {log_g}')
    axs[1].set_title(f'Anisotropic Diffusion Filter')

    # axs[1].set_title(f'smoothed w/ $\kappa$ {k} | $\gamma$ {lr} | #iter {n_iter} ')
    axs[2].set_title('Error')
    cax0 = f.add_axes([0.02, 0.1, 0.01, 0.8]) 
    cax_diff = f.add_axes([0.92, 0.1, 0.01, 0.8]) 

    f.colorbar(ss, cax = cax0)
    f.colorbar(s_diff, cax = cax_diff)
    # if span == "Log_g":
    #     row = len(Ls)
    #     dy = 2
    #     y, z = "Fe_H", "Log_g"
    #     ys, zs = Fs, Ls
    # elif span == "Fe_H":
    #     row = len(Fs)
    #     dy= 2
    #     ys, zs = Ls, Fs
    #     y, z = "log_g", "Fe_H"   
    for ax in axs:
        get_ax_label(ax, grid, y='Fe_H', dy=2, dx = 20)
    return images[-1]

def plot_slice(params, log_idx=10, pid=26, n_pad=0, vmin=None, vmax=None, ax = None, cbar=True):
    if ax is None: ax = plt.gca()
    ss = ax.imshow(params[:, :, log_idx + n_pad, 0, 0, pid],\
                    aspect='auto', cmap='Spectral', vmin=vmin, vmax=vmax)
    ax.grid(True)
    if cbar: plt.colorbar(ss, ax = ax)
    return ss

def plot_image(params, vmin=None, vmax=None, ax = None, cbar=True):
    if ax is None: ax = plt.gca()
    ss = ax.imshow(params, aspect='auto', cmap='Spectral', vmin=vmin, vmax=vmax)
    ax.grid(True)
    if cbar: plt.colorbar(ss, ax = ax)
    return ss

def process_params(fit, gap_fill=-1000., fill_gap=np.nan):
    params = fit.grid.get_value('params')    
    params[params == gap_fill] = fill_gap
    params[~fit.grid.value_indexes['params']] = np.nan
    params_exist = copy.deepcopy(params)
    return params, params_exist    

def get_ax_label(ax = None, grid = None, x = 'T_eff', y = 'Fe_H', dx = 20, dy = 2):
    if ax is None: ax = plt.gca()
    ax.set_xticks(np.arange(grid.grid.axes[x].values.shape[0])[::dx])
    ax.set_xticklabels(grid.grid.axes[x].values[::dx])
#     ax.set_xlabel(r'$T_\mathrm{eff}$')
    ax.set_xlabel(x)
    ax.set_yticks(np.arange(grid.grid.axes[y].values.shape[0])[::dy])
    ax.set_yticklabels(grid.grid.axes[y].values[::dy])
    # ax.set_ylabel(r'[Fe/H]')
    ax.set_ylabel(y)

def get_ax_label_xy(ax, xs, ys, dx = 20, dy = 3):
    ax.set_xticks(np.arange(xs.shape[0])[::dx])
    ax.set_xticklabels(xs[::dx])
    ax.set_yticks(np.arange(ys.shape[0])[::dy])
    ax.set_yticklabels(ys[::dy])

# def plot_one_slice(params, params_exist, s):
#     ax.set_title(f'{param_name[param_idx]}')
#     ax.imshow(np.isnan(image_exist), aspect = 'auto', cmap = cmap)
#     if param_idx == 2:
#         ss = ax.imshow(image, aspect='auto', vmin = gap_vmin[limit_id])  
#     else:
#         ss = ax.imshow(image, aspect='auto')  
#     f.colorbar(ss, ax = ax)

def plot_gap_params(params, params_exist, grid = None, span = "Log_g", limit_id = 0, offset = 0, C_M_idx = 0, O_M_idx = 0, dx = 20):

    cmap = ListedColormap(['w', 'lightpink'])
    col = 5
    Ts = grid.grid.axes["T_eff"].values     
    Fs = grid.grid.axes["Fe_H"].values     
    Ls = grid.grid.axes["log_g"].values
    if span == "Log_g":
        row = len(Ls)
        dy = 2
        y, z = "Fe_H", "Log_g"
        ys, zs = Fs, Ls
    elif span == "Fe_H":
        row = len(Fs)
        dy= 2
        ys, zs = Ls, Fs
        y, z = "log_g", "Fe_H"    
    param_name = ['a: gap', 'b: slope', 'c: mid_pt', 's0', 's1']
    gap_names = ["3000A", "", "Balmer", "", "Paschen"]
    gap_vmin = [np.log(3000), 0, np.log(3640), 0, np.log(8200)]
    gap_name = gap_names[limit_id]
    offset1 = offset + (limit_id // 2) * col
    for r in range(row):
        f, axs = plt.subplots(1, col, figsize=(r_pix, c_pix), sharex='all', sharey='all', facecolor='w')
        f.suptitle(f'{z} = {zs[r]} | Gap {limit_id}: {gap_name}')
        for param_idx in range(col):
            ax = axs[param_idx]
            if span == "Log_g":
                s = np.s_[:, :, r, C_M_idx, O_M_idx, offset1 + param_idx]  
                image, image_exist = params[s] ,params_exist[s]
            elif span == "Fe_H":
                s = np.s_[r, :, :, C_M_idx, O_M_idx, offset1 + param_idx]  
                image, image_exist = params[s].T ,params_exist[s].T
            else:
                raise NotImplementedError             
            ax.set_title(f'{param_name[param_idx]}')
            ax.imshow(np.isnan(image_exist), aspect = 'auto', cmap = cmap)
            if param_idx == 2:
                ss = ax.imshow(image, aspect='auto', vmin = gap_vmin[limit_id])  
            # elif param_idx == 3 or param_idx == 4:
            #     # vmin = np.nanquantile(image, 0.9) 
            #     ss = ax.imshow(image, aspect='auto')  
            else:
                ss = ax.imshow(image, aspect='auto')  
            
            f.colorbar(ss, ax = ax)
            get_ax_label(ax, grid, y=y, dy=dy, dx = dx)


def plot_param_a(params, params_exist, grid, span="Log_g", col=4, limit_id=2, offset=21, C_M_idx=0, O_M_idx=0, dx=20, cmap="spectral", cmap_exist="lightpink"):
    cmap_filter = ListedColormap(['w', cmap_exist])
    Ts = grid.grid.axes["T_eff"].values     
    Fs = grid.grid.axes["Fe_H"].values     
    Ls = grid.grid.axes["log_g"].values
    if span == "Log_g":
        row = len(Ls)
        dy = 2
        y, z = "Fe_H", "Log_g"
        ys, zs = Fs, Ls
    elif span == "Fe_H":
        row = len(Fs)
        dy= 2
        ys, zs = Ls, Fs
        y, z = "log_g", "Fe_H"    
    param_name = ['a: gap', 'b: slope', 'c: mid_pt', 's0', 's1']
    gap_names = ["3000A", "", "Balmer", "", "Paschen"]
    gap_vmin = [np.log(3000), 0, np.log(3640), 0, np.log(8200)]
    gap_name = gap_names[limit_id]
    param_idx = offset + (limit_id // 2) * 5
    f, axss = plt.subplots(col, int(row // col) + 1, figsize=(r_pix, c_pix * col), sharex='all', sharey='all', facecolor='w')

    f.suptitle(f'Amplitude @ Gap {limit_id}: {gap_name}')
    axs = axss.flatten()
    for r in range(row):
        ax = axs[r]
        if span == "Log_g":
            s = np.s_[:, :, r, C_M_idx, O_M_idx, param_idx]  
            image, image_exist = params[s] ,params_exist[s]
        elif span == "Fe_H":
            s = np.s_[r, :, :, C_M_idx, O_M_idx, param_idx]  
            image, image_exist = params[s].T ,params_exist[s].T
        else:
            raise NotImplementedError             
        ax.set_title(f'{z} = {zs[r]}')
        ax.imshow(np.isnan(image_exist), aspect = 'auto', cmap = cmap_filter)
            # elif param_idx == 3 or param_idx == 4:
            #     # vmin = np.nanquantile(image, 0.9) 
            #     ss = ax.imshow(image, aspect='auto')  
        ss = ax.imshow(image, aspect='auto', cmap = cmap)  
        f.colorbar(ss, ax = ax)
        get_ax_label(ax, grid, y=y, dy=dy, dx = dx)

def plot_param_a_alex(params, params_exist, grid, col=4, span="Log_g", limit_id=2, offset=21, vmax=0.8, C_M_idx=0, O_M_idx=0, dx=20, cmap="Spectral", cmap_exist="lightpink"):
    cmap_filter = ListedColormap(['w', cmap_exist])
    Ts = grid.grid.axes["T_eff"].values     
    Fs = grid.grid.axes["Fe_H"].values     
    Ls = grid.grid.axes["log_g"].values
    if span == "Log_g":
        row = len(Ls)
        dy = 2
        y, z = "Fe_H", "Log_g"
        ys, zs = Fs, Ls
    elif span == "Fe_H":
        row = len(Fs)
        dy= 2
        ys, zs = Ls, Fs
        y, z = "log_g", "Fe_H"    
    param_name = ['a: gap', 'b: slope', 'c: mid_pt', 's0', 's1']
    gap_names = ["3000A", "", "Balmer", "", "Paschen"]
    gap_vmin = [np.log(3000), 0, np.log(3640), 0, np.log(8200)]
    gap_name = gap_names[limit_id]
    param_idx = offset + (limit_id // 2) * 5
    # f, axss = plt.subplots(col, int(row // col) + 1, figsize=(20, 5 * col), sharex='all', sharey='all', facecolor='w')
    f, axss = plt.subplots(col, int(row // col) + 1, figsize=(r_pix, c_pix * col), sharex='all', sharey='all', facecolor='w')

    axs = axss.flatten()
    f.suptitle(f'Amplitude @ {gap_name} Break')
    for r in range(row):
        ax = axs[r]
        if span == "Log_g":
            s = np.s_[:, :, r, C_M_idx, O_M_idx, param_idx]  
            image, image_exist = params[s] ,params_exist[s]
        elif span == "Fe_H":
            s = np.s_[r, :, :, C_M_idx, O_M_idx, param_idx]  
            image, image_exist = params[s].T ,params_exist[s].T
        else:
            raise NotImplementedError             
        ax.set_title(f'{z} = {zs[r]}')
        ax.imshow(np.isnan(image_exist), aspect = 'auto', cmap = cmap_filter)
            # elif param_idx == 3 or param_idx == 4:
            #     # vmin = np.nanquantile(image, 0.9) 
            #     ss = ax.imshow(image, aspect='auto')  
        ss = ax.imshow(image, aspect='auto', cmap = cmap, vmax=vmax)  
        get_ax_label(ax, grid, y=y, dy=dy, dx = dx)
    f.colorbar(ss, ax = axs[-1])

####_ansiodiff_###################################################################

def plot_2d_smoothed(padded_params, log_idx, pid=26, n_pad=1, n_iter=1, k=25, lr=0.1, vmin=0.0, grid=None):
    f,axs = plt.subplots(1,2, figsize=(10,5), facecolor='w')
    ss = plot_slice(padded_params[n_pad:-n_pad, n_pad:-n_pad, n_pad:-n_pad], ax=axs[0], vmin=vmin, vmax=None, cbar=0)
    image = padded_params[:, :, log_idx + n_pad , 0, 0, pid]
    smoothed_params = ansiodiff_2d(image, n_iter=n_iter, k=k, lr=lr, option = 1)
    vmax = np.nanmax(image[n_pad:-n_pad, n_pad:-n_pad])
    _ = plot_image(smoothed_params[n_pad:-n_pad, n_pad:-n_pad], ax=axs[1], vmin=vmin, vmax=vmax, cbar = 0)
    log_g = grid.grid.axes['log_g'].values[log_idx]
    axs[0].set_title(f'amplitude @ log g = {log_g}')
    axs[1].set_title(f'smoothed w/ $\kappa$ {k} | $\lambda$ {lr} | #iter {n_iter} ')
    cbaxes = f.add_axes([0.95, 0.1, 0.02, 0.8]) 
    f.colorbar(ss, cax = cbaxes)
    for ax in axs:
        get_ax_label(ax = ax, grid = grid)

def plot_2d_smoothed_diff(padded_params, log_idx, pid=26, n_pad=1, n_iter=1, k=25, lr=0.1, vmin=0.0, grid=None):
    f,axs = plt.subplots(1,3, figsize=(12,2.5), facecolor='w')
    pad_cropped = padded_params[n_pad:-n_pad, n_pad:-n_pad, n_pad:-n_pad]
    ss0 = plot_slice(pad_cropped, ax=axs[0], vmin=vmin, vmax=None, cbar=0)

    image = padded_params[:, :, log_idx + n_pad , 0, 0, pid]
    smoothed_params = ansiodiff_2d(image, n_iter=n_iter, k=k, lr=lr, option = 1)
    smth_cropped = smoothed_params[n_pad:-n_pad, n_pad:-n_pad]
    vmax = np.nanmax(image[n_pad:-n_pad, n_pad:-n_pad])
    _ = plot_image(smth_cropped, ax=axs[1], vmin=vmin, vmax=vmax, cbar = 0)

    diff = pad_cropped[:, :, log_idx, 0, 0, pid] - smth_cropped
    ss1 = plot_image(diff, ax=axs[2], cbar = 0)

    log_g = grid.grid.axes['log_g'].values[log_idx]
    axs[0].set_title(f'amplitude @ log g = {log_g}')
    axs[1].set_title(f'smoothed w/ $\kappa$ {k} | $\lambda$ {lr} | #iter {n_iter} ')
    axs[2].set_title('diff')
    cax0 = f.add_axes([0.05, 0.1, 0.01, 0.8]) 
    cax1 = f.add_axes([0.95, 0.1, 0.01, 0.8]) 

    f.colorbar(ss0, cax = cax0)
    f.colorbar(ss1, cax = cax1)
    
    for ax in axs:
        get_ax_label(ax = ax, grid = grid)

def anisotropic_nd(img, niter=1, kappa=50, gamma=0.1, option=1):
    if option == 1:
        def condgradient(delta):
            return np.exp(-(delta/kappa)**2.)
    elif option == 2:
        def condgradient(delta):
            return 1./(1.+(delta/kappa)**2.)
    out = np.array(img, dtype=np.float32, copy=True)
    deltas = [np.zeros_like(out) for _ in range(out.ndim)]
    for _ in range(niter):
        for i in range(out.ndim):
            slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
            deltas[i][slicer] = np.diff(out, axis=i)
        matrices = [condgradient(delta) * delta for delta in deltas]
        for i in range(out.ndim):
            slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
            matrices[i][slicer] = np.diff(matrices[i], axis=i)
        out += gamma * (np.sum(matrices, axis=0))
    return out



def ansiodiff_2d(image_orig, n_iter = 1, k = 25, lr = 0.1, option = 1):
    assert len(image_orig.shape) == 2 
    image = np.copy(image_orig)
    image[np.isnan(image)] = 0.0
    row, col = image.shape
    for i in range(n_iter):
        new = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        dN = new[:row, 1:(col + 1)] 
        dS = new[2:, 1:(col + 1)]
        dE = new[1:(row + 1), 2:]
        dW = new[1:(row + 1), :col]
        dd = np.array([dN, dS, dE, dW]) - image
        if option == 1:
            coeff = np.exp(-(dd / k) ** 2.)
        elif option == 2:
            coeff = 1. / (1. + (dd / k) ** 2.)
        else:
            raise NotImplementedError
        update = (coeff * dd).sum(axis = 0)
        image = image + lr * update 
    return image

def plot_ansiodiff(image, grid = None, n_iter = 1, k = 25, lr = 0.1):
    f, axs = plt.subplots(1, 3, figsize=(20, 5))
    image_orig = np.copy(image)
    vmax, vmin = np.max(image), np.min(image)
    ss = axs[0].imshow(image, aspect = 'auto', vmin = vmin, vmax = vmax)
    image_op1 = ansiodiff_2d(image, n_iter = n_iter, k = k, lr = lr, option = 1)
    axs[1].imshow(image_op1, aspect = 'auto',  vmin = vmin, vmax = vmax)    
    axs[1].set_title('option 1')
    image_op2 = ansiodiff_2d(image, n_iter = n_iter, k = k, lr = lr, option = 2)
    axs[2].imshow(image_op2, aspect = 'auto',  vmin = vmin, vmax = vmax)
    axs[2].set_title('option 2')
    [get_ax_label(ax = ax, grid = grid) for ax in axs]
    f.colorbar(ss, ax = axs[0])

    f.suptitle(f'smoothed with kappa {k} | lambda {lr} | iter {n_iter}')

####_ansiodiff_###################################################################

def check_spec(Fe_H, T_eff, log_g, C_M, O_M, grid = None):
    spec = grid.get_nearest_model(Fe_H=-2.5, T_eff=3000, log_g=4.0, C_M=-0.5, O_M=0)
    if spec is None: 
        return "no model"
    else:
        return "model exist"


# def check_fit_all(FList, TList, LList, grid=None, out=False, C_M=0.0, O_M=0.0):
#     self = AlexContinuumModel(debug = True)    
#     offset = [0, 0.02, 0.01]
#     for Fe_H in FList:
#         for T_eff in TList:
#             for log_g in LList:
#                 spec = grid.get_nearest_model(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g, C_M=C_M, O_M=O_M)
#                 if spec is None: continue
#                 f, axs = plt.subplots(1, 3, figsize=((16, 3)))
#                 name = f'Fe_H {Fe_H} | T_eff {T_eff} | Log(g) {log_g}'
#                 f.suptitle(name)
#                 self.trace.limit_fit = {0: True, 2: True, 4: True, 6: False}
#                 params = self.normalize(spec)
#                 for idx, limit_id in enumerate([0, 2, 4]):
#                     mask = self.gap_masks[limit_id]
#                     x, y = self.log_wave[mask], self.trace.norm_flux[mask]
#                     ymin = np.median(y[:50])
#                     if self.trace.limit_fit[limit_id] is True: 
#                         hb = self.trace.hb[limit_id]
#                         axs[idx].scatter(*hb.T, c = 'b', alpha = 0.5)
#                         hb2 = self.trace.hull[limit_id]
#                         axs[idx].scatter(*hb2.T, c = 'r')
                        
#                         y0, slope_mid, x_mid, s0, s1 = self.trace.params_est[limit_id]
#                         f_init = self.sigmoid_fn(x, y0, slope_mid, x_mid, s0, s1)
#                         axs[idx].plot(x, f_init, c = 'g', alpha = 0.2)
#     #                         c, b, half_y = self.get_init_sigmoid_estimation(hb)                        
#                         pmt = self.params[limit_id]
#                         fx = self.sigmoid_fn(x, *pmt)
#                         axs[idx].plot(x, fx, c = 'r', alpha = 1)
#                         axs[idx].plot(x_mid, -y0/2., 'go')
#                         axs[idx].axhline(-y0, c = "cyan", linestyle = ":")   
#                         ymin = -y0

#                     axs[idx].plot(self.log_wave, self.trace.norm_flux, 'k', lw = 0.4, alpha = 0.3)
#                     axs[idx].plot(self.log_wave, spec.cont, 'k', lw = 1)
#                     axs[idx].axvline(x[0], c = "cyan", linestyle = ":")                
#                     axs[idx].set_ylim(ymin * 1.1, 0.)
#                     axs[idx].set_xlim(min(x) - offset[idx], np.log(self.gap_masks_ub[limit_id]))
#                 plt.show()
#     #                     axs[1, idx].plot(hb[self.offset[limit_id]:, 0], slope, 'mo', alpha = 1)
#     if out: return self


def check_fit_all(FList, TList, LList, grid=None, out=False, C_M=0.0, O_M=0.0):
    self = AlexContinuumModel(debug = True)    
    offset = [0, 0.02, 0.01]
    nPlot = len(FList) * len(TList) * len(LList)
    f, axss = plt.subplots(nPlot, 3, figsize=((12,3 * nPlot)))
    i = 0
    for Fe_H in FList:
        for T_eff in TList:
            for log_g in LList:
                axs = axss if nPlot == 1 else axss[i]
                i += 1
                name = f'Fe_H {Fe_H} | T_eff {T_eff} | Log(g) {log_g}'
                spec = grid.get_nearest_model(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g, C_M=C_M, O_M=O_M)
                if spec is None: continue
                params = self.normalize(spec)
                for idx, limit_id in enumerate([0, 2, 4]):
                    mask = self.gap_masks[limit_id]
                    x, y = self.log_wave[mask], self.trace.norm_flux[mask]
                    ymin = np.median(y[:50])
                    
                    if self.trace.limit_fit[limit_id] is True: 
                        hb = self.trace.hb[limit_id]
                        axs[idx].scatter(*hb.T, c = 'b', alpha = 0.5)
                        if 1==1:
                            hb2 = self.trace.hull[limit_id]
                            axs[idx].scatter(*hb2.T, c = 'r')

                        y0, slope_mid, x_mid, s0, s1 = self.trace.params_est[limit_id]
                        f_init = self.sigmoid_fn(x, y0, slope_mid, x_mid, s0, s1)
                        axs[idx].plot(x, f_init, c = 'g', alpha = 0.2)
    #                         c, b, half_y = self.get_init_sigmoid_estimation(hb)                        
                        pmt = self.params[limit_id]
                        fx = self.sigmoid_fn(x, *pmt)
                        axs[idx].plot(x, fx, c = 'r', alpha = 1)

                        axs[idx].plot(x_mid, -y0/2., 'go')
                        axs[idx].axhline(-y0, c = "cyan", linestyle = ":")   
                    axs[idx].plot(self.log_wave, self.trace.norm_flux, 'k', lw = 0.4, alpha = 0.3)
                    axs[idx].plot(self.log_wave, spec.cont, 'k', lw = 1)
                    axs[idx].axvline(x[0], c = "cyan", linestyle = ":")                
                    axs[idx].set_ylim(ymin * 1.1, 0.)
                    axs[idx].set_xlim(min(x) - offset[idx], np.log(self.gap_masks_ub[limit_id]))
    #                     axs[1, idx].plot(hb[self.offset[limit_id]:, 0], slope, 'mo', alpha = 1)
    if out: return self

def plot_smooth_diff(grid, smth_params, out=1, idx = [12, 11, 10, 3, 1]):
    # r_pix = 7
    # c_pix = 2
    r_pix, c_pix = 16, 8
    smth_idx = tuple(idx[:3] + [0,0])
    spec = grid.get_model_at(tuple(idx))
    smth_limit_param = smth_params[smth_idx][21:].reshape(3, -1)
    trace = AlexContinuumModelTrace()
    self = AlexContinuumModel(trace=trace)
    self.init_wave(spec.wave)
    params = self.fit(spec)
    self.normalize(spec, params)
    offset = [0.04, 0.04]
    # offset_y = [0.02, 0.0]
    limit_ids = [1, 2]
    limit_names = ['Balmer', 'Paschen']
    color = ['lightsalmon', 'lightgreen']
    f, axs = plt.subplots(1, 2, figsize=((r_pix, c_pix)), facecolor = 'w')
    # name = f'[Fe/H] = {Fe_H} | T_eff = {T_eff}K | log(g) = {log_g} | [C/M] = {C_M} | [O/M] = {O_M}'

    for idx, limit_id in enumerate(limit_ids):
        axs[idx].axvspan(np.log(self.limit_wave[limit_id]), np.log(self.blended_bounds[limit_id]),\
                         alpha=0.3, color=color[idx], label=limit_names[idx])
        axs[idx].axvspan(np.log(self.wave_min), np.log(self.limit_wave[limit_id]), \
                         alpha=0.3,  color= 'lightyellow')

        # print(xmin, xmax)
        xmax = np.log(self.blended_bounds[limit_id])
        axs[idx].plot(self.log_wave, self.trace.model_blended, c = 'b', alpha = 1, label='fit')
        axs[idx].plot(self.log_wave, self.trace.cont_norm_flux, 'k', lw = 0.6, alpha = 0.3)

        if self.trace.blended_fit[limit_id]:
                
            p0 = self.trace.blended_p0[limit_id]
            p_fit = self.trace.blended_params[limit_id]
            p_smth = smth_limit_param[limit_id]
            model_est, mask = self.eval_blended(p0, limit_id)
            # model_fit, _ = self.eval_blended(p_fit, limit_id)
            model_smth, _ = self.eval_blended(p_smth, limit_id)

            x, y = self.log_wave[mask], trace.model_cont[mask]
            ymin = np.median(y[:50])
            xmin = np.log(self.limit_wave[limit_id]) - offset[idx]
            # hb = self.trace.hb[limit_id]
            # axs[idx].scatter(*hb.T, c = 'b', alpha = 0.5)
            # hb2 = self.trace.hull[limit_id]
            # axs[idx].scatter(*hb2.T, c = 'r')
            y0, slope_mid, x_mid, s0, s1 = p_smth
            y0_fit, _, x_mid_fit, _, _ = p_fit


            # y0, slope_mid, x_mid, s0, s1 = self.trace.params_est[limit_id]
            # f_init = self.sigmoid_fn(x, y0, slope_mid, x_mid, s0, s1)
            # axs[idx].plot(x, f_init, c = 'g', alpha = 0.2)
#                         c, b, half_y = self.get_init_sigmoid_estimation(hb)
            ymin = -y0
            xmax = x[np.where(model_smth > -0.0001)[0][0]]
            # xmax = self.trace.blended_control_points[limit_id][0][-3]
            # axs[idx].plot(x, model_est, c = 'k', alpha = 1, label='est')
            axs[idx].plot(x, model_smth, c = 'r', alpha = 1, label='smth')

            axs[idx].plot(x_mid, -y0/2., 'go')
            axs[idx].plot(x_mid_fit, -y0_fit/2., 'go')

            axs[idx].axhline(-y0, c = color[idx], linestyle = ":")   
            axs[idx].axvline(x[0], c = color[idx],  linestyle = "-")                
            axs[idx].set_title('a_fit {:.3f} | a_smth {:.3f}'.format(-y0_fit, -y0))

        # axs[idx].plot(self.log_wave, spec.cont, 'k', lw = 1)
        axs[idx].set_ylim(ymin * 1.1, abs(ymin* 0.1))
        axs[idx].set_xlim(xmin, xmax)
        axs[idx].legend(loc = 4)
    for ax in axs:
        ax.set_xticklabels([int(np.around(np.exp(label),-2)) for label in ax.get_xticks().tolist()])
    # # f.suptitle(name)
    if out: return self

    
def plot_bk(model, axs,  mode='blend', color = ['lightsalmon', 'lightgreen']):
    limit_ids=[1, 2]
    limit_names = ['Balmer', 'Paschen']
    for idx, limit_id in enumerate(limit_ids):
        axs[idx].axvspan(np.log(model.limit_wave[limit_id]), np.log(model.blended_bounds[limit_id]),\
                        alpha=0.3, color=color[idx], label=limit_names[idx])
        axs[idx].axvspan(np.log(model.wave_min), np.log(model.limit_wave[limit_id]), \
                        alpha=0.3,  color= 'lightyellow')
        axs[idx].axvline(np.log(model.limit_wave[limit_id]), c = color[idx],  linestyle = "-")                

        # Plot cont flux and model
        if mode == 'blend':
            axs[idx].plot(model.log_wave, model.trace.cont_norm_flux, 'k', lw = 0.6, alpha = 0.3)
            axs[idx].plot(model.log_wave, model.trace.model_blended, c = 'b', alpha = 1, label='fit')                    

def plot_blend(model, limit_id, ax, color = ['lightsalmon', 'lightgreen']):
    offset = 0.04
    ymin = -0.001
    xmin = np.log(model.limit_wave[limit_id]) - offset
    xmax = np.log(model.blended_bounds[limit_id])
    if model.trace.blended_fit[limit_id]:
        cc = model.trace.blended_control_points[limit_id]
        ax.scatter(*cc, c = 'r')
        y0, slope_mid, x_mid, s0, s1 = model.trace.blended_params[limit_id]
        ymin = -y0
        xmax = cc[0][-1]
        # xmax = x[np.where(y > -0.0001)[0][0]]
        # axs[idx].plot(x, model, c = 'r', alpha = 1)
        ax.plot(x_mid, -y0/2., 'go')
        ax.axhline(-y0, c = color[limit_id-1], linestyle = ":")
    return ymin, xmin, xmax   



def plot_limits(FList, TList, LList, grid=None, out=False, C_M=0.0, O_M=0.0):
    # r_pix, c_pix = 7, 2
    r_pix, c_pix = 20, 4
    trace = AlexContinuumModelTrace()
    self = AlexContinuumModel(trace=trace)  
    self.init_wave(grid.wave)
    color2 = ['lightsalmon', 'lightgreen']
    limit_ids = [1, 2]
    nPlot = len(FList) * len(TList) * len(LList)
    f, axss = plt.subplots(nPlot, 2, figsize=((r_pix, c_pix * nPlot)), facecolor = 'w')
    i = 0
    for Fe_H in FList:
        for T_eff in TList:
            for log_g in LList:
                axs = axss if nPlot == 1 else axss[i]
                i += 1
                name = f'[Fe/H] = {Fe_H} | T_eff = {T_eff}K | log(g) = {log_g} | [C/M] = {C_M} | [O/M] = {O_M}'
                spec = grid.get_nearest_model(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g, C_M=C_M, O_M=O_M)
                if spec is None: print('no spec'); continue
                params = self.fit(spec)  
                self.normalize(spec, params)
                plot_bk(self, axs,  mode='blend', color=color2)
                for idx, limit_id in enumerate(limit_ids):                                           
                    # mask = self.cont_eval_masks[limit_id]
                    # y = trace.model_cont[mask]
                    ymin, xmin, xmax = plot_blend(self, limit_id, axs[idx], color =color2)
                    axs[idx].plot(self.log_wave, self.trace.cont_norm_flux, 'k', lw = 0.6, alpha = 1)
                    axs[idx].set_ylim(ymin * 1.1, abs(ymin * 0.1))
                    axs[idx].set_xlim(xmin, xmax)
                    axs[idx].legend(loc = 4)
#                     axs[idx].set_title(break_name[idx])
                for ax in axs:
                    ax.set_xticklabels([int(np.around(np.exp(label),-2)) for label in ax.get_xticks().tolist()])
                axs[0].set_title(name)
    # f.suptitle(name)
    if out: return self

def plot_jac(FList, TList, LList, grid=None, out=False, C_M=0.0, O_M=0.0):
    # r_pix = 7
    # c_pix = 2
    r_pix, c_pix = 20, 4
    trace = AlexContinuumModelTrace()
    self = AlexContinuumModel(trace=trace)  
    self.init_wave(grid.wave)

    offset = [0.04, 0.04]
    # offset_y = [0.02, 0.0]
    limit_ids = [1, 2]
    limit_names = ['Balmer', 'Paschen']
    color = ['lightsalmon', 'lightgreen']
    
    nPlot = len(FList) * len(TList) * len(LList)
#     f, axss = plt.subplots(nPlot, 3, figsize=((20,5 * nPlot)), facecolor = 'w')
    f, axss = plt.subplots(nPlot, 2, figsize=((r_pix, c_pix * nPlot)), facecolor = 'w')
    i = 0
    for Fe_H in FList:
        for T_eff in TList:
            for log_g in LList:
                axs = axss if nPlot == 1 else axss[i]
                i += 1
                name = f'[Fe/H] = {Fe_H} | T_eff = {T_eff}K | log(g) = {log_g} | [C/M] = {C_M} | [O/M] = {O_M}'
                spec = grid.get_nearest_model(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g, C_M=C_M, O_M=O_M)
                if spec is None: 
                    print('no spec')
                    continue
                params = self.fit(spec)  
                self.normalize(spec, params)
                for idx, limit_id in enumerate(limit_ids):
                    # Plot Region Shading
                    axs[idx].axvspan(np.log(self.limit_wave[limit_id]), np.log(self.blended_bounds[limit_id]),\
                                    alpha=0.3, color=color[idx], label=limit_names[idx])
                    axs[idx].axvspan(np.log(self.wave_min), np.log(self.limit_wave[limit_id]), \
                                    alpha=0.3,  color= 'lightyellow')
                    # Plot cont flux and model
                    axs[idx].plot(self.log_wave, self.trace.cont_norm_flux, 'k', lw = 0.6, alpha = 0.3)
                    axs[idx].plot(self.log_wave, self.trace.model_blended, c = 'b', alpha = 1, label='fit')                    
                               
                    mask = self.cont_eval_masks[limit_id]
                    x, y = self.log_wave[mask], trace.model_cont[mask]
                    ymin = np.median(y[:50])
                    xmin = np.log(self.limit_wave[limit_id]) - offset[idx]
                    xmax = np.log(self.blended_bounds[limit_id])
                    if self.trace.blended_fit[limit_id]:
                        cc = self.trace.blended_control_points[limit_id]
                        axs[idx].scatter(*cc, c = 'r')
                        y0, slope_mid, x_mid, s0, s1 = self.trace.blended_params[limit_id]
                        ymin = -y0
                        xmax = cc[0][-1]
                        # xmax = x[np.where(y > -0.0001)[0][0]]
                        # axs[idx].plot(x, model, c = 'r', alpha = 1)
                        axs[idx].plot(x_mid, -y0/2., 'go')
                        axs[idx].axhline(-y0, c = color[idx], linestyle = ":")   
                    axs[idx].plot(self.log_wave, self.trace.cont_norm_flux, 'k', lw = 0.6, alpha = 1)
                    axs[idx].axvline(x[0], c = color[idx],  linestyle = "-")                
                    axs[idx].set_ylim(ymin * 1.1, abs(ymin* 0.1))
                    axs[idx].set_xlim(xmin, xmax)
                    axs[idx].legend(loc = 4)
#                     axs[idx].set_title(break_name[idx])
                for ax in axs:
                    ax.set_xticklabels([int(np.around(np.exp(label),-2)) for label in ax.get_xticks().tolist()])
                axs[0].set_title(name)
    # f.suptitle(name)
    if out: return self

def plot_legendre(FList, TList, LList, grid=None, out=False, C_M=0.0, O_M=0.0):
    # r_pix = 14
    # c_pix = 2.8
    self = AlexContinuumModel(debug = True)    
    offset = [0, 0.02, 0.02]
    # break_name = ['3000A', 'Balmer', 'Paschen']
    seg_name = ['segment 1', 'segment 2', 'segment 3']
    xmin = np.log(np.array([3000, 3600, 8100]))
    xmax = np.log(np.array([3646, 8205, 14000]))

    nPlot = len(FList) * len(TList) * len(LList)
#     f, axss = plt.subplots(nPlot, 3, figsize=((20,5 * nPlot)), facecolor = 'w')
    f, axss = plt.subplots(nPlot, 3, figsize=((r_pix, c_pix * nPlot)), facecolor = 'w')
    i = 0
    for Fe_H in FList:
        for T_eff in TList:
            for log_g in LList:
                axs = axss if nPlot == 1 else axss[i]
                i += 1
                name = f'Fe_H = {Fe_H} | T_eff = {T_eff}K | log(g) = {log_g}'
                spec = grid.get_nearest_model(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g, C_M=C_M, O_M=O_M)
                if spec is None: continue
                log_flux, log_cont = self.prepare(spec)
                fits, params = self.fit_legendre(log_cont)
                model_cont = self.eval_legendre(fits)
                norm_flux, norm_params = self.get_norm_flux_n_params(spec)
                for idx, limit_id in enumerate([0, 2, 4]): 
                    axs[idx].plot(self.log_wave, log_flux, 'k', lw = 0.5, alpha = 0.8, label = seg_name[idx])
                    axs[idx].plot(self.log_wave, model_cont, 'r', lw = 2)
                    # axs[idx].axvline(x[0], c = "cyan", linestyle = ":")                
                    axs[idx].set_ylim(min(model_cont - 1), None)
                    axs[idx].set_xlim(xmin[idx], xmax[idx])
                    axs[idx].legend(loc = 4)
#                     axs[idx].set_title(break_name[idx])
                for ax in axs:
                    ax.set_xticklabels([int(np.around(np.exp(label),-2)) for label in ax.get_xticks().tolist()])
    f.suptitle(name)
    if out: return self


def plot_norm(FList, TList, LList, grid=None, out=False, C_M=0.0, O_M=0.0):
    # r_pix = 14
    # c_pix = 2.8
    self = AlexContinuumModel(debug = True)    
    offset = [0, 0.02, 0.02]
    # break_name = ['3000A', 'Balmer', 'Paschen']
    seg_name = ['segment 1', 'segment 2', 'segment 3']
    xmin = np.log(np.array([3000, 3600, 8100]))
    xmax = np.log(np.array([3646, 8205, 14000]))

    nPlot = len(FList) * len(TList) * len(LList)
#     f, axss = plt.subplots(nPlot, 3, figsize=((20,5 * nPlot)), facecolor = 'w')
    f, axss = plt.subplots(nPlot, 3, figsize=((r_pix, c_pix * nPlot)), facecolor = 'w')
    i = 0
    for Fe_H in FList:
        for T_eff in TList:
            for log_g in LList:
                axs = axss if nPlot == 1 else axss[i]
                i += 1
                name = f'Fe_H = {Fe_H} | T_eff = {T_eff}K | log(g) = {log_g}'
                spec = grid.get_nearest_model(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g, C_M=C_M, O_M=O_M)
                if spec is None: continue
                log_flux, log_cont = self.prepare(spec)
                fits, params = self.fit_legendre(log_cont)
                model_cont = self.eval_legendre(fits)
                norm_flux, norm_params = self.get_norm_flux_n_params(spec)
                norm_cont = np.zeros_like(norm_flux)
                for idx, limit_id in enumerate([0, 2, 4]): 
                    axs[idx].plot(self.log_wave, norm_flux, 'k', lw = 0.5, alpha = 0.8, label = seg_name[idx])
                    axs[idx].plot(self.log_wave, norm_cont, 'r', lw = 2)
                    # axs[idx].axvline(x[0], c = "cyan", linestyle = ":")                
                    # axs[idx].set_ylim(min(model_cont - 1), None)
                    axs[idx].set_xlim(xmin[idx], xmax[idx])
                    axs[idx].legend(loc = 4)
#                     axs[idx].set_title(break_name[idx])
                for ax in axs:
                    ax.set_xticklabels([int(np.around(np.exp(label),-2)) for label in ax.get_xticks().tolist()])
    f.suptitle(name)
    if out: return self

def plot_sigmoid():
    x = np.linspace(-1.1, 1.1, 40)
    a = 1
    b = 1
    c = 0.5
    r0 = 0.7
    r1 = 0.7

    x0 = c - 1 / (2 * b)
    x1 = c + 1 / (2 * b)
    beta0  = 2 * b / r0
    alpha0 = r0 / (2 * np.e)   
    beta1  = 2 * b / r1
    alpha1 = r1 / (2 * np.e)

    t0 = x0 + 1 / beta0
    t1 = x1 - 1 / beta1
    i0 = (x <= t0)
    i1 = (x >= t1)
    im = (x > t0) & (x < t1)

    arg0 = beta0 * (x[i0] - x0)
    arg1 = -beta1 * (x[i1] - x1)

    yl = alpha0 * np.exp(arg0)
    yr = 1 - alpha1 * np.exp(arg1)
    ym = b * (x - c) + 0.5
    # yy = a * (y - 1)
    f, ax = plt.subplots(1, facecolor='w')

    ax.plot(c, 0.5, 'go')
    ax.axvline(c, ymin=0, ymax = 0.5, c = 'g', linestyle = ":")

    ax.plot(x, ym, 'k-.')
    ax.plot(x[i0], yl, '-r')
    ax.plot(x[i1], yr, '-b')

    ax.axvline(x0, c = 'cyan', linestyle = ":")
    ax.axvline(x1,  c = 'cyan', linestyle = ":")
    ax.axvline(t0, c = 'orange', linestyle = ":")
    ax.axvline(t1, c = 'orange', linestyle = ":")
    ax.axhline(0, c = 'k', linestyle = ":")
    ax.axhline(1, c = 'k', linestyle = ":")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1,1.1)
    ax.set_xticks([x0, t0, c, t1, x1])
    ax.set_xticklabels(['$x_0$', '$t_0$','$c$', '$t_1$', '$x_1$'])


def plot_seg(yy, ax = None, x_offset = 0.05, legend=True, ymin_mode = 0):
    xmin, xmax = np.log(3000), np.log(14000)
    if ymin_mode == 0:
        ymin, ymax = np.max([0, np.quantile(yy, 0.02)]), np.max(yy)
    else:
        ymin, ymax = np.min(yy), np.max(yy)
    ymaxx = ymax + 0.3
    xBalmer, xPaschen = np.log([3647.04, 8205.96])
    x1Balmer, x1Paschen = np.log([4200, 12000])


    # ax.vlines([x1Balmer, x1Paschen] , color = 'b', ymin = ymin, ymax = ymaxx)
    ax.vlines([xmin, xmax] , color = 'y', ymin = ymin, ymax = ymaxx)
    ax.axvspan(xmin, xmax, alpha=0.5, color='lightyellow', label='Fitting')

    ax.vlines([xBalmer, x1Balmer] , color = 'r', ymin = ymin, ymax = ymaxx)
    ax.axvspan(xBalmer, x1Balmer, alpha=0.5, color='r', label='Balmer')

    cP = 'g'
    ax.vlines([xPaschen, x1Paschen] , color = cP, ymin = ymin, ymax = ymaxx)
    ax.axvspan(xPaschen, x1Paschen, alpha=0.5, color= cP, label='Paschen')
    ax.set_xlim(xmin - x_offset, xmax + x_offset)
    ax.set_ylim(ymin, ymaxx)
    ax.set_xticks([xmin, xBalmer, x1Balmer, xPaschen, x1Paschen, xmax])
    ax.set_xticklabels(['3000', '3647', '4200', '8205', '12000', '14000'])
    ax.set_xlabel('wave length ($A$)')
    if ymin_mode == 0 :
        ax.set_ylabel('ln(flux)')
    else:
        ax.set_ylabel('Normalized ln(flux)')
        
    if legend: ax.legend()
    # ax.set_yscale('log', basey=np.e)
    # ax.set_xscale('log', basex=np.e)


def plot_svd_timing(sList, ts, ax=None):
    if ax is None: ax = plt.gca()
    ax.plot(sList, ts, 'ro-')


def plot_svd_s(SSList, ax=None):
    if ax is None: ax = plt.gca()
    
    for ss in SSList:
        ss0 = ss / np.sum(ss) 
        ax.plot(np.arange(len(ss0)), ss0, 'o-')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Log Singular Value Ratio')
    ax.set_ylabel('Log Pixel Size')    
#     ax.legend()


def plot_svd_s(SSList, labels = None, ax=None):
    if ax is None: f, ax = plt.subplots(1, 1, facecolor='w')
    if labels is None: labels = np.arange(len(SSList))
    colors = ['k', 'r','orange','g','b','purple']
    for ii, ss in enumerate(SSList):
        ss0 = ss / np.sum(ss) 
        ax.plot(np.arange(len(ss0)), ss0, 'o-',label = labels[ii], color=colors[ii], ms=2, lw=0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel('Log Singular Value Ratio')
    ax.set_xlabel('Log Rank')    

    if labels is not None: ax.legend()


def plot_svd_sdiff(SSList, labels = None, ax=None):
    if ax is None: ax = plt.gca()
    if labels is None: labels = np.arange(len(SSList))
    ax.set_xscale('log')
    colors = ['k', 'r','orange','g','b','purple']


    for ii, ss in enumerate(SSList):
        if np.sum(ss) == 0: 
            ss0 = ss
        else:
            ss0 = ss / np.sum(ss) 
        ax.plot(np.arange(len(ss0)), ss0, label = labels[ii],color=colors[ii])

    # ax.set_yscale('log')

    ax.set_ylabel('|$\delta$ Singular Valur Error|')    
    ax.set_xlabel('Log Rank')

    if labels is not None: ax.legend()