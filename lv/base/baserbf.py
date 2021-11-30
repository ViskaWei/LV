idx1, idx2 = b.get_pdx_idx([0,0,1,0,0]), b.get_pdx_idx([0,0,3,0,1])
interpFlux = b.rbf([[ 0, 0,  2,  0,  0]])[0]
plt.plot(b.wave, b.flux[idx1], label = b.para[idx1])
plt.plot(b.wave, interpFlux, label= 0.5*(b.para[idx1]+b.para[idx2]))
plt.plot(b.wave, b.flux[idx2], label = b.para[idx2])
plt.legend()

idx1, idx2 = b.get_pdx_idx([0,0,0,0,1]), b.get_pdx_idx([0,0,0,0,2])
interpFlux = b.rbf([[ 0, 0,  0,  0,  1.5]])[0]
plt.plot(b.wave, b.flux[idx1], label = b.para[idx1])
plt.plot(b.wave, interpFlux, label= 0.5*(b.para[idx1]+b.para[idx2]))
plt.plot(b.wave, b.flux[idx2], label = b.para[idx2])
plt.legend()