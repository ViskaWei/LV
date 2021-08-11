# SPARSE PRINCIPAL COMPONENTS OF SPECTRA

Principal Component Analysis will decompose a complex vector into a linear combination of orthogonal feature vectors, therefore it has not been used much for stellar spectroscopy. However, once we realize that that if we take the log of the stellar flux, it becomes an additive process, we can use PCA to decompose spectra. In Figure 1, we show the typical normalized stellar spectrum at high and low temperature. The overall shape and spectral line profiles are drastically different for the high and low T stars. The yellow lines denote the Ca triplet at 8498A, 8542A, and 8662A, which are only evident in the low temperature or cool stars. Meanwhile, the Paschen series in the green region are only evident in the high temperature or hot stars. Therefore, we build separate PCA for hot and cool stars and evaluate their eigenvectors (eigv) as shown in Figure 2 and Figure 3. 

![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/highlowFlux.png?raw=true)
> Figure 1: Normalized model spectrum of typical stars at high and low T. The Ca triplet (yellow) only appears in cool stars while the Paschen series (green) shows in hot stars)

![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/LOWTPC01.png?raw=true)
> Figure 2: Top 2 eigenvectors for 3,000K < T < 7,000K

![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/HighTPC01.png?raw=true)
> Figure 3: Top 2 eigenvectors for 12,000K < T < 30,000K


It is clear that we have a case of sparse subspaces where the data occupies an intersection of two very low dimensional manifolds, one for the cool stars and one for the hot stars. Each manifold are characterized with different set of spectral lines. In particular for hot stars (Figure 3), pca[1] picks up the sharp lines at $8875\AA, 9000\AA, 9250\AA, 9600\AA$, their strength changing with $T_{eff}$, while the sombrero-shaped profiles in pca[2] (right panel) will change their widths. Since higher surface gravity ($\log g$) broadens spectral lines, pca[2] is a strong indicator for the surface gravity. 
This effect can be nicely seen in Figure 4.


![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/LgTep01.png?raw=true)
> FIgure 4: Scatter plots of the top 2 pca coefficients of high T stars, colored by T (right) and log surface gravity (left)

