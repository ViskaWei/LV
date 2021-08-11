# SPARSE PRINCIPAL COMPONENTS OF SPECTRA

Principal Component Analysis will decompose a complex vector into a linear combination of orthogonal feature vectors, therefore it has not been used much for stellar spectroscopy. However, once we realize that that if we take the log of the stellar flux, it becomes an additive process, we can use PCA to decompose spectra. In Figure 1, we show the typical normalized stellar spectrum at high and low temperature. The overall shape and spectral line profiles are drastically different for the high and low T stars. The yellow lines denote the Ca triplet at 8498A, 8542A, and 8662A, which are only evident in the low temperature or cool stars. Meanwhile, the Paschen series in the green region are only evident in the high temperature or hot stars. Therefore, we build separate PCA for hot and cool stars and evaluate their eigenvectors (eigv) as shown in Figure 2 and Figure 3. 

![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/highlowFlux.png?raw=true)
> Figure 1: Normalized model spectrum of typical stars at high and low T. The Ca triplet (yellow) only appears in cool stars while the Paschen series (green) shows in hot stars)

![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/LOWTPC01.png?raw=true)
> Figure 2: Top 2 eigenvectors for 3,000K < T < 7,000K

![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/HighTPC01.png?raw=true)
> Figure 3: Top 2 eigenvectors for 12,000K < T < 30,000K


It is clear that we have a case of sparse subspaces where the data occupies an intersection of two very low dimensional manifolds, one for the cool stars and one for the hot stars. Each manifold are characterized with different set of spectral lines. In particular for hot stars (Figure 3), pca[1] picks up the sharp lines at $8875\AA, 9000\AA, 9250\AA, 9600\AA$, their strength changing with Teff, while the sombrero-shaped profiles in pca[2] (right panel) will change their widths. Since higher surface gravity log(g) broadens spectral lines, pca[2] is a strong indicator for the surface gravity. 
This effect can be nicely seen in Figure 4.


![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/LgTep01.png?raw=true)
> FIgure 4: Scatter plots of the top 2 pca coefficients of high T stars, colored by T (right) and log surface gravity (left)

## Hierarchical Scheme
While the last figure looks quite good, the real task is much more complex. We need to determine many other stellar parameters, various elemental abundances (C,He,O, etc). In order to do that, we will have to use deep networks to learn the detailed necessary features.
Subdividing spectra in different domains of temperature and gravity will simplify the problem, as the relevant features are drastically different between low to high temperature stars. Then for each cluster we train deep neural nets such as autoencoders to infer finer stellar parameters. In short, we divide and conquer. 
The selected features should also be stable under noise as certain regions will have strong sky lines in real observations. We expect the engineered sparse features take those into account as well. 

The hierarchical scheme not only reduces the complexity of recovering the embedded structure but also reduces computational cost and hardware requirements for the training. More importantly, the optimal sparsification scheme will increase the signal to noise in the data and improve the performance of inference and learning.

##Feature engineering
Given that most of the pixels are noise and few informative pixels live in a union of subspaces, engineering informative features is tricky.  Currently, we are exploring several approaches to rank the pixels by their importance in parameter inference. That is to measure how sensitive one pixel is to the parameter changes. For example, the presence of the Ca triplet can constrain the temperature parameter to be lower than 5000K.  

Following Elhamifar and Vidal's work on spare subspace clustering, we would first build orthogonal sparse basis for hot stars only, and for cool stars only. We then deproject these subspaces from the data and build a PCA basis on the remaining subspaces to get all other features.



