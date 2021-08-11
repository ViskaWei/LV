# SPARSE PRINCIPAL COMPONENTS OF SPECTRA

Principal Component Analysis will decompose a complex vector into a linear combination of orthogonal feature vectors, therefore it has not been used much for stellar spectroscopy. However, once we realize that that if we take the log of the stellar flux, it becomes an additive process, we can use PCA to decompose spectra. In Figure \ref{fig:highlowflux}, we show the typical normalized stellar spectrum at high and low temperature. The overall shape and spectral line profiles are drastically different for the high and low T stars. The yellow lines denote the Ca triplet at 8498A, 8542A, and 8662A, which are only evident in the low temperature or cool stars. Meanwhile, the Paschen series in the green region are only evident in the high temperature or hot stars. Therefore, we build separate PCA for hot and cool stars and evaluate their eigenvectors (eigv) as shown in Figure \ref{fig:lowTpc01} and Figure \ref{fig:highTpc01}. 


![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/LgTep01.png?raw=true)

![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/HighTPC01.png?raw=true)
![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/LOWTPC01.png?raw=true)
![alt text](https://github.com/ViskaWei/LV/blob/master/nb/img/highlowFlux.png?raw=true)

