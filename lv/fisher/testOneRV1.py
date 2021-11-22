def showSpec(wwm,ssm, sso):
    #------------------------------------
    # show the input spectrum as well
    # as the noisy version
    #------------------------------------
    SN = getSN(sso)
    plt.figure(figsize=(15,4))
    plt.plot(wwm, sso, lw=0.2)
    plt.plot(wwm,ssm,lw=1)
    plt.title(sname+'    S/N={:5.1f}'.format(SN))
    plt.savefig('figs/F-rvspec.png')
    plt.show() 
    
def lorentz(x, a,b,c,d):
    return a/(1+(x-b)**2/c**2) + d

def makeRVGuess(ssobs,vmobs,temp):
    #--------------------
    # get initial guess
    #--------------------
    v0  = np.linspace(-500, 500, 31)
    y0 = []
    for v in v0:
        y0.append(-llh0(v,ssobs,vmobs,temp))
    pp = [200,0,100,0.5*(y0[0]+y0[-1])]
    bb=((0,-300,100,min(y0)),(5000,300,300,min(y0)+4*(max(y0)-min(y0))))
    pp, pcov = curve_fit(lorentz, v0, y0, pp, bounds=bb)
    
    return pp[1]


def llh0(rv,sobs,varm,t):
    nu,phi,chi = llh1(rv,sobs,varm,t)
    return -nu

def llh1(rv,sobs,varm,t):
    #--------------------------------------------------
    # A variant of the log likelihood for the model fitting.
    # Also returns phi and chi for the Fisher matrix,
    # not to be used for the rv fitting, but rather for 
    # testing and error estimation.
    #--------------------------------------------------
    # Input
    #  rv : radial velocity in km/s
    #  sobs: the observed noisy spectrum
    #  varm: the variance of the observed data
    #  ss : noiseless template spectrum in h-pix
    #  t  : template array with columns (wwm,ssm,skym)
    # Output
    #  nu: the negative log likelihood, to be minimized
    # phi:  the cross term in the llh
    # chi:  the model^2 term in the llh
    #--------------------------------------------------
    # build the resampled, shifted template and variance
    # from the observed spectrum 
    #------------------------------------
    m = getModel(t.sst,rv)  
    #---------------------------------
    # clip to the spectrograph range
    #---------------------------------
    tm  = m[t.iwm]
    vm  = varm[t.iwm]
    obs = sobs[t.iwm]
    #----------------------------------
    # build the significance function
    #----------------------------------
    phi = np.sum(obs*tm/vm)
    chi = np.sum(tm*tm/vm)
    nu  = phi/np.sqrt(chi)    
    return nu,phi,chi


def testOneRV1(ss,temp,rv,NL,plot=1):
    ssm   = getModel(ss,rv)
    wwm   = temp.wwm
    varm  = getVar(ssm,temp.skym)
    noise = getNoise(varm)  
    ssobs = ssm + NL*noise
    vmobs = varm*NL**2
    SN    = getSN(ssobs)
    RV    = getRV1(ssobs,vmobs,temp)

    F     = getFisherMatrix(RV,ssobs,vmobs,temp)
    det   = F[0][0]*F[1][1]-F[1][0]**2
    print('sigma_z=',np.sqrt(F[0][0]/det))
    
    if (plot==1):
        showSpec(wwm,ssm,ssobs)
        showRV1(rv, RV, ssobs,vmobs, temp)
    else:
        print('SN=',SN, 'RV=',RV)
    return RV, F

#----------------------------------------------------------
# testing functions for TEST 1-2 to plot the LLH as a function of rv
#----------------------------------------------------------
def showRV1(rv, RV, ssobs,vmobs,temp):
    
    v1  = np.linspace(-300, 300,101)
    v2  = np.linspace(rv-6, rv+6,25)
    
    y1 = []
    y2 = []
    for v in v1:
        y1.append(-llh0(v,ssobs,vmobs,temp))
    for v in v2:
        y2.append(-llh0(v,ssobs,vmobs,temp))
        
    SN = getSN(ssobs)
    sigz2 = getFisher1(rv,ssobs,vmobs,temp)

    plt.figure(figsize=(15,6))
    plt.plot(v1,y1,'g.-',markersize=7)    
    plt.plot(rv,-llh0(rv,ssobs,vmobs,temp),'ro')
    plt.plot(RV,-llh0(RV,ssobs,vmobs,temp),'ko')
    ts = 'rv={:6.4f} km/s,  '.format(rv)+ 'RV={:6.4f} km/s,  '.format(RV)
    ts = ts + 'S/N={:3.1f},  '.format(SN) + 'sigz={:6.4f} km/s,  '.format(np.sqrt(sigz2))
    plt.title(ts)
    plt.grid()
    plt.ylim((min(y1),min(y1)+(max(y1)-min(y1))*1.5))
    plt.legend(['llh','rv(true)','RV(est)'])
    ax = plt.gca()
    ins = ax.inset_axes([0.1,0.45,0.4,0.5])
    ins.plot(v2,y2,'g.-',markersize=7)
    ins.plot(rv,-llh0(rv,ssobs,vmobs,temp),'ro')
    ins.plot(RV,-llh0(RV,ssobs,vmobs,temp),'ko')
    ins.grid()
    
    plt.savefig('figs/F-rvfit.png');
    plt.show()
    

def getFisher1(rv,sobs,vobs,t):
    #---------------------------
    # compute the Fisher matrix
    #---------------------------
    m0 = getModel(t.sst,rv)
    m2 = getModel(t.sst,rv+1)
    m1 = getModel(t.sst,rv-1)
    #---------------------------------
    # clip to the spectrograph range
    #---------------------------------
    t0  = m0[t.iwm]
    #-----------------------------
    # get the centered difference
    #-----------------------------
    t1  = 0.5*(m2[t.iwm]-m1[t.iwm])
    vm  = vobs[t.iwm]
    obs = sobs[t.iwm]
    #----------------------------------
    # build the different terms
    #----------------------------------
    psi00 = np.sum(t0*t0/vm)
    psi01 = np.sum(t0*t1/vm)
    psi11 = np.sum(t1*t1/vm)
    phi   = np.sum(obs*t0/vm)
    chi   = np.sum(t0*t0/vm)    
    a0    = phi/chi
    dpsi  = psi00*psi11-psi01**2
    sigz2 = psi00/a0**2/dpsi
    return sigz2


def getFisherMatrix(rv,sobs,vobs,temp):
    #---------------------------
    # compute the Fisher matrix
    #---------------------------
    nu0,phi0,chi0 = llh1(rv,sobs,vobs,temp)
    num,phim,chim = llh1(rv-1,sobs,vobs,temp)
    nup,phip,chip = llh1(rv+1,sobs,vobs,temp)
    f11 = chi0
    f12 = 0.5*(phip-phim)
    f22 = - nu0*(nup+num-2*nu0)+ f12**2
    F = [[f11, f12],[f12,f22]]
    return F


def getRV1(sso,vmo,tm):
    #------------------------------------------------------
    # fit the radial velocity. 
    # Test for the success of the minimization
    # Return nan, if error
    #------------------------------------------------------
    rv0 = makeRVGuess(sso,vmo,tm)
    #print('guess:',rv0)
    out = sp.optimize.minimize(llh0,rv0,  args=(sso,vmo,tm), method="Nelder-Mead")
    if (out.success==True):
        RV = out.x[0]
    else:
        print('getRV error in '+tm.name)
        RV = np.nan
    return RV
