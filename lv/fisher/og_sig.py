
RVSim = collections.namedtuple('RVSim',['name','N','NT','rv','RV','SN','S2','X2'])

def getSigmaSim(NL,NV,N,NT,ss,T):
    #--------------------------------------------
    # create simulations of the analysis
    # estimate sigma from the data, and 
    # return a normalized variate (rv-RV)/sig
    # NL:  noise level
    # NV:  number of random velocities
    # N :  number of independent realizations
    # NT:  index of template from the stencil
    # ss:  the high resolution spectrum
    # T :  the template array over the stencil
    # Returns:
    # S : list of simulation outputs, one item for each rv
    #--------------------------------------------
    rvarr  = 200*(np.random.rand(NV)-0.5)
    temp  = T[NT]
    #
    S = []
    for rv in rvarr:
        ssm   = getModel(ss,rv)
        wwm   = temp.wwm
        varm  = getVar(ssm,temp.skym)
        vmobs = NL**2*varm

        RV = []
        SN = []
        A2 = []
        S2 = []
        X2 = []

        for n in range(N):
            ssobs = ssm + getNoise(vmobs)    
            sn  = getSN(ssobs) 
            rvo = getRV1(ssobs,vmobs,temp)
            F2  = getFisherMatrix2(rvo,ssobs,vmobs,temp)
            iF2 = sp.linalg.inv(F2)
            sa2 = iF2[0,0]
            sg2 = iF2[1,1]

            RV.append(rvo)
            SN.append(sn)
            A2.append(sa2)
            S2.append(sg2)
            X2.append((rvo-rv)/np.sqrt(sg2))

        sim = RVSim(name,N,NT,rv,np.array(RV)-rv,SN,S2,X2)
        S.append(sim)

    return S

def mergeSims(S):
    name = S[0].name
    N  = S[0].N
    NT = S[0].NT
    rv = []
    RV = []
    SN = []
    A2 = []
    S2 = []
    X2 = []    
    for s in S:
        rv.append(s.rv)
        RV.append(s.RV)
        SN.append(s.SN)
        S2.append(s.S2)
        X2.append(s.X2)                
    sim = RVSim(name,N,NT,rv,RV,SN,S2,X2)
    return sim

def showSigmaHisto(s):
    rv = s.rv
    if (type(s.rv)==type([0,1])):
        rv = 0
    
    drv = np.mean(s.RV)
    srv = np.std(s.RV)
    sn0 = np.mean(s.SN)
    sg0 = np.sqrt(np.mean(s.S2))

    plt.figure(figsize=(12,6))
    nbins = 30
    bins = np.linspace(-3,3,nbins+1)
    #plt.hist(X2,bins,alpha=0.25)
    if (s.N==len(s.X2)):
        plt.hist(s.X2,bins)
    else:
        for x2 in s.X2:
            plt.hist(x2,bins, alpha=0.35)
    x = np.linspace(-3,3,31)
    y = s.N*6/nbins*np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
    ts = s.name+'    '
    #ts = ts + '  rv={:6.3f} km/s,  '.format(rv)
    ts = ts + 'dRV={:6.4f}+/-'.format(drv)
    ts = ts + '{:6.4f} km/s,  '.format(srv)
    ts = ts + 'sigz={:6.4f} km/s,  '.format(sg0)
    ts = ts + 'S/N={:3.1f}  '.format(sn0)
    plt.plot(x,y)
    plt.title(ts)
    plt.xlabel('sigma')
    plt.ylabel('N')
    fname = 'figs/F-sigmahisto-'+s.name+'-SNR{:03d}.png'.format(np.int(sn0))
    plt.savefig(fname)
    plt.show()


def getSigmaTemplate(rv,nlarr,ss,T):
    ssm = getModel(ss,rv)
    wwm = T[4].wwm
    varm= getVar(ssm,T[4].skym)

    Q = []
    for temp in T:
        Q.append(temp.name)

    O = []
    for NL in nlarr:
        vmobs = NL**2*varm
        ssobs = ssm + getNoise(vmobs)    
        sn  = getSN(ssobs) 
        for NT in range(len(T)):
            temp  = T[NT]
            rvo = getRV1(ssobs,vmobs,temp)
            F2  = getFisherMatrix2(rvo,ssobs,vmobs,temp)
            iF2 = sp.linalg.inv(F2)
            sg2 = iF2[1,1]
            O.append((rv,sn,NT,rvo,np.sqrt(sg2)))

    O = np.array(O)
    name = T[4].name
    fname1 = 'csv/'+name+'-sigma.csv'
    fname2 = 'csv/'+name+'-names.csv'
    np.savetxt(fname1, O, delimiter=",",fmt="%10.6f")
    np.savetxt(fname2, Q, delimiter=",",fmt="%s")
    return O