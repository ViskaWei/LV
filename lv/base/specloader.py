import os
import urllib
import bz2

import numpy as np


#-----------------------------------------------------------------------------------
# generate the url and the pathname from the parameters to download the spectra
# from the STScI website
#-----------------------------------------------------------------------------------
def fmt(x):
    y = np.round(np.abs(10*x)+0.2).astype(np.int32)
    z = '{:+03.0f}'.format(y).replace('+','p')
    if (np.sign(x)<0):
        z = z.replace('p','m')
    return z

def fmn(x):    
    return '{:02d}'.format(np.floor(x).astype(np.int32))

def ffm(x):
    return '{:+4.2f}'.format(x)   
    
def getfilename(m,t,g,c,a,R):
    #---------------------------------------
    # get the long filename for the website
    #---------------------------------------
    fname = 'am'+fmt(m)+'c'+fmt(c)+'o'+fmt(a)+'t'+fmn(t)+'g'\
        +fmn(10*g)+'v20modrt0b'+'{:d}'.format(R)+'rs.asc.bz2'
    return fname

def getname(m,t,g,c,a):
    #----------------------------------
    # get short name for the spectrum
    #----------------------------------
    fname = 'T'+fmn(t)+'G'+fmn(10*g)+'M'+fmt(m)+'A'+fmt(a)+'C'+fmt(c)
    return fname
    
def geturl(m,t,g,c,a,R):
    #----------------------------------------
    # build the url for the STScI site
    #----------------------------------------    
    url = 'https://archive.stsci.edu/missions/hlsp/bosz/ascii/insbroad_'+'{:06d}'.format(R)+'/'
    url = url + 'metal_'+ffm(m) + '/carbon_'+ffm(c) + '/alpha_'+ffm(a)
    url = url + '/'+ getfilename(m,t,g,c,a,R)
    return url

# example URL
#url = 'https://archive.stsci.edu/missions/hlsp/bosz/ascii/insbroad_020000/metal_-1.00/carbon_+0.00/alpha_+0.25/amm10cp00op03t10500g25v20modrt0b20000rs.asc.bz2'

#--------------------------
# get the data from a url
#--------------------------
def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)

def getSpectrum(m,t,g,c,a,R=50000, lb=6250, ub=9750):
    #-----------------------------------------------------------
    # get a spectrum given the 5 parameter values
    # Input:
    #   m : M/H
    #   c : C/H
    #   a : ALPHA/H
    #   t : Teff
    #   g : log g
    # Output:
    #   o[:,0] is the wavelength in \AA
    #   o[:,1] is the flux
    #   o[:,2] is the theoretical envelope
    #-----------------------------------------------------------
    # The spectrum extends a bit beyond the MR range
    #     6250 A < lambda < 9750 A
    # If the parameter values are not on the grid, returns an 
    # empty vector and writes an error message
    #-----------------------------------------------------------
    # set the resolution in the BOSZ convention, 
    # corresponding to R=100,000


    #--------------------------------------
    # fbuild the url and fetch the data
    #--------------------------------------
    url = geturl(m,t,g,c,a,R)
    print(url)
    download_url(url,'tmp/','test.bz2')
    #
    filepath    = 'tmp/test.bz2'
    newfilepath = 'tmp/test.txt'
    #--------------------------
    # unzip the spectrum
    #--------------------------
    with open(newfilepath, 'w') as new_file, bz2.BZ2File(filepath, 'rb') as file:
        data = file.read().decode("utf-8") 
        new_file.write(data)
    d = np.genfromtxt(newfilepath, delimiter=' ')
    #----------------------------------------------------------------------
    # select the wavelength range for the medium resolution spectrograph
    #----------------------------------------------------------------------
    iw = (d[:,0]>=lb) & (d[:,0]<=ub)
    o  = d[iw,:]
    return o


