import numpy as np
import re
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict

class KLine():
    def __init__(self, w):
        self.Els = ['0', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                        'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe']
        self.Egs = {"0": [1,11,12,20,26], "1": [1, 8, 12, 20, 26]}
        self.Ws = {"Blue": [3800, 6500, 2300, "Blue"], "RedL": [6300, 9700, 3000, "RedL"], 
                   "RedM": [7100, 8850, 5000, "RedM"], "NIR": [9400, 12600, 4300, "NIR"]}
        self.W = self.Ws[w]
        self.dfLL = None
        self.dfLL26 = None 

        self.load_LL()
        # self.get_SLs()

    def load_LL(self):
        dfLL = pd.read_csv(f"/scratch/ceph/szalay/swei20/LL/kurucz/gfall_vac_{self.W[3]}.csv")
        # self.dfLL = dfLL
        self.dfLL = pd.DataFrame(dfLL[(dfLL["I"]>-3.)].values, columns = dfLL.columns)
        self.dfLL26 = dfLL[(dfLL["Z"] < 27)]

    def prepare_LL(self, w):
        pass

    def get_SLs(self, df=None, lw=5):
        if df is None: df = self.dfLL
        dfI = df.groupby([(df['I'].shift() != df['I']).cumsum()])
        dfII=dfI.size().sort_values(ascending=False)    
        dfII2 = dfII[:np.sum(dfII.reset_index()[0]>lw)]
        SLs=np.zeros((len(dfII2),4))
        for ii, (gdx, _) in enumerate(dfII2.items()):
            dfgps=dfI.get_group(gdx)
            w= dfgps["W"].values
            I = dfgps["I"].values[0]
            Z = dfgps["Z"].values[0]
            SLs[ii,0] = w[0]
            SLs[ii,1] = w[-1]
            SLs[ii,2] = I
            SLs[ii,3] = Z
        self.dfSL = pd.DataFrame(SLs, columns=["W0","W1","I","Z"])
        self.SZs = np.sort(self.dfSL["Z"].unique())

    def plot_dfSL(self, dfSL=None, alpha=1):
        if dfSL is None: dfSL = self.dfSL
        nPlots = len(self.SZs)
        neg= - 1/nPlots
        ax = plt.subplots(1, figsize=(20, 3))[1]
        SLU = dfSL.values
        for i in range(len(SLU)):
            SL = SLU[i]
            c = np.where(self.SZs==SL[3])[0][0]
            cc=cm.gist_rainbow((c+neg)/nPlots)
            ax.axvspan(SL[0], SL[1], ymin=0, ymax=1, lw=2, color=cc, alpha=alpha, label=int(SL[3]))
        self.set_unique_legend(ax,loc=0)
        ax.xaxis.grid(1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("I")
        ax.set_xlim(self.W[0], self.W[1])

    def plot_Z(self, Z, df=None):
        if df is None: df = self.dfLL
        plt.figure(figsize=(16,2), facecolor="w")
        dfZ = df[df["Z"]==Z]
        plt.vlines(dfZ["W"].values, 0,1,color="k")
        plt.title(f"Z={Z} lines")
        # ax=plt.gca()
        # ax.set_xlim(8220, 8850)
        # ax.set_xticks(np.arange(8220, 8850, 200))

    def plot_dfAL(self, dfAL=None, alpha=1):
        SZs = dfAL["Z"].unique()
        nPlots = len(SZs)
        neg= - 1/nPlots
        f, axs = plt.subplots(nPlots, 1, figsize=(20, 1*nPlots), sharex="col", sharey="col")
        for ii, SZ in enumerate(SZs): 
            ax=axs[ii]
            SLZ = dfAL[dfAL["Z"]==SZ].values
            # if ZU == 20:
            #     ax.vlines([8544.490, 8500.412,8664.576], 0, 1,"k",linestyle="-.", alpha=0.4)
            for i in range(len(SLZ)):
                SL = SLZ[i]
                c = np.where(SZs==SL[2])[0][0]
                cc=cm.gist_rainbow((c+neg)/nPlots)
                ax.axvline(SL[1], ymin=0, ymax=1, lw=2, color=cc, alpha=alpha, label=SL[2])
                self.set_unique_legend(ax,loc=0)
                ax.xaxis.grid(1)
        ax.set_ylim(0, 1)
        # ax.set_ylabel("I")
        # ax.set_xlim(self.W[0], self.W[1])

    def plot_dfSLU(self, dfSL=None, alpha=1):
        if dfSL is None: dfSL = self.dfSL
        nPlots = len(self.SZs)
        neg= - 1/nPlots
        f, axs = plt.subplots(nPlots, 1, figsize=(20, 1*nPlots), sharex="col", sharey="col")
        for ii, SZ in enumerate(self.SZs): 
            ax=axs[ii]
            SLZ = dfSL[dfSL["Z"]==SZ].values
            # if ZU == 20:
            #     ax.vlines([8544.490, 8500.412,8664.576], 0, 1,"k",linestyle="-.", alpha=0.4)
            for i in range(len(SLZ)):
                SL = SLZ[i]
                c = np.where(self.SZs==SL[3])[0][0]
                cc=cm.gist_rainbow((c+neg)/nPlots)
                ax.axvspan(SL[0], SL[1], ymin=0, ymax=1, lw=2, color=cc, alpha=alpha, label=int(SL[3]))
                self.set_unique_legend(ax,loc=0)
                ax.xaxis.grid(1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("I")
        ax.set_xlim(self.W[0], self.W[1])
        
    def filter_df(self, df=None, W0=None,W1=None,Ilb=None, Z=None,Q=None):
        if df is None: df = self.dfLL
        if W0 is not None:
            df= df[(df["W"]>W0)&(df["W"]<W1)]
        if Z is not None:
            df = df[df["Z"]==Z]
        if Q is not None:
            df = df[df["Q"]==Q]
        if Ilb is not None:
            df = df[df["I"]>Ilb]
        return df


    def get_zdf(self,Zs):
        self.zdf = self.dfLL26[self.dfLL26["Z"].isin(Zs)]

    def plot_LL26(self, df, gp=1, rng=[3800, 6500], ax=None):
#     if gp==1: 
        dff = df[(df["W"] >=rng[0]) & (df["W"] <=rng[1])]
        Z=dff["Z"].values
        W=dff["W"].values
    #     c = cm.gist_rainbow((c-np.min(c))/(np.max(c)-np.min(c)))
        c = cm.gist_rainbow((Z-1)/(27-1))
        if ax is None: ax = plt.subplots(figsize=(20,2))[1]
        for i in range(len(W)):
            ax.axvline(W[i], color=c[i], label=f"{Z[i]}{self.Els[Z[i]]}", alpha=0.5)
        ax.set_xlim(rng)
        self.set_unique_legend(ax, fix_ncol=0)
        # return dff

    def plot_LL(self, df, rng=[3800, 6500], ax=None):
        dff = df[(df["W"] >=rng[0]) & (df["W"] <=rng[1])]
        Z=dff["Z"].values
        W=dff["W"].values
    #     c = cm.gist_rainbow((c-np.min(c))/(np.max(c)-np.min(c)))
        c = cm.gist_rainbow((Z // 10 -1)/(10 -1))
        if ax is None: ax = plt.subplots(figsize=(20,2))[1]
        for i in range(len(W)):
            ax.axvline(W[i], color=c[i], label=f"{Z[i]}", alpha=0.5)
        ax.set_xlim(rng)
        self.set_unique_legend(ax, fix_ncol=0)

    def set_unique_legend(self, ax, fix_ncol=0, loc=4):
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        vals = by_label.values()
        if not fix_ncol:
            nn = len(vals) // 2
            ax.legend(vals, by_label.keys(), ncol=nn, loc=loc)

    def get_wave_axis(self, wave= None, ax=None, xgrid=True):
        if wave is None: 
            ax.set_xlim(self.nwave[0]-1, self.nwave[-1]+2)
            ax.set_xticks(np.arange(self.W[0], self.W[1], 200))  
        else:
            ax.set_xlim(wave[0]-1, wave[-1]+2)
            ax.set_xticks(np.arange(int(wave[0]), np.ceil(wave[-1]), 200))
        ax.xaxis.grid(xgrid)


def get_LL(filename, n=3, DIR="/scratch/ceph/szalay/swei20/LL/kurucz/", save=0, w=0):
    # "gfallvac08oct17.dat"
    path = os.path.join(DIR, filename)
    with open(path) as f:
        lines = f.readlines()
        mat = np.zeros((len(lines), n+1))
        print(mat.shape)
        for ii, line in enumerate(lines[:]): 
            line=re.split("\s+", line.strip())[:3]
            for nn, ll in enumerate(line):
                if ("-" in line[0]):
                    if  (nn == 0):
                        s01 = ll.split("-")
                        try:
                            mat[ii, 0] = float(s01[0])
                            mat[ii, 1] = -float(s01[1])
                        except:
                            print("loop",ii, nn, ll, s01)
                        mat[ii, 2] = line[1]
                        continue
                    else:
                        continue
                else:
                    try:
                        mat[ii,nn] = float(ll)
                    except:
                        print("other", ii, nn, ll, mat[ii], "line",line)
                        break
    mat[:,0] *= 10
    Z = mat[:,2] // 1
    Q = (mat[:,2] - Z ) * 100 // 1
    mat[:,2] = Z
    mat[:,3] = Q
    ll=pd.DataFrame(mat, columns=["W","I","Z","Q"])


    if save: 
        if w:
            ll = ll[(ll["vac_A"] >= 3800) & (ll["vac_A"]<=12600)]
        ll.to_csv("/scratch/ceph/szalay/swei20/LL/kurucz/gfall_vac_A.csv")
    return ll

