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

        self.dfLL = None
        self.dfLL26 = None 

        self.load_LL(w)

    def load_LL(self, w):
        dfLL = pd.read_csv(f"/scratch/ceph/szalay/swei20/LL/kurucz/gfall_vac_{w}.csv")
        # self.dfLL = dfLL
        self.dfLL = dfLL[(dfLL["I"]>-3.)]
        self.dfLL26 = dfLL[(dfLL["Z"] < 27)]

    def prepare_LL(self, w):
        pass

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

