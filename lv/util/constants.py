class Constants():
    dWs={"Blue": [3800, 6500, 2300, "Blue"], "RedL": [6300, 9700, 3000, "RedL"], 
        "RedM": [7100, 8850, 5000, "RedM"], "NIR": [9400, 12600, 4300, "NIR"],
        "BL": [3800, 6500, 1000, "BL"], "RML": [7100, 8850, 1000, "RML"],
        "NL": [9400, 12600, 1000,"NL"]}

    dRs={"M": [[-2.5, 0.0], [3500, 5000], [0.0, 1.5],[-0.75, 0.5], [-0.25, 0.5]], 
        "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]],
        "C": [[-2.0, 0.0], [3750, 5500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]], 
        "B": [[-2.5,-1.5], [6750, 9500], [2.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]],
        "R": [[-1.0, 0.0], [5500, 6750], [2.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]], 
        "G": [[-2.5,-1.0], [4000, 5500], [1.5, 3.5],[-0.75, 0.5], [-0.25, 0.5]]}
    
    dR={"M":"M31G","W":"MWW","C":"MWC","B":"BHB","R":"RHB","G":"DGG"}

    dRR={"M":"M31G","W":"MWW","C":"MWC","B":"BHB","R":"RHB","G":"DGG"}
    dRC={"M": "orange" ,"W":"lightgreen","C":"brown","B":"dodgerblue","R":"red","G":"fuchsia"}
    dWw={"RedM": ["mr", "RML", "RedM"], "Blue": ["b", "BL","Blue"], "NIR": ["n", "NL", "NIR"]}
    dwW={"RML": ["mr", "RedM", "RML"], "BL": ["b", "Blue","BL"], "NL": ["n", "NIR", "NL"]}

    dC = {"MH": "plasma", "Teff": "gist_rainbow", "logg": "turbo",  "CH": "gist_rainbow", "AH":"winter"}
    Cnms = list(dC.values())
    Pnms = list(dC.keys())
    Ws = list(dWs.keys())
    RRnms = list(dRR.values())
    Rnms = list(dRR.keys())

    pshort = ["M", "T", "G", "C", "A"]

    uM = [-2.5 , -2.25, -2.  , -1.75, -1.5 , -1.25, -1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75]
    uT = [ 3500.,  3750.,  4000.,  4250.,  4500.,  4750.,  5000.,  5250.,
            5500.,  5750.,  6000.,  6250.,  6500.,  6750.,  7000.,  7250.,
            7500.,  7750.,  8000.,  8250.,  8500.,  8750.,  9000.,  9250.,
            9500.,  9750., 10000., 10250., 10500., 10750., 11000., 11250.,
            11500., 11750., 12000., 12500., 13000., 13500., 14000., 14500.,
            15000., 15500., 16000., 16500., 17000., 17500., 18000., 18500.,
            19000., 19500., 20000., 21000., 22000., 23000., 24000., 25000.,
            26000., 27000., 28000., 29000., 30000., 31000., 32000., 33000.,
            34000., 35000.]
    uG = [0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ]
    uC = [-0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ]
    uA = [-0.25,  0.  ,  0.25,  0.5 ]


    @staticmethod
    def get_path(boszR=5000,pixelR=5000, R="B"):
        bosz=f"/scratch/ceph/dobos/data/pfsspec/import/stellar/grid/bosz_{boszR}"
        rbf=f"/scratch/ceph/swei20/data/pfsspec/import/stellar/rbf/bosz_{boszR}_{dRR[R]}"
        grid=f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/M31G/grid/bosz_R{pixelR}_RedM_m19"
        return bosz, rbf


#path-----------------------------------------------------------------------------------------------------------------------
    GRID_DIR = "/datascope/subaru/user/swei20/data/pfsspec/import/stellar/grid/"



        # dRs_old={"M": [[-2.5, 0.0], [3500, 5000], [0.0, 2.0],[-0.75, 0.5], [-0.25, 0.5]], 
    #     "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]],
    #     # "C": [[-2.0, 0.0], [4500, 6000], [4.0, 5.0],[-0.75, 0.5], [-0.25, 0.5]], 
    #     "C": [[-1.0, 0.0], [4500, 5500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]], 

    #     "B": [[-2.5,-1.5], [7000, 9500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]],
    #     # "R": [[-1.0, 0.0], [5000, 6500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]], 
    #     "R": [[-1.0, 0.0], [5500, 6500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]], 

    #     "G": [[-2.5,-1.0], [3500, 5500], [0.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]]}
    