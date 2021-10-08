class Constants():
    dWs={"Blue": [3800, 6500, 2300, "Blue"], "RedL": [6300, 9700, 3000, "RedL"], 
        "RedM": [7100, 8850, 5000, "RedM"], "NIR": [9400, 12600, 4300, "NIR"],
        "BL": [3800, 6500, 1000, "BL"], "RML": [7100, 8850, 1000, "RML"],
        "NL": [9400, 12600, 1000,"NL"]}
    dRs={"M": [[-2.5, 0.0], [3500, 5000], [0.0, 2.0],[-0.75, 0.5], [-0.25, 0.5]], 
        "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]],
        "C": [[-2.0, 0.0], [4500, 6000], [4.0, 5.0],[-0.75, 0.5], [-0.25, 0.5]], 
        "B": [[-2.5,-1.5], [7000, 9500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]],
        "R": [[-1.0, 0.0], [5000, 6500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]], 
        "G": [[-2.5,-1.0], [3500, 5500], [0.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]]}
    
    dR={"M":"M31G","W":"MWW","C":"MWC","B":"BHB","R":"RHB","G":"DGG"}
    dC = {"[M/H]": "plasma", "Teff": "gist_rainbow", "Logg": "turbo",  "[C/M]": "gist_rainbow", "[a/M]":"winter"}
    Cs = list(dC.values())
    Pnms = list(dC.keys())
    Ws = list(dWs.keys())
    RRnms = list(dR.values())
    Rnms = list(dR.keys())

    # Rk = Rs.keys()