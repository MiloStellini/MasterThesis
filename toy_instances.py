import pandas as pd
import numpy as np
from corberan_data import CorberanData
from pathlib import Path

def dataset_amelia(epsg : int = 4326) -> CorberanData:

    # df = pd.read_csv("/home/marco/Documents/work/corberan/tesi_milo/NIL_dataset_AMELIA_mimb.csv", sep= ';')
    # df = pd.read_csv(Path(__file__).parent / "../amelia_mi_70_epsg4326.csv", sep=";")
    df = pd.read_csv(Path(__file__).parent / "../../NIL_dataset_AMELIA.csv", sep=";")
    torem = []
    for c in ['usable','personnel']:
        if c in df.columns:
            torem.append(c)
    df.drop(columns=torem, inplace=True)

    cData = CorberanData.from_dataframe(df,
                                        "geometry", epsg,
                                        "demand",
                                        resourcecol="beds")

    return cData

def example1()-> CorberanData:

    cData = CorberanData()

    F = ["A", "B"]
    C = ["c1", "c2"]
    cData.nF = len(F)
    cData.nC = len(C)

    cData.nt = 3                          # Time intervals
    cData.T = range(1,cData.nt+1)
    cData.Tstar = range(cData.nt+1)

    cData._cf = np.array([5, 5])
    cData.ca = np.array([[1, 2],          # Costs
                [2, 1]])
    cData._cr = np.array([10, 10])
    cData.cm = np.array([[0, 1],
                [1, 0]])

    cData.d = np.array([[0, 1, 0, 1],     # Demand
                [0, 0, 1, 0]])


    cData.q_in = np.array([2, 1])         # Initial capacity
    cData.M = cData.q_in.sum()  # parametro per maggiorazione
    
    return cData

def example2()-> CorberanData:

    cData = CorberanData()

    F = ["A", "B"]
    C = ["c1", "c2", "c3"]
    cData.nF = len(F)
    cData.nC = len(C)

    cData.nt = 3
    cData.T = range(1, cData.nt + 1)
    cData.Tstar = range(cData.nt + 1)

    # Costi
    cData._cf = np.array([5, 5])   # Costi fissi di apertura
    cData._cr = np.array([5, 5])   # Costi operativi
    cData.ca = np.array([[1, 2, 3],   # Costi assegnamento clienti
                   [200, 100, 100]])
    
    # Costo di trasferimento tra facilities
    cData.cm = np.array([[0, 1],
                   [2, 0]])

    # Domanda (ogni colonna è un tempo)
    cData.d = np.array([[0, 3, 1, 2],    # c1
                  [1, 0, 2, 1],    # c2
                  [0, 2, 3, 0]])   # c3

    # Capacità iniziali molto sbilanciate
    cData.q_in = np.array([5, 1])

    cData.M = cData.q_in.sum()  # maggiorazione

    cData.repeat_all_costs_per_time()

    return cData
