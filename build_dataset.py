#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 09:42:59 2025

@author: marco
"""


import pandas as pd
import numpy as np
import os
import json


def  build_datasets_nodes_density():
    df_orig = pd.read_csv("../NIL_dataset_AMELIA.csv", sep=";")

    df = df_orig[["ID", "geometry","demand", "beds"]].copy()

    areas=df.loc[~df.demand.isna()].copy()
    fac = df.loc[df.demand.isna()].copy()
    all_dem = fac.beds.sum()

    for p in np.arange(1,10):
        f = p/10
        C = np.floor(areas.shape[0] * f ).astype(int)
        F = np.floor(fac.shape[0] *f).astype(int)


        areas_s = areas.iloc[np.random.choice(np.arange(areas.shape[0]),C,False)]
        fac_s = fac.iloc[np.random.choice(np.arange(fac.shape[0]),F,False)]

        for d in [0.25*i for i in range(1,5)]:
            areas_sd = areas_s.copy()
            areas_sd["demand"] = np.floor((areas_s.demand * (fac_s.beds.sum() * d)/ areas_s.demand.sum()).values)

            x = pd.concat([areas_sd,fac_s])

            x.to_csv("../amelia_N%d_d%d.csv" % (f*100, d*100), sep=";", index=None)

def create_df(df, T, basedir, filename):


    df = df[[ "geometry", "ID", "demand", "beds"]].copy()
    
    areas = df.loc[~df.demand.isna()].copy()
    
    fac = df.loc[~df.beds.isna()].copy()
    fac["opencost"] = 1
    fac["runcost"] = 1
    
    # T = 3
    
    dt = np.zeros((areas.shape[0], T))
    dt[:, 0] = areas.demand.values
    
    dp =  np.array([np.random.permutation(areas.demand.values) for _ in range(T-1)]).transpose()
    dt[:, 1:] = dp
    
    ds = [",".join(["%.3f" % el for el in row ]) for row in dt]
    
    areas["demand_T"] = ds
    
    dfsol = pd.concat([areas, fac])
    
    dfsol.to_csv(f"{basedir}{filename}_T{T}.csv", sep=";", index=None)

def create_json(fname, basedir ):
    j = {
        "CSV": fname + ".csv",
        "sep": ";",
        "epsg": 4326,
        "geomcol": "geometry",
        "demandcol": "demand_T",
        "demandcolsep": ",",
        "resourcecol": "beds",
        "opencostcol": "opencost",
        "runcostcol": "runcost"}
    with open(basedir + fname + ".json", "w") as fout:
        json.dump(j, fout)

datadir = "/home/marco/Documents/work/corberan/data/corberan_profiling_dataset_orig/"
outdir = "/home/marco/Documents/work/corberan/data/corberan_profiling_dataset/"
for f in os.listdir(datadir):
    print( f.split(".")[0])
    _df = pd.read_csv(datadir + f, sep=";")
    for T in [5,10,15,20,25]:
        # create_df(_df, T, outdir, f.split(".")[0])
        # create_json(f.split(".")[0]+f"_T{T}", outdir)
        pass

# basedir = '/home/marco/Documents/work/corberan/data/'
# df = pd.read_csv(f"{basedir}amelia_mi_70_epsg4326.csv", sep=";")
# df["demand"] = df["demand"]/400
# for T in [5,10,15,20,25]:
#     create_df(df, T, basedir, "amelia_mi_70_epsg4326" )



