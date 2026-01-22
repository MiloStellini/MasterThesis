import json

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist
import gurobipy as gp
from typing import Dict
from enum import Enum
from pandas.api.types import is_numeric_dtype

class CorberanAlgoType(Enum):
    COMPACT = 0
    CG_SEQ = 1
    CG_PAR = 2
    ROOT = 3
    LP_REL = 4
    GA_ONLY = 5
    HEUR_ONLY = 6
    DRMP_ONLY = 7

    @classmethod
    def from_label(cls, label: str):
        _label = label.lower()
        if _label == "compact":
            return cls.COMPACT
        elif _label == "cg_seq":
            return cls.CG_SEQ
        elif _label == "cg_par":
            return cls.CG_PAR
        elif _label == "root":
            return cls.ROOT
        elif _label == "lp_rel":
            return cls.LP_REL
        elif _label == "ga_only":
            return cls.GA_ONLY
        elif _label == "heur_only":
            return cls.HEUR_ONLY
        elif _label == "drmp_only":
            return cls.DRMP_ONLY
        raise ValueError("expecting a label in set ['compact', 'cg_par', 'cg_seq']")
    

def get_index(name: str) -> tuple:
    if "[" in name and name.endswith("]"):
        inside = name.split("[", 1)[1][:-1]
        return tuple(s.strip() for s in inside.split(","))
    elif "_" in name:
        return tuple(int(el.strip()) for el in name.split("_")[1:])
    return ()

class CorberanData:

    def __init__(self):
        self.nF = None
        self.nC = None
        self.nt = None
        self.T = None
        self.Tstar = None
        self.d = None 
        self.q_in = None 

        self._cf = None
        self._cr = None
        self.ca = None 
        self.cm = None 
        
        self.M  = None
        self.log_out_dir = None
        self.debug_info = False
        self.nonzero_tolerance = 1e-5

        self.cft = None 
        self.crt = None 
        self.cat = None 
        self.cmt = None

        self.outfilename = None

        self.multipliers = {
            "cf" : 1,
            "cr" : 1,
            "ca" : 1,
            "cm" : 1
        }

        ##################### ADDED ####################
        self.use_nsga = False
        self.nsga_min_cols_per_t = 5
        self.nsga_pop_size = 50
        self.nsga_n_gen = 100

        self.ca_sorted = None
        self.seed = 0
        self.nsga_pool_frac = 1.0
        self.do_cg_pool = False
        self.pool_file = ""
        self.heur_pool_file = ""
        self.cols_file = ""
        ################################################

    def _get_mult(self, lab):
        return self.multipliers[lab]

    def get_mult_cf(self):
        return self._get_mult("cf")

    def get_mult_ca(self):
        return self._get_mult("ca")

    def get_mult_cr(self):
        return self._get_mult("cr")

    def get_mult_cm(self):
        return self._get_mult("cm")

    def repeat_costs_per_time(self, costs : np.ndarray):

        return np.repeat(costs, self.nt).reshape(*costs.shape, self.nt)
    
    def repeat_all_costs_per_time(self):
        if self._cf.ndim == 1:
            self.cft = self.repeat_costs_per_time(self._cf)
        if self._cr.ndim == 1:
            self.crt = self.repeat_costs_per_time(self._cr)
        self.cat = self.repeat_costs_per_time(self.ca)
        self.cmt = self.repeat_costs_per_time(self.cm)

    def set_log_out_dir(self, dirname : str):
        self.log_out_dir = dirname

    @classmethod
    def from_json_params(cls,
                         basedir : str,
                         filename: str):

        with open(basedir + filename + ".json", 'r') as fin:
            jf = json.load(fin)

            if "CSV" in jf:
                df = pd.read_csv(basedir + jf["CSV"], sep=jf.get("sep", ";"))
                geom_col = jf.get("geomcol", "geometry")
                epsg = jf.get("epsg", 4326)
                demandcol = jf["demandcol"]
                resourcecol = jf["resourcecol"]
                demandsep = jf.get("intracolsep", ",")
                opencostcol = jf["opencostcol"]
                runcostcol = jf["runcostcol"]
                return cls.from_dataframe(df, geom_col, epsg,
                                          demandcol, resourcecol,
                                          opencostcol, runcostcol,
                                          demandsep)
            else:
                pass

        return None

    @classmethod
    def from_dataframe(cls,
                       df : pd.DataFrame,
                       geom_col : str,
                       epsg : int,
                       demandcol : str,
                       resourcecol : str,
                       opencostcol : str,
                       runcostcol : str,
                       intracolsep : str):
        
        cData = cls()

        gdf = gpd.GeoDataFrame(df)
        gdf[geom_col] = gpd.GeoSeries.from_wkt(gdf[geom_col])
        gdf = gdf.set_geometry(geom_col)
        gdf = gdf.set_crs(epsg=epsg)

        F = df[df[demandcol].isna()]
        C = df[df[demandcol].notna()]

        geo_fac = gdf[gdf[demandcol].isna()]
        geo_cust = gdf[gdf[demandcol].notna()]

        F_coords = np.array([[geom.x, geom.y] for geom in geo_fac.centroid])
        C_coords = np.array([[geom.x, geom.y] for geom in geo_cust.centroid])

        ################ PARAMETRI  ###################
        cData.nF = len(F)  # Numero di facilities
        cData.nC = len(C)  # Numero di customers

        dist_F_C = cdist(F_coords, C_coords)  # Distanze facilities-customers
        dist_F_F = cdist(F_coords, F_coords) # Distanze tra facilities

        demand = C[demandcol]  # Domanda dei clienti
        if is_numeric_dtype(demand):
            cData.d = demand.values
            cData.nt = 1
        else:
            cData.d = np.array([[0] + [float(el) for el in v.split(intracolsep)] for v in demand])
            # np.column_stack([np.random.permutation(demand) for _ in cData.Tstar])
            cData.nt = cData.d.shape[1]-1# Numero intervalli di tempo

        cData.Tstar = range(cData.nt+1)
        cData.T = range(1, cData.nt+1)

        cData.q_in = np.array(F[resourcecol])  # Capacità delle facilities

        if is_numeric_dtype(F[opencostcol]):
            cData.cf = F[opencostcol].values # np.ones(cData.nF, dtype=int)  # Costo fisso di apertura delle facilities
        else:
            cData.cf = np.array([[float(el) for el in v.split(intracolsep)]
                                    for v in F[opencostcol].values])

        if is_numeric_dtype(F[runcostcol]):
            cData.cr = F[runcostcol].values # np.ones(cData.nF, dtype=int)
        else:
            cData.cr = np.array([[float(el) for el in v.split(intracolsep)]
                                 for v in F[runcostcol].values])

        cData.ca = dist_F_C
        cData.cm = dist_F_F

        cData.ca = cData.ca / cData.ca.max()
        cData.cm = cData.cm / cData.cm.max()

        cData._cf =  np.round(cData.cf, 4)
        cData._cr = np.round(cData.cr, 4)

        cData.ca = np.round(cData.ca, 4)
        cData.cm = np.round(cData.cm, 4)

        cData.M = cData.q_in.sum()  # parametro per maggiorazione

        cData._normalize_costs()

        cData.repeat_all_costs_per_time()

        cData.ca_sorted = np.argsort(cData.ca, axis=0)

        return cData

    def _normalize_costs(self):
        over_ca = 1/self.ca.sum()
        over_cm = 1/self.cm.sum()
        over_cr = 1/self._cr.sum()
        over_cf = 1/self._cf.sum()

        overmax = max(over_ca, over_cm, over_cr, over_cf)

        self.ca = (over_ca/overmax) * self.ca
        self.cm = (over_cm / overmax) * self.cm
        self._cr = (over_cr / overmax) * self._cr
        self._cf = (over_cf / overmax) * self._cf

    @staticmethod
    def _get_cost_facility_time( f, t, costs):
        if costs.ndim > 1:
            return costs[f, t]
        else:
            return costs[f]

    def get_cf(self, f, t):
        return self._get_cost_facility_time(f, t, self._cf)

    def get_cr(self, f, t):
        return self._get_cost_facility_time(f, t, self._cr)


class PricerTimeColumn:

    def __init__(self, 
                 col_name : str, 
                 cg_iter : int, 
                 time_index: int, 
                 nonzero_vars : Dict[str, float], 
                 var : gp.Var):
        self.col_name = col_name 
        self.cg_iter = cg_iter 
        self.time_index =  time_index 
        self.nonzero_vars = nonzero_vars 
        self.var = var
        self.replicas = 1

    def add_replicas(self):
        self.replicas+=1

