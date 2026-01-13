import numpy as np
import pickle
from collections import defaultdict
from nsga_config import parse_index_q, parse_index_x, parse_index_y

def save_population(pool: dict, path: str):
    data = {}
    for t, chrom in pool.items():
        if len(chrom) == 0:
            continue
        data[f"t{t}"] = np.vstack(chrom)

    np.savez_compressed(path, **data)

def load_population(path: str) -> dict:

    data = np.load(path)
    pool = {}
    for k in data.files:
        if k.startswith("t"):
            t = int(k[1:])
            pool[t] = data[k]
    return pool


def save_columns(columns: defaultdict[list], path):
    serial = {}
    for t, cols in columns.items():
        serial[t] = []
        for col in cols:
            serial[t].append({
                "time": t,
                "name": col.col_name,
                "nonzero_vars": col.nonzero_vars,
                "cg_iter": col.cg_iter,
            })
    with open(path, "wb") as f:
        pickle.dump(serial, f)

def load_columns(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def rebuild_xyq_from_nonzero(nonzero, cData):
    nF, nC = int(cData.nF), int(cData.nC)
    Y = np.zeros(nF, dtype=float)
    Q = np.zeros(nF, dtype=float)
    X = np.zeros((nF, nC), dtype=float)

    for name, val in nonzero.items():
        if name.startswith("y"):
            idx = parse_index_y(name)
            Y[idx] = float(val)

        elif name.startswith("q"):
            idx = parse_index_q(name)
            Q[idx] = float(val)

        elif name.startswith("x"):
            idx1, idx2 = parse_index_x(name)
            X[idx1, idx2] = float(val)

    return Y, Q, X


