from corberan_data import CorberanData, PricerTimeColumn
from dataclasses import dataclass
from typing import List, Callable
import numpy as np
import re
import math
from typing import Dict
import networkx as nx

@dataclass
class NsgaProblemConfig:
    name: str
    get_n_var: Callable[[CorberanData], int]    # lunghezza cromosoma
    column_to_chromosome: Callable[[PricerTimeColumn, CorberanData], np.ndarray]
    fitness: Callable[[np.ndarray, CorberanData, int], float]
    get_header_vars: Callable[[CorberanData], List[str]]


def parse_index_y(name: str):
    if name.startswith("y_"):
        return int(name.split("_")[1])
    m = re.match(r"y\[(\d+)\]", name)
    if m:
        return int(m.group(1))
    return None

def parse_index_x(name: str):
    if name.startswith("x_"):
        parts = name.split("_")
        if len(parts) >= 3:
            return int(parts[1]), int(parts[2])
    m = re.match(r"x\[(\d+),\s*(\d+)\]", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def parse_index_q(name: str):
    if name.startswith("q_"):
        return int(name.split("_")[1])
    m = re.match(r"q\[(\d+)\]", name)
    if m:
        return int(m.group(1))
    return None

def split_chromosome(chrom: np.ndarray, cData: CorberanData):
    nF = cData.nF
    nC = cData.nC

    y = chrom[:nF].astype(float)
    q = chrom[nF:nF+nF].astype(float)
    x = chrom[nF+nF:].astype(float).reshape(nF,nC)

    return y, q, x

def join_chromosome(y, q, X):
    return np.concatenate([y, q, X.reshape(-1)])

def check_chrom_feasibility(chrom, cData, t):
    y, q, x = split_chromosome(chrom, cData)
    y = y.astype(float)
    q = q.astype(float)
    x = x.astype(float)
    
    nF, nC = cData.nF, cData.nC
    
    report = {}

    if cData.d.sum() > 0.0:
        min_cov = 1.0
        viol_cov = 0.0
        for c in range(nC):
            coverage = float(np.sum(x[:, c]))
            if cData.d[c, t] > 0:
                min_cov = min(min_cov, coverage)
                viol_cov += max(0.0, 1.0 - coverage)
        report["min_coverage"] = float(min_cov)
        report["viol_coverage"] = float(viol_cov)
    
    else:
        report["min_coverage"] = 0.0
        report["viol_coverage"] = 0.0

    report["sum_q"] = float(q.sum())
    report["viol_sum_q"] = max(0.0, report["sum_q"] - cData.M)

    viol_cap = 0.0
    for f in range(nF):
        load_f = float(np.sum(cData.d[:, t] * x[f, :]))
        viol_cap += max(0.0, load_f - q[f])
    report["viol_capacity"] = viol_cap

    report["open_facilities"] = int(y.sum())
    
    return report


def columns_equal(pattern1: Dict[str, float],
                  pattern2: Dict[str, float],
                  tol: float = 1e-6) -> bool:
    
    if pattern1.keys() != pattern2.keys():
        return False
    for k in pattern1:
        if not math.isclose(pattern1[k], pattern2[k], rel_tol=tol, abs_tol=tol):
            return False
    return True

def x_build_from_yq(y: np.ndarray, q: np.ndarray, cData, t: int, top_k: int=30, scale=1e6):
 
    nF, nC = cData.nF, cData.nC
    d = cData.d[:, t]
    rank = cData.ca_sorted

    active = np.flatnonzero(d > 0.0)
    if active.size == 0:
        return np.zeros((nF, nC), dtype=float), True
    
    open_y = (y > 0.5)
    q_eff = np.asarray(q, dtype=float).copy()
    q_eff[~open_y] = 0.0

    s = q_eff.sum()
    if s > cData.M:
        q_eff *= (cData.M / s)

    if q_eff.sum() < d[active].sum():
        return np.zeros((nF, nC), dtype=float), False
    
    x = np.zeros((nF, nC), dtype=float)
    rem = q_eff

    def assign_client(c, facilities_order):
        dc = d[c]
        left = dc

        for f in facilities_order:
            cap = rem[f]
            if cap <= 1e-8:
                continue
            if cap >= left:
                rem[f] = cap - left
                x[f, c] += left / dc
                return True
            else:
                rem[f] = 0.0
                x[f, c] += cap / dc
                left -= cap

        return False
    
    for c in active:
        cand = rank[:top_k, c]
        cand = cand[open_y[cand]]

        ok = assign_client(c, cand)
        if not ok:
            cand2 = rank[:, c]
            cand2 = cand2[open_y[cand2]]
            ok2 = assign_client(c, cand2)
            if not ok2:
                return x, False
            
    return x, True


# class FlowXBuilder:
#     def __init__(self, cData: CorberanData, t: int, top_k: int=30, scale: float=1e6):
#         self.cData = cData
#         self.t = t
#         self.nF = cData.nF
#         self.nC = cData.nC
#         self.top_k = top_k
#         self.scale = scale

#         self.d = np.rint(np.asarray(cData.d[:, t], dtype=float) * self.scale)

#         self.G = None
#         self._built = False


#     def build_graph(self):
#         G = nx.DiGraph()

#         for f in range(self.nF):
#             G.add_node(f"F{f}", demand=0)

#         for c in range(self.nC):
#             G.add_node(f"C{c}", demand=self.d[c])

#         for f in range(self.nF):
#             for c in range(self.nC):
#                 G.add_edge(f"F{f}", f"C{c}", weight = self.cData.ca[f, c], capacity=10**12)

#         G.add_node("Slack", demand=0)
#         G.add_edge(f"F{f}", "Slack", weight=0.0, capacity=10**12) 

#         self.G = G
#         self._built = True

#     def update_graph(self, y: np.ndarray, q: np.ndarray):
#         assert self._built, "Call build_graph() first."

#         qI = np.rint(q * self.scale)

#         for f in range(self.nF):
#             self.G.nodes[f"F{f}"]["demand"] = - qI[f] * y[f]

#         total = sum(data["demand"] for _, data in self.G.nodes(data=True))
#         if total < 0:
#             self.G.nodes["Slack"]["demand"] = -total

#         self._last_infeasible = (total > 0)

#     def solve_graph(self):
#         if getattr(self, "_last_infeasible", False):
#             return None, False
#         try:
#             _, flowDict = nx.network_simplex(self.G)
#             return flowDict, True
#         except nx.NetworkXUnfeasible:
#             return None, False
        
#     def extract_x(self, flowDict):
#         x = np.zeros((self.nF, self.nC))

#         for f in range(self.nF):
#             for c in range(self.nC):
#                 x[f, c] = flowDict[f"F{f}"].get(f"C{c}", 0.0) / (self.scale)

        # for c in range(self.nC):
        #     if x[:, c].sum() != 0:
        #         norm = x[:, c].sum()
        #         for f in range(self.nF):
        #             x[f, c] = x[f, c] / norm
#         return x
    
#     def build_x(self, y, q):
#         self.update_graph(y, q)
#         flow, feas = self.solve_graph()
#         if not feas:
#             return np.zeros((self.nF, self.nC), dtype=float), False
#         return self.extract_x(flow), True
        
    
        





         


