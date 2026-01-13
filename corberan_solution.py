import numpy as np
from corberan_data import CorberanData
import construction_heuristic
import json

class CorberanSolution:

    def __init__(self, data : CorberanData ):
        self.cData = data

        self.xvarsol = np.zeros((self.cData.nF, self.cData.nC, self.cData.nt), dtype=float)
        self.yvarsol = np.zeros((self.cData.nF, self.cData.nt), dtype=float)
        self.qvarsol = np.zeros((self.cData.nF, self.cData.nt), dtype=float)
        self.zvarsol = np.zeros((self.cData.nF, self.cData.nt), dtype=float)
        self.wvarsol = np.zeros((self.cData.nF, self.cData.nF, self.cData.nt), dtype=float)
        self.is_feasible = False
        self.objval = None
        self.errorlog = []

    def set_x(self, f, c, t, value):
        self.xvarsol[f,c,t-1] = value

    def set_y(self, f,t, value):
        self.yvarsol[f,t-1] = value

    def set_q(self, f,t, value):
        self.qvarsol[f,t-1] = value

    def set_z(self, f,t, value):
        
        self.zvarsol[f,t-1] = value
    
    def set_w(self, f_from, f_to, t, value):
        
        self.wvarsol[f_from, f_to,t-1] = value

    def set_objvalue(self, value):
        self.objval = value

    def compute_objval(self):
        self.objval = (self.cData.get_mult_cf()*(self.zvarsol*self.cData.cft).sum() +
                       self.cData.get_mult_ca()*(self.xvarsol*self.cData.cat).sum() +
                       self.cData.get_mult_cr()*(self.yvarsol*self.cData.crt).sum() +
                       self.cData.get_mult_cm()*(self.wvarsol*self.cData.cmt).sum())
        return self.objval

    def set_all_vars(self, Y, X, Q, Z, W):
        self.xvarsol = X
        self.yvarsol = Y
        self.zvarsol = Z
        self.qvarsol = Q
        self.wvarsol = W

        self.compute_objval()

    
    def export_json(self, filename : str):

        # TODO: implementa da numpy a dizionario con la soluzione
        j_sol = {
            "objval" : self.objval,
            "is_feasible" : self.is_feasible
        }

        for t in range(self.cData.nt):
            f, c = np.where(self.xvarsol[:,:,t] > 1e-5)
            for i, _f in enumerate(f):
                _c = c[i]
                j_sol[f'x_{_f}_{_c}_{t+1}'] = float(self.xvarsol[_f,_c,t])

            f = np.where(self.yvarsol[:, t] > 1e-5)[0]
            for _f in f:
                j_sol[f'y_{_f}_{t+1}'] = float(self.yvarsol[_f, t])

            f = np.where(self.zvarsol[:, t] > 1e-5)[0]
            for _f in f:
                j_sol[f'z_{_f}_{t+1}'] = float(self.zvarsol[_f, t])

            f = np.where(self.qvarsol[:, t] > 1e-5)[0]
            for _f in f:
                j_sol[f'q_{_f}_{t+1}'] = float(self.qvarsol[_f,t])

            f1, f2 = np.where(self.wvarsol[:,:,t] > 1e-5)
            for i, _f1 in enumerate(f1):
                _f2 = f2[i]
                j_sol[f'w_{_f1}_{_f2}_{t+1}'] = float(self.wvarsol[_f1,_f2,t])

        with open(filename, 'w') as fout:
            json.dump(j_sol, fout)

    def build_construction_heuristic(self):
        return CorberanSolution.construction_heuristic(self.cData)

    @classmethod
    def construction_heuristic(cls, data : CorberanData):
        
        obj =  cls(data)

        # Variables
        # x = np.zeros((data.nF, data.nC, data.nt))
        # y = np.ones((data.nF, data.nt))
        # w = np.zeros((data.nF, data.nF, data.nt))
        # q = np.tile(data.q_in[:, None], (1, data.nt))
        # z = np.zeros((data.nF, data.nt))


        # # Check for feasibility
        # for t in range(data.nt):
        #     if data.q_in.sum() < sum(data.d[c, t+1] for c in range(data.nC)):
                
        #         obj.errorlog.append(f"Not enough resources for demand at time {t}. Problem is infeasible.")
        #         return obj

        # # TODO: parallelizza il loop per ogni t \in T eseguendo il codice attuale
        # # oppure sempre parallelizzando il loop per ogni t \in T, ogni singolo T è un problema di flusso
        # # aprendo ogni facility, che può essere risolto all'ottimo in tempo polinomiale
        # # networkx
        # for t in range(data.nt):
        #     for c in range(data.nC):
        #         already_assigned = 0

        #         f = 0
        #         residual_capacity = data.q_in[f] - sum(x[f, k, t] for k in range(data.nC))

        #         while already_assigned < data.d[c, t+1]:
        #             if residual_capacity > data.d[c, t+1] - already_assigned:
        #                 x[f, c, t] += data.d[c, t+1] - already_assigned
        #                 already_assigned = data.d[c, t+1]
        #             else:
        #                 if residual_capacity > 0:
        #                     x[f, c, t] += residual_capacity
        #                     already_assigned += residual_capacity
        #                 f += 1
        #                 if f < data.nF:
        #                     residual_capacity = data.q_in[f] - sum(x[f, k, t] for k in range(data.nC))

        # for t in range(data.nt):
        #     for j in range(data.nC):
        #         if x[:, j, t].sum() != 0:
        #             for i in range(data.nF):
        #                 x[i, j, t] = x[i, j, t] / x[:, j, t].sum()

        x, y, w, q, z = construction_heuristic.construction_heuristic_cpp(data.nF, data.nC, data.nt, data.d, data.q_in)

        obj.set_all_vars(y, x, q, z, w)
        obj.isfeasible = True

        return obj

