from gurobipy import Model, GRB
import toy_instances as toy
from corberan_data import CorberanData, get_index
from corberan_solution import CorberanSolution
import logging


class CompactModel:

    def __init__(self, data : CorberanData):
        self.cData  = data

        self.y = None
        self.x = None
        self.w = None
        self.z = None
        self.q = None

        self.model = None

        self._build_model()

    def _build_model(self):
        ################ VARIABILI ###################
        nI = self.cData.nF
        Tstar = self.cData.nt + 1
        nJ = self.cData.nC
        T = self.cData.T
        # cf = self.cData.cf
        # cr = self.cData.cr
        ca = self.cData.ca
        cm = self.cData.cm
        d = self.cData.d
        q_in = self.cData.q_in
        M = self.cData.M 

        m = Model("LM1")

        m.setParam("LogFile",
                str(self.cData.log_out_dir) + "/" + self.cData.outfilename + f"_gurobi.log"
        )

        logging.info("init variabiles")

        self.y = m.addVars(nI, Tstar, vtype=GRB.BINARY, name="y")  # Apertura delle facilities
        self.x = m.addVars(nI, nJ, T, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name="x")  # Assegnazione della domanda
        self.w = m.addVars(nI, nI, T, vtype=GRB.CONTINUOUS, lb = 0,  name="w")  # Trasferimento tra facilities
        self.z = m.addVars(nI, T, vtype=GRB.BINARY, name="z")   # Apertura dopo periodo di chiusura delle facilities
        self.q = m.addVars(nI, Tstar, vtype=GRB.CONTINUOUS, lb = 0, name="q")   # Capacita' ausiliaria, ad ogni tempo

        ################ OBIETTIVO ###################
        logging.info("set objective function")

        m.setObjective(
            sum((
                self.cData.get_mult_cf()*self.cData.get_cf(i,t-1) * self.z[i, t] +
                 self.cData.get_mult_cr()*self.cData.get_cr(i,t-1) * self.y[i, t] +
                self.cData.get_mult_ca()*sum(ca[i, j] * self.x[i, j, t] for j in range(nJ)) +
                self.cData.get_mult_cm()*sum(cm[i, k] * self.w[i, k, t] for k in range(nI) if i != k))
                for i in range(nI) for t in T),
                GRB.MINIMIZE
        )

        ################ VINCOLI ###################

        logging.info("constraints 1")
        # Apertura facilities
        for t in T:
            for i in range(nI):
                m.addConstr(self.y[i, t] - self.y[i, t-1] <= self.z[i, t], f"z_cont{i}_{t}")

        logging.info("constraints 2")
        # Soddisfazione domanda
        for t in T:
            for j in range(nJ):
                if d[j,t] != 0:
                    m.addConstr(sum(self.x[i, j, t] for i in range(nI)) == 1, f"demand{j}_{t}")

        logging.info("constraints 3")
        # Facility soddisfa solo se aperta
        for t in T:
            for i in range(nI):
                    for j in range(nJ):
                        m.addConstr(self.x[i, j, t] <= self.y[i, t], f"closing_{i}_{j}_{t}")

        logging.info("constraints 4")
        # Rispetto della capacità
        for t in T:
            for i in range(nI):
                m.addConstr(sum(d[j, t] * self.x[i, j, t] for j in range(nJ)) <= self.q[i, t], f"capacity_{i}_{t}")

        logging.info("constraints 5")
        # Flusso capacità
        for t in T:
            for i in range(nI):
                m.addConstr(self.q[i, t] == self.q[i, t-1] + sum(self.w[k, i, t] - self.w[i, k, t] for k in range(nI) if i != k), f"aux_capacity_{i}_{t}")

        logging.info("constraints 6")
        # Massimo trasferimento di capacità
        for t in T:
            for i in range(nI):
                m.addConstr(sum(self.w[i, k, t] for k in range(nI) if i != k) <= self.q[i, t-1], f"transfer_{i}_{t}")

        logging.info("constraints ")
        # Trasferimento solo per facility di partenza aperta al tempo precedente e facility di arrivo aperta al tempo corrente
        for t in T:
            for i in range(nI):
                for k in range(nI):
                    if i != k:
                        m.addConstr(self.w[i, k, t] <= M * self.y[i, t-1], f"transfer_limit_{i}_{k}_{t}")
                        m.addConstr(self.w[i, k, t] <= M * self.y[k, t], f"transfer_limit_{i}_{k}_{t}")

        logging.info("starting time-slot")
        # Facility considerate aperte al tempo 0
        for i in range(nI):
            m.addConstr(self.y[i, 0] == 1, f"open_fac_{i}")

        # Capacità iniziale
        for i in range(nI):
            m.addConstr(self.q[i, 0] == q_in[i], f"start_capacity_{i}")

        self.model = m

    def run(self, timelim : int = None,
            rootonly : bool = False,
            LP_relaxation : bool = False) -> CorberanSolution | None:

        if timelim is not None:
            self.model.setParam('TimeLimit', timelim)

        if rootonly:
            self.model.setParam('NodeLimit', 0)
        elif LP_relaxation:
            for x in [self.y, self.z]:
                for el in x.values():
                    el.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)

        self.model.optimize()
        
        status_compact = self.model.getAttr(GRB.Attr.Status)

        if status_compact == GRB.OPTIMAL:
            logging.info("[COMPACT] optimal solution found")
        elif self.model.getAttr(GRB.Attr.SolCount) > 0:
            logging.info(f"[COMPACT] sub-optimal solution found with status {status_compact}")
        else:
            logging.info(f"[COMPACT] no solution found {status_compact}")

            return None

        new_sol = CorberanSolution(self.cData)

        for v in self.model.getVars():
            if v.X < 1e-5:
                continue
            idx = get_index(v.VarName)
            if v.VarName.startswith('y'):
                i, t = int(idx[0]), int(idx[1])
                if t == 0:
                    continue
                new_sol.set_y(i, t, v.X)
            elif v.VarName.startswith('z'):
                i, t = int(idx[0]), int(idx[1])
                new_sol.set_z(i, t, v.X)
            elif v.VarName.startswith('w'):
                i, j, t = int(idx[0]), int(idx[1]), int(idx[2])
                new_sol.set_w(i,j,t, v.X)
            elif v.VarName.startswith('x'):
                i, j, t = int(idx[0]), int(idx[1]), int(idx[2])
                new_sol.set_x(i, j, t, v.X)
            elif v.VarName.startswith('q'):
                i, t = int(idx[0]), int(idx[1])
                if t == 0:
                    continue
                new_sol.set_q(i, t, v.X)   

        new_sol.is_feasible = True
        new_sol.set_objvalue(self.model.Objval)

        return new_sol

################ PARAMETRI  ###################

def main():

    m = CompactModel(toy.example2())

    sol = m.run(1200)

    sol.export_json("../compact_solutions.json")

if __name__ == "__main__":
    main()