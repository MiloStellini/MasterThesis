import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict
from corberan_data import CorberanData, PricerTimeColumn, get_index
from corberan_solution import CorberanSolution
import logging
import os
import outputs_logging as csv_out

class DiscreteRMP:

    def __init__(self, 
                 rmp : gp.Model, 
                 chi_vars : Dict[str, List[PricerTimeColumn]], 
                 data  : CorberanData):
        
        self.model = rmp

        self.model.setParam("LogFile",
                        os.path.join(str(data.log_out_dir), data.outfilename + f"_DRMP.log")
                        )

        self.chi_vars = chi_vars

        self.cData = data

        self.last_sol = None

        for _, column in chi_vars.items():
            for col in column:
                col.var.setAttr(GRB.Attr.VType, GRB.INTEGER)

    def run(self, timelimit : int = 2000 ):
        
        self.model.setParam('TimeLimit', timelimit)

        # self.model.write("/home/marco/Desktop/DRMP.lp")

        self.model.optimize()

        status_drmp = self.model.getAttr(GRB.Attr.Status)

        if status_drmp == GRB.OPTIMAL:
            logging.info("[DISCRETE RMP] optimal solution found")
        elif self.model.getAttr(GRB.Attr.SolCount) > 0:
            logging.info("[DISCRETE RMP] sub-optimal solution found with status {status_drmp}")
        else:
            logging.info("[DISCRETE RMP] no solution found {status_drmp}")
            return None
        
        csv_out.log_master_vars(self.cData, self.model, phase="DRMP")
        
        new_sol = CorberanSolution(self.cData)

        for t, all_vars in self.chi_vars.items():
            for el in all_vars:
                if el.var.X > 1e-3:
                    for colvarname, val in el.nonzero_vars.items():
                        if colvarname.startswith("x") and val > 1e-3:
                            idx = get_index(colvarname)
                            i, j = int(idx[0]), int(idx[1])
                            new_sol.set_x(i, j, t, 1)
                        elif colvarname.startswith("y") and val > 1e-3:
                            idx = get_index(colvarname)
                            i = int(idx[0])
                            new_sol.set_y(i, t, 1)
                        elif colvarname.startswith("q") and val > 1e-3:
                            idx = get_index(colvarname)
                            i = int(idx[0])
                            new_sol.set_q(i, t, val)      

        for v in self.model.getVars():
            if v.X < 1e-5:
                continue
            if v.VarName.startswith('z'):
                idx = get_index(v.VarName)
                i, t = int(idx[0]), int(idx[1])
                new_sol.set_z(i, t, 1)
            elif v.VarName.startswith('w'):
                idx = get_index(v.VarName)
                i, j, t = int(idx[0]), int(idx[1]), int(idx[2])
                new_sol.set_w(i,j,t,v.X)

        new_sol.set_objvalue(self.model.Objval)
        new_sol.is_feasible = True
        self.last_sol = new_sol

        return new_sol
    