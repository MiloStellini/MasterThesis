import argparse
import logging
import os.path
from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import math

# from construction_heuristic import construction_heuristic
import toy_instances as toy
# from compact_model import CompactModel
from corberan_data import CorberanData, PricerTimeColumn
from corberan_solution import CorberanSolution
from DiscreteRMP import DiscreteRMP
from multiprocessing import Process, Queue
from typing import Dict
import time

import nsga as ga
from nsga_problems import facility_opening_config, facility_opening_yx_config, facility_opening_mix_config, facility_opening_yq_config
from nsga_config import split_chromosome, check_chrom_feasibility, columns_equal, x_build_from_yq

import outputs_logging as csv_out
import csv
import helper as hlp
import json

def dict_equal_float(d1, d2, tol=1e-5):
    if d1.keys() != d2.keys():
        return False
    return all(math.isclose(d1[k], d2[k], rel_tol=tol, abs_tol=tol) for k in d1)


def add_nonzero_term(column_obj, val, constraint):
    if abs(val) > 1e-5:
        column_obj.addTerms(val, constraint)

class PricerProcess:
    
    def __init__(self, 
                 worker_id: int, 
                 cData: CorberanData, 
                 rc_limit : float, 
                 gp_env : gp.Env = None):

        self.env = gp_env
        self.is_parallel = False
        if gp_env is None:
            self.env = gp.Env()
            self.is_parallel = True
        
        self.model = gp.Model(env=self.env, name=f"pricer_t{worker_id}")
        self.worker_id = worker_id
        self.data = cData
        self.duals = {}
        self.yvar = None
        self.xvar = None
        self.qvar = None
        self.rc_limit = rc_limit
        self.all_nonzeros = [] 

        self._build_pricer()
        
    def theta(self, name : str) -> float:
        return float(self.duals[name])
    
    def update(self, new_duals: Dict[str, float]):

        self.duals.update(new_duals)
        t = self.worker_id

        for f in range(self.data.nF):

            yvarobj = (self.data.get_mult_cr()*self.data.get_cr(f,t-1)
                    + self.theta(f"DW_T1_{f}_{t}")
                    - self.data.M * sum(self.theta( f"DW_T7_t_{k}_{f}_{t}") for k in range(self.data.nF) if k != f)
                    )
            if t + 1 in self.data.T:
                yvarobj = yvarobj - self.theta( f"DW_T1_{f}_{t + 1}") - self.data.M * sum(
                    self.theta( f"DW_T7_tprev_{f}_{k}_{t + 1}") for k in range(self.data.nF) if k != f)
            if abs(yvarobj) < 1e-5:
                yvarobj = 0
            self.yvar[f].Obj = yvarobj

            qvarobj = self.theta( f"DW_T5_{f}_{t}")
            if t + 1 in self.data.T:
                qvarobj = qvarobj - self.theta( f"DW_T5_{f}_{t + 1}") - self.theta( f"DW_T6_{f}_{t + 1}")
            if abs(qvarobj) < self.data.nonzero_tolerance:
                qvarobj = 0
            self.qvar[f].Obj = qvarobj

        self.model.optimize()
        status_pricer = self.model.getAttr(GRB.Attr.Status)

        if status_pricer == GRB.OPTIMAL:

            pricer_objval = self.model.Objval
            reduced_cost = pricer_objval - self.theta( f"DW_TCONV_{t}")
            if reduced_cost < - self.rc_limit:

                nz_vars = {v.VarName: v.X for v in self.model.getVars() if v.X > self.data.nonzero_tolerance}

                if any(dict_equal_float(prev, nz_vars)
                                for prev in self.all_nonzeros):
                    return None

                pY = np.zeros(self.data.nF, dtype=float)
                pQ = np.zeros(self.data.nF, dtype=float)
                pX = np.zeros((self.data.nF, self.data.nC), dtype=float)

                for f in range(self.data.nF):
                    yval = self.yvar[f].X
                    if yval > 1e-5:
                        pY[f] = yval

                    qval = self.qvar[f].X
                    if qval > self.data.nonzero_tolerance:
                        pQ[f] = qval

                    for c in range(self.data.nC):
                        xval = self.xvar[f, c].X
                        if xval > 1e-5:
                            pX[f, c] = xval

                self.all_nonzeros.append(nz_vars)

                return status_pricer, pY, pQ, pX, reduced_cost, nz_vars
            
        return status_pricer, None, None, None, None, None
    
    def cleanup(self):
        
        self.env.close()

    def _build_pricer(self):
        t = self.worker_id
        self.model.ModelSense = GRB.MINIMIZE

        self.model.setParam("LogFile",
                        str(self.data.log_out_dir) +"/" + self.data.outfilename + f"_gurobi_pricer_t{t}.log"
                        )

        # VARIABLES
        self.yvar = self.model.addVars(self.data.nF, vtype=GRB.BINARY, lb=0.0, ub=1.0, name="y", obj = 0)
        self.xvar = self.model.addVars(self.data.nF, self.data.nC, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x",
                                       obj = self.data.get_mult_ca() * self.data.ca)
        self.qvar = self.model.addVars(self.data.nF,
                            lb=0.0,
                            ub=self.data.M,
                            vtype=GRB.CONTINUOUS, name="q", obj = 0)

        # CONSTRAINTS
        for c in range(self.data.nC):
            if self.data.d[c, t] > 0:
                self.model.addConstr(sum(self.xvar[f, c] for f in range(self.data.nF)) >= 1.0, name=f"PR_T2_{c}")
            for f in range(self.data.nF):
                self.model.addConstr(self.xvar[f, c] <= self.yvar[f], name=f"PR_T3_{f}_{c}")

        for f in range(self.data.nF):
            self.model.addConstr(sum(self.data.d[c, t] * self.xvar[f, c] for c in range(self.data.nC)) <= self.qvar[f],
                                    name=f"PR_T4_{f}")

        ##################### Added constraint from base model ######################
        self.model.addConstr(self.qvar.sum() <= self.data.M, name=f"PR_T8")

        if self.is_parallel:
            self.model.Params.Threads = 1

def worker_process(worker_id: int, 
                   task_queue: Queue, result_queue: Queue, ready_queue: Queue,
                   rc_limit : float, 
                   cData : CorberanData):
    
    
    # Initialize the complex object once
    obj = PricerProcess(worker_id, cData, rc_limit)
    
    # Signal to master that this worker is ready
    ready_queue.put(worker_id)
    print(f"Worker {worker_id}: Ready and waiting for tasks")
    
    try:
        while True:
            try:
                # Wait for task from master
                task = task_queue.get()
                
                # Check for shutdown signal
                if task is None:
                    print(f"Worker {worker_id}: Received shutdown signal")
                    break
                
                target_worker_id, params = task
                
                # Only process if this task is for this worker
                if target_worker_id == worker_id:
                    # Process the task
                    result = obj.update(params)
                    
                    # Send result back to master
                    result_queue.put((worker_id, result))
                else:
                    # Put task back for other workers
                    task_queue.put(task)
                    time.sleep(0.01)  # Brief sleep to avoid busy waiting
                    
            except Exception as e:
                print(f"Worker {worker_id}: Error - {e}")
                result_queue.put((worker_id, str(e)))
    
    finally:
        # This runs on normal exit (after break or exception)
        print(f"Worker {worker_id}: Shutting down gracefully")
        obj.cleanup()

class CG_w_in_master:

    def __init__(self, data :CorberanData, 
                 run_parallel : bool = False):
        pass
        self.cData = data
        self.master_model = None
        self.all_constraints = dict()
        self.duals = dict()
        self.run_parallel = run_parallel

        self.task_queue = None
        self.result_queue = None
        self.ready_queue = None
        self.pricer_t_worker = None
        self.env = None

        self.build_master()

    def theta(self, name : str) -> float:
        return float(self.duals[name])

    def build_column(self, t, Y, Q, X,
                rc_exp : float = None
                ):

        check_rc = rc_exp is not None and self.cData.debug_info

        c = gp.Column()

        nF = Y.shape[0]
        nC = self.cData.ca.shape[1]
        rc = 0

        c.addTerms(1.0, self.all_constraints[f"DW_TCONV_{t}"])
        
        if check_rc:
            rc += - self.theta(f"DW_TCONV_{t}")

        for f in range(nF):
            add_nonzero_term(c, -Y[f], self.all_constraints[f"DW_T1_{f}_{t}"])
            if check_rc:
                rc += Y[f] * self.theta( f"DW_T1_{f}_{t}")

            add_nonzero_term(c, -Q[f], self.all_constraints[f"DW_T5_{f}_{t}"])
            if check_rc:
                rc += Q[f] * self.theta(f"DW_T5_{f}_{t}")

            if t+1 in self.cData.T:
                add_nonzero_term(c, Y[f], self.all_constraints[f"DW_T1_{f}_{t+1}"])
                add_nonzero_term(c, Q[f], self.all_constraints[f"DW_T5_{f}_{t+1}"])
                add_nonzero_term(c, Q[f], self.all_constraints[f"DW_T6_{f}_{t+1}"])
                if check_rc:
                    rc += (
                        - Y[f] * self.theta( f"DW_T1_{f}_{t+1}")
                        - Q[f] * self.theta( f"DW_T5_{f}_{t+1}")
                        - Q[f] * self.theta( f"DW_T6_{f}_{t+1}")
                    )

            for k in range(nF):
                if k != f:
                    add_nonzero_term(c, self.cData.M * Y[f], self.all_constraints[f"DW_T7_t_{k}_{f}_{t}"])
                    if check_rc:
                        rc += - self.cData.M * Y[f] * self.theta( f"DW_T7_t_{k}_{f}_{t}")
                    if t+1 in self.cData.T:
                        add_nonzero_term(c, self.cData.M * Y[f], self.all_constraints[f"DW_T7_tprev_{f}_{k}_{t+1}"])
                        if check_rc:
                            rc += -self.cData.M * Y[f] * self.theta( f"DW_T7_tprev_{f}_{k}_{t+1}")

        obj = sum( self.cData.get_mult_cr() * self.cData.get_cr(f,t-1) * Y[f] +
                sum( self.cData.get_mult_ca()*self.cData.ca[f, c] * X[f, c] for c in range(nC))
                for f in range(nF)
                )

        if check_rc and abs(rc + obj - rc_exp) > 1e-5:
            logging.error(f"timeslot {t}: reduced cost {rc_exp} and column cost {rc + obj} not matching")
            exit(-1)

        return c, obj


    def build_master(self):

        self.env = gp.Env()
        self.master_model = gp.Model(env=self.env, name="master")
        self.master_model.ModelSense = GRB.MINIMIZE

        self.master_model.setParam("LogFile",
                            str(self.cData.log_out_dir) +"/"+ self.cData.outfilename + "_gurobi_master.log")

        zvar = dict()
        wvar = dict()

        # Variables
        # add w
        for t in self.cData.T:
            for f1 in range(self.cData.nF - 1):
                for f2 in range(f1+1, self.cData.nF):
                
                    varname = f"w_{f1}_{f2}_{t}"
                    wvar[varname] = self.master_model.addVar(vtype=GRB.CONTINUOUS,
                                            name=varname,
                                            lb=0,
                                            ub=self.cData.M,
                                            obj=self.cData.get_mult_cm()*self.cData.cm[f1, f2]
                                            )

                    varname = f"w_{f2}_{f1}_{t}"
                    wvar[varname] = self.master_model.addVar(vtype=GRB.CONTINUOUS,
                                                        name=varname,
                                                        lb=0,
                                                        ub=self.cData.M,
                                                        obj=self.cData.get_mult_cm()*self.cData.cm[f2, f1]
                                                        )

            for f in range(self.cData.nF):
                varname = f"z_{f}_{t}"
                zvar[varname] = self.master_model.addVar(
                    vtype=GRB.CONTINUOUS,
                    name=varname,
                    lb=0,
                    ub=1,
                    obj=self.cData.get_mult_cf()*self.cData.get_cf(f, t-1)
                )

        # Constraints
        for t in self.cData.T:
            constr_name = f"DW_T8_{t}"
            self.all_constraints[constr_name] = self.master_model.addConstr(
                sum(wvar[f"w_{f2}_{f1}_{t}"] for f1 in range(self.cData.nF) for f2 in range(self.cData.nF) if f1 != f2) <= self.cData.M,
                constr_name
            )

            constr_name = f"DW_TCONV_{t}"
            self.all_constraints[constr_name] = self.master_model.addConstr(
                gp.LinExpr() >= 1,
                constr_name
            )

            for f in range(self.cData.nF):
                constr_name = f"DW_T1_{f}_{t}"
                self.all_constraints[constr_name] = self.master_model.addConstr(
                    zvar[f"z_{f}_{t}"] >= 0,
                    constr_name
                )
                constr_name = f"DW_T5_{f}_{t}"
                self.all_constraints[constr_name] = self.master_model.addConstr(
                    sum(- wvar[f"w_{f}_{k}_{t}"]+wvar[f"w_{k}_{f}_{t}"] for k in range(self.cData.nF) if k!=f)
                    >= 0,
                    constr_name
                )
                constr_name = f"DW_T6_{f}_{t}"
                self.all_constraints[constr_name] = self.master_model.addConstr(
                    sum(- wvar[f"w_{f}_{k}_{t}"] for k in range(self.cData.nF) if k!=f) >= 0,
                    constr_name
                )

                for k in range(self.cData.nF):
                    if k != f:
                        constr_name = f"DW_T7_tprev_{f}_{k}_{t}"
                        self.all_constraints[constr_name] = self.master_model.addConstr(
                            -wvar[f"w_{f}_{k}_{t}"] >= 0,
                            constr_name
                        )

                        constr_name = f"DW_T7_t_{f}_{k}_{t}"
                        self.all_constraints[constr_name] = self.master_model.addConstr(
                            -wvar[f"w_{f}_{k}_{t}"] >= 0,
                            constr_name
                        )

        self.master_model.update()

        for f in range(self.cData.nF):
            self.all_constraints[f"DW_T1_{f}_{1}"].rhs = self.all_constraints[f"DW_T1_{f}_{1}"].rhs - 1
            self.all_constraints[f"DW_T5_{f}_{1}"].rhs = self.all_constraints[f"DW_T5_{f}_{1}"].rhs - self.cData.q_in[f]
            self.all_constraints[f"DW_T6_{f}_{1}"].rhs = self.all_constraints[f"DW_T6_{f}_{1}"].rhs - self.cData.q_in[f]

            for k in range(self.cData.nF):
                if k != f:
                    self.all_constraints[f"DW_T7_tprev_{f}_{k}_{1}"].rhs = self.all_constraints[f"DW_T7_tprev_{f}_{k}_{1}"].rhs - self.cData.M

    def build_pricers_sequential(self, rc_limit : float):
        
        self.pricer_t_worker = { 
            t : PricerProcess(t, self.cData, rc_limit, self.env) 
                for t in self.cData.T 
            }

    def build_pricers_parallel(self, rc_limit : float):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.ready_queue = Queue()
        self.pricer_t_worker = dict()

        """Initialize and start all worker processes"""
        logging.info(f"Master: Starting workers...")
        
        for worker_id in self.cData.T:
            p = Process(
                target=worker_process,
                args=(worker_id, 
                      self.task_queue, 
                      self.result_queue, 
                      self.ready_queue, rc_limit, self.cData)
            )
            p.start()
            self.pricer_t_worker[worker_id] = p
        
        # Wait for all workers to signal they are ready
        logging.info("Master: Waiting for workers to initialize...")

        for _ in self.cData.T:
            worker_id = self.ready_queue.get()  # Blocks until a worker is ready
            logging.info(f"Master: Worker {worker_id} is ready")
        
        logging.info(f"Master: All workers are ready")

    def run_pricers_sequential(self):
        # PRICERS for each t
        pricer_results_parallel = {}
        for t in self.cData.T:
            # OBJECTIVE
            pricer_results_parallel[t] = self.pricer_t_worker[t].update(self.duals)

        return pricer_results_parallel          

    def run_pricers_parallel(self):
        for t in self.cData.T:
            self.task_queue.put((t, self.duals))

        pricer_results_parallel = {}
        # Get results
        for _ in self.cData.T:
            """Get result from any worker"""
            try:
                worker_id, result = self.result_queue.get()
                logging.info(f"Master: Received result from worker {worker_id}")
                pricer_results_parallel[worker_id] = result
            except Exception as e:
                logging.warning(f"Master: got exception {e}")

        return pricer_results_parallel

    def cg_loop(self, LR_epgap : float = 1e-5,
                rc_limit : float = 1e-5,
                timelimit : int = 1200
                ) -> CorberanSolution:

        # -----------------------------
        # Initial heuristic solution
        # -----------------------------

        heur_sol = CorberanSolution.construction_heuristic(self.cData)
        if not heur_sol.isfeasible:
            logging.error("\n".join(heur_sol.errorlog))
            return None

        chi_heur = {}
        
        # Per adesso ho indicizzazione da 0 anche per i tempi di X Y W Q Z quindi scalo con t-1 e t-2
        all_columns = defaultdict(list)
        for t in self.cData.T:

            c, obj_c = self.build_column(t,
                                heur_sol.yvarsol[:, t-1],
                                heur_sol.qvarsol[:, t-1],
                                heur_sol.xvarsol[:,:,t-1])

            nz_vars = {}
            for lab, varsol in [['y', heur_sol.yvarsol[:, t-1]],
                                ['q', heur_sol.qvarsol[:, t-1]],
                                ['x', heur_sol.xvarsol[:, :, t-1]]]:
                r = np.where(varsol > 1e-5)
                if len(r) == 1:
                    nz_vars.update( { f'{lab}_{el}_{t-1}' : varsol[el] for el in r[0] } )
                else:
                    a, b = r
                    nz_vars.update({f'{lab}_{a[i]}_{b[i]}_{t-1}': varsol[a[i], b[i]] for i in range(len(a))})

            colname = f"chi_heur_{t}"
            chi_heur[t] = self.master_model.addVar(lb=0.0, obj=obj_c, column=c, name=colname)

            all_columns[t].append(PricerTimeColumn(colname,
                                                   -1,
                                                   t,
                                                   nz_vars,
                                                   chi_heur[t]
                                                   ))

        cg_iter = 0
        # LR_gap_master = None

        # -----------------------------
        # Diagnostics helpers
        # -----------------------------
        MASTER_ITER_CSV = str(self.cData.log_out_dir) + "/" + self.cData.outfilename + "_master_iterations.csv"
        csv_out.write_csv(MASTER_ITER_CSV, [], header=["iter", "status", "objval", "LR_lb", "LR_gap", "columns_added", "master_time", "iter_time", "total_time", "ga_time", "ga_n_gen"])

        ####
        #### INIT T PRICERS WITHOUT OBJECTIVE FUN. COEFFICIENTS OF THE VARIABLES
        ####
        if self.run_parallel:
            self.build_pricers_parallel(rc_limit)
        else:
            self.build_pricers_sequential(rc_limit)

        # self.master_model.write(f"{BASEDIR}master_iter_init.lp")

        # -----------------------------
        # Column Generation loop
        # -----------------------------
        start_t_loop = time.time()
        self.master_model.setParam("Method", 1)
        try:
            while ((time.time() - start_t_loop) < timelimit) or not(self.cData.use_nsga and self.cData.nsga_min_cols_per_t == 0): # cg_iter < 150:
                # Force simplex for the master

                start_t_cgiter = time.time()
                self.master_model.optimize()

                status = self.master_model.getAttr(GRB.Attr.Status)

                if status == GRB.OPTIMAL:

                    master_time = time.time() - start_t_cgiter
                    master_LB = self.master_model.ObjVal

                    logging.info(f"MASTER [{cg_iter}] obj: {master_LB:.9g}")

                    # Dump duals
                    self.duals = {name: constr.Pi for name, constr in self.all_constraints.items()}

                    column_added = 0
                    LR_bound = master_LB

                    # TODO: run pricers
                    if self.run_parallel:
                        pricer_results_parallel = self.run_pricers_parallel()
                    else:
                        pricer_results_parallel = self.run_pricers_sequential()

                    for t, pricer_result in pricer_results_parallel.items():

                        if pricer_result is None:
                            logging.warning(f"COLUMN ALREADY GENERATED ITER {cg_iter} TIME-SLOT {t}.")                            
                            continue

                        status_pricer, pY, pQ, pX, reduced_cost, nz_vars = pricer_result

                        if status_pricer != GRB.OPTIMAL:
                            logging.warning(f"[{cg_iter}] t={t} pricer not optimal (status={status_pricer}).")
                            
                            exit()
                        elif reduced_cost is None:
                            continue

                        LR_bound += reduced_cost
                        c, obj_c = self.build_column(t, pY, pQ, pX)  # , duals, reduced_cost)

                        # Add var/column to master
                        col_name = f"Chi_iter{cg_iter}_t_{t}"
                        chi_var = self.master_model.addVar(vtype=GRB.CONTINUOUS,
                                                    lb=0, obj=obj_c,
                                                    column=c, name=col_name)
        
                        all_columns[t].append(PricerTimeColumn(col_name,
                                            cg_iter,
                                            t,
                                            nz_vars,
                                            chi_var
                                            ))
                        column_added += 1

                    LR_gap_master = abs(LR_bound-master_LB)/abs(1e-10+master_LB)

                    end_time = time.time()
                    iter_time = end_time - start_t_cgiter
                    total_cg_time = end_time - start_t_loop
                    log_iter_elements = [cg_iter, status, f"{master_LB:.9g}", f"{LR_bound:.9g}", f"{LR_gap_master:.9g}", column_added, master_time, iter_time, total_cg_time, 0.0, 0]
                    csv_out.write_csv(MASTER_ITER_CSV, [log_iter_elements])
                    logging.info("CG_ITER: " + ';'.join([str(el) for el in log_iter_elements]))
                    
                    if column_added == 0:
                        logging.info("No more columns with strictly negative reduced cost: CG iterations END")
                        break

                    if LR_gap_master < LR_epgap:
                        logging.info(f"LR GAP {LR_gap_master} LESS THAN THRESHOLD {LR_epgap}. END CG LOOP")
                        break

                else:
                    logging.error(f"Master not solved to optimality (status={status}). Stopping.")
                    csv_out.write_csv(MASTER_ITER_CSV, [[cg_iter, status, "nan", 0]])
                    break
                ##########################################################
                if self.cData.do_cg_pool:
                    enough_cols = True
                    for t in self.cData.T:
                        if len(all_columns[t]) < self.cData.nsga_min_cols_per_t:
                            enough_cols = False
                            break
                    if enough_cols:
                        logging.info("Stopping CG early: enough columns to store")
                        break                        
                ######################### ADDED #########################
                if self.cData.use_nsga:
                    enough_cols = True
                    for t in self.cData.T:
                        if len(all_columns[t]) < self.cData.nsga_min_cols_per_t:
                            enough_cols = False
                            break
                    if enough_cols:
                        logging.info("Stopping CG early: enough columns for all time-slots in GA mode")
                        break
                #########################################################

                cg_iter += 1
        finally:
            if self.run_parallel:
                for _ in self.cData.T:
                    self.task_queue.put(None)
            
                # Wait for all workers to finish
                for _, p in self.pricer_t_worker.items():
                    p.join(timeout=5)
                    if p.is_alive():
                        logging.info(f"Master: Force terminating worker {p.pid}")
                        p.terminate()

        if self.cData.debug_info:
            for t, columns in all_columns.items():
                for col in columns:
                    if col.replicas > 2:
                        logging.warning(f"column generated more than two times: {col.col_name} {col.replicas}")

        logging.info("Column Generation loop ended.")
        # out.cg_solution_json(f"{BASEDIR}CG_solutions.json", all_columns, zvar, get_index, T, self.master_model.ObjVal)

        if self.cData.do_cg_pool:
            config = facility_opening_yq_config
            pool = ga.extract_population(all_columns, self.cData, config)

            # dove salvare l'npz: se pool_file è settato uso quello, altrimenti default in outdir
            pool_path = self.cData.pool_file if getattr(self.cData, "pool_file", "") else os.path.join(self.cData.log_out_dir, "cg_pool.npz")
            hlp.save_population(pool, pool_path)
            hlp.save_columns(all_columns, os.path.join(self.cData.log_out_dir, "cg_cols.pkl"))

            # tempo CG per arrivare al pool (start_t_loop è già definito sopra)
            cg_pool_time = time.time() - start_t_loop

            # metrics.json
            import json
            metrics_path = os.path.join(self.cData.log_out_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({
                    "mode": "CG_POOL",
                    "dataset_prefix": self.cData.outfilename,
                    "cg_pool_time_sec": float(cg_pool_time),
                    "nsga_min_cols_per_t": int(self.cData.nsga_min_cols_per_t),
                    "nsga_pop_size": int(self.cData.nsga_pop_size),
                    "pool_file": str(pool_path)
                }, f, indent=2)

            return None

        
        ######################### ADDED #########################
        if self.cData.use_nsga:
            config = facility_opening_yq_config
            logging.info("GA start")

            ga_start = time.time()
            population = ga.extract_population(all_columns, self.cData, config)
            ga_results, ga_gens = ga.run_nsga(population, self.cData, config)
            ga_time = time.time() - ga_start
            logging.info("GA ended")

            rows = list(csv.reader(open(MASTER_ITER_CSV)))
            rows[-1][-2] = f"{ga_time}"
            rows[-1][-1] = f"{max(ga_gens.values())}"
            csv.writer(open(MASTER_ITER_CSV, "w", newline="")).writerows(rows)

            for t, ga_pop in ga_results.items():
                if ga_pop is None or len(ga_pop) == 0:
                    continue
                
                for i in range(ga_pop.shape[0]):
                    chrom = ga_pop[i, :]
                    # report = check_chrom_feasibility(chrom, self.cData, t)
                    # if (report["viol_coverage"] + report["viol_sum_q"] + report["viol_capacity"] != 0 or report["open_facilities"] == 0):
                    #     print(f"t{t}_i{i}_NOT FEASIBLE:")
                    #     print(report)

                    # y, q, x = split_chromosome(chrom, self.cData)
                    # y = y.astype(float)
                    # q = q.astype(float)
                    # x = x.astype(float)
                    y = chrom[:self.cData.nF].astype(float)
                    q = chrom[self.cData.nF:].astype(float)
                    x, feasible = x_build_from_yq(y, q, self.cData, t)
                    if not feasible:
                        continue
                    
                    c, obj_c = self.build_column(t, y, q, x)

                    nz_vars = {}
                    for lab, varsol in [['y', y], ['q', q]]:
                        r = np.where(varsol > 1e-5)
                        for el in r[0]:
                            nz_vars[f'{lab}_{el}_{t-1}'] = float(varsol[el])

                    r = np.where(x > 1e-5)
                    if len(r) == 1:
                        for el in r[0]:
                            nz_vars[f'x_{el}_{t-1}'] = float(x[el])

                    else:
                        a, b = r
                        for i in range(len(a)):
                            nz_vars[f'x_{a[i]}_{b[i]}_{t-1}'] = float(x[a[i], b[i]])

                    is_duplicate = any(
                        columns_equal(nz_vars, existing_col.nonzero_vars)
                        for existing_col in all_columns[t]
                    )

                    if is_duplicate:
                        logging.info(f"Skipping GA column t={t}, i={i}: duplicate pattern already present.")
                        continue
                    
                    colname = f"chi_GA_t{t}_ind{i}"
                    chi_var = self.master_model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0.0,
                        obj=obj_c,
                        column=c,
                        name=colname
                    )

                    all_columns[t].append(
                        PricerTimeColumn(
                            colname,
                            -2,
                            t,
                            nz_vars,
                            chi_var
                        )
                    )
            self.master_model.update()                     
            
        #########################################################

        logging.info("DRMP start")

        drpm = DiscreteRMP(self.master_model, all_columns, self.cData)
        dsol = drpm.run()
        logging.info("DRMP ended")

        return dsol
        #if dsol is not None and out_filepath != "":
        #    dsol.export_json(out_filepath)

    def ga_only(self, saved_cols, pool: dict, frac: float, seed: int):

        config = facility_opening_yq_config
        nF = int(self.cData.nF)
        P = int(self.cData.nsga_pop_size)

        def build_init_for_t(t: int, pool_mat: np.ndarray | None) -> np.ndarray:
            rng = np.random.default_rng(seed + 1000 * int(t))

            k = int(round(frac * P))
            k = max(0, min(P, k))

            if pool_mat is None or pool_mat.size == 0 or k == 0:
                from_pool = np.empty((0, 2*nF), dtype=float)
            else:
                pool_mat = pool_mat.astype(float)
                k_eff = min(k, pool_mat.shape[0])
                idx = rng.choice(pool_mat.shape[0], size=k_eff, replace=False)
                from_pool = pool_mat[idx, :]

            n_rand = P - from_pool.shape[0]
            rand_part = ga._create_YQ_initial_population(self.cData, n_rand, rng) if n_rand > 0 else np.empty((0, 2*nF), dtype=float)

            return np.vstack([from_pool, rand_part])

        all_columns = defaultdict(list)

        for t, cols in saved_cols.items():
            for i, col in enumerate(cols):
                y, q, x = hlp.rebuild_xyq_from_nonzero(col["nonzero_vars"], self.cData)
                c, obj_c = self.build_column(t, y, q, x)

                chi_var = self.master_model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, obj=obj_c, column=c, name=col["name"]
                )
                all_columns[t].append(PricerTimeColumn(col["name"], col["cg_iter"], t, col["nonzero_vars"], chi_var))
                
        self.master_model.update()

        # for t in self.cData.T:
        #     pool_mat = pool.get(t, None)
        #     if pool_mat is None or pool_mat.size == 0:
        #         continue

        #     for i in range(pool_mat.shape[0]):
        #         chrom = pool_mat[i, :].astype(float)

        #         y = chrom[:nF]
        #         q = chrom[nF:]

        #         x, feasible = x_build_from_yq(y, q, self.cData, t)
        #         if not feasible:
        #             continue

        #         c, obj_c = self.build_column(t, y, q, x)
                
        #         nz_vars = {}
        #         for f in np.where(y > 1e-5)[0]:
        #             nz_vars[f"y_{int(f)}_{t-1}"] = float(y[f])
        #         for f in np.where(q > 1e-5)[0]:
        #             nz_vars[f"q_{int(f)}_{t-1}"] = float(q[f])
        #         a, b = np.where(x > 1e-5)
        #         for kk in range(len(a)):
        #             nz_vars[f"x_{int(a[kk])}_{int(b[kk])}_{t-1}"] = float(x[a[kk], b[kk]])

        #         colname = f"chi_pool_t{t}_ind{i}"
        #         chi_var = self.master_model.addVar(
        #             vtype=GRB.CONTINUOUS, lb=0.0, obj=obj_c, column=c, name=colname
        #         )
        #         all_columns[t].append(PricerTimeColumn(colname, -10, t, nz_vars, chi_var))

        population = {}
        for t in self.cData.T:
            init_pop = build_init_for_t(t, pool.get(t, None))
            population[t] = [init_pop[i, :] for i in range(init_pop.shape[0])]

        logging.info(f"[GA_ONLY] start GA: frac={frac}, seed={seed}, pop_size={P}")

        ga_start = time.time()
        ga_results, ga_gens = ga.run_nsga(population, self.cData, config)
        ga_time = time.time() - ga_start

        logging.info(f"[GA_ONLY] end GA: time={ga_time:.3f}s, max_gen={max(ga_gens.values()) if ga_gens else 0}")

        for t, ga_pop in ga_results.items():
            if ga_pop is None or len(ga_pop) == 0:
                continue

            if ga_pop.ndim == 1:
                ga_pop = ga_pop.reshape(1, -1)

            for i in range(ga_pop.shape[0]):
                chrom = ga_pop[i, :].astype(float)
                y = chrom[:nF]
                q = chrom[nF:]

                x, feasible = x_build_from_yq(y, q, self.cData, t)
                if not feasible:
                    continue

                c, obj_c = self.build_column(t, y, q, x)

                nz_vars = {}
                for f in np.where(y > 1e-5)[0]:
                    nz_vars[f"y_{int(f)}_{t-1}"] = float(y[f])
                for f in np.where(q > 1e-5)[0]:
                    nz_vars[f"q_{int(f)}_{t-1}"] = float(q[f])
                a, b = np.where(x > 1e-5)
                for kk in range(len(a)):
                    nz_vars[f"x_{int(a[kk])}_{int(b[kk])}_{t-1}"] = float(x[a[kk], b[kk]])

                if any(columns_equal(nz_vars, col.nonzero_vars) for col in all_columns[t]):
                    continue

                colname = f"chi_GA_t{t}_ind{i}"
                chi_var = self.master_model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, obj=obj_c, column=c, name=colname
                )
                all_columns[t].append(PricerTimeColumn(colname, -2, t, nz_vars, chi_var))

        self.master_model.update()

        logging.info("[GA_ONLY] DRMP start")
        drpm = DiscreteRMP(self.master_model, all_columns, self.cData)
        dsol = drpm.run()
        logging.info("[GA_ONLY] DRMP end")

        metrics_path = os.path.join(self.cData.log_out_dir, "metrics.json")

        obj = None
        status = None
        if dsol is not None:
            obj = float(getattr(dsol, "objval", getattr(dsol, "objVal", None)) or getattr(dsol, "ObjVal", None) or 0.0)
            status = str(getattr(dsol, "status", ""))

        with open(metrics_path, "w") as f:
            json.dump({
                "mode": "GA_ONLY",
                "dataset_prefix": self.cData.outfilename,
                "seed": int(seed),
                "pool_frac": float(frac),
                "ga_time_sec": float(ga_time),
                "ga_n_gen_max": int(max(ga_gens.values())) if ga_gens else 0,
                "drmp_obj": obj,
                "sol_status": status,
                "pool_file": str(getattr(self.cData, "pool_file", ""))
            }, f, indent=2)
        return dsol

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='multi-time facility location allocation',
        description='command line interface for the multi-time facility location allocation')

    parser.add_argument("-b",
                        "--basedir",
                        dest="basedir", type=str, help="str: Base directory for input",
                        required=True)

    parser.add_argument("-g",
                        "--epgap",
                        dest="epgap",
                        type=float,
                        help="float: LR gap threshold to stop CG loop. default to 1e-5",
                        default =1e-5,
                        required=False)

    parser.add_argument("-r",
                        "--redcostlimit",
                        dest="redcostlimit",
                        type=float,
                        help="float: threshold to consider a reduced cost to be negative. default to 1e-5",
                        default =1e-5,
                        required=False)

    parser.add_argument("-e",
                        "--example",
                        dest="runexample",
                        type=str,
                        help="str: run one of the toy examples: input 1, 2 or amelia",
                        default="",
                        required=False)

    parser.add_argument("-o",
                        "--outdir",
                        dest="outdir",
                        type=str,
                        help="str: Output directory for logging",
                        required=True)

    parser.add_argument("-f",
                        "--prefix",
                        dest="fileprefix",
                        type=str,
                        default="",
                        help="str: Input/output file prefix",
                        required=False)

    parser.add_argument("-v",
                        "--verbose",
                        dest="isverbose",
                        action="store_true",
                        help = "increase output verbosity",
                        required=False)

    parser.add_argument("-p",
                    "--parallel",
                    dest="isparallel",
                    action="store_true",
                    help = "run pricers in parallel",
                    required=False)

    parser.add_argument("-z",
                        "--nonzerotolerance",
                        dest="nonzerotolerance",
                        type=float,
                        default=1e-5,
                        help="float: tolerance to consider a non-zero variable. default to 1e-5",
                        required=False)

    _cmd_options = parser.parse_args()

    # BASEDIR = "/home/marco/Documents/work/corberan/logs/"

    # -----------------------------
    # Logging setup
    # -----------------------------

    RUN_DIR = Path(_cmd_options.outdir)
    RUN_DIR.mkdir(exist_ok=True, parents=True)

    logfilename = "run.log"
    if  _cmd_options.fileprefix != "":
        logfilename = _cmd_options.fileprefix + ".log"

    logging.basicConfig(
        filename=f"{_cmd_options.outdir}{logfilename}",
        filemode="w",
        format="{asctime}.s{msecs:.0f} - {levelname} - {message}",
        style="{",
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    # READ DATA
    # self.cData = set_p.example2()
    _cData = None

    if _cmd_options.runexample == "1":
        _cData = toy.example1()
    elif _cmd_options.runexample == "2":
        _cData = toy.example2()
    elif _cmd_options.runexample == "amelia":
        _cData = toy.dataset_amelia()

    if _cData is None:
        if _cmd_options.basedir != "" and _cmd_options.fileprefix != "" and os.path.exists(_cmd_options.basedir):

            _cData = CorberanData.from_json_params(
                os.path.join(_cmd_options.basedir, _cmd_options.fileprefix) + ".json")

    if _cData is not None:
        _cData.set_log_out_dir(RUN_DIR)
        _cData.debug_info = _cmd_options.isverbose
        _cData.nonzero_tolerance = _cmd_options.nonzerotolerance
        cg_algo = CG_w_in_master(_cData, _cmd_options.isparallel)
        sol = cg_algo.cg_loop(
           _cmd_options.epgap, 
           _cmd_options.redcostlimit
        )
        if sol is not None:
            sol.export_json(os.path.join(_cmd_options.outdir, _cmd_options.fileprefix + "_sol.json"))

    else:
        logging.error("no input chosen. exit execution.")

        
