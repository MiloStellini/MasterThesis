import json
import csv
from pathlib import Path
import logging
# from gurobipy import Model, GRB
from collections import defaultdict

def compact_solution_json(path, model, nonzero_only=True, tol=1e-8):
    
    variables = {}
    for v in model.getVars():
        val = float(v.X)
        if (not nonzero_only) or (abs(val) > tol):
            variables[v.VarName] = val

    objective = f"Objective value: {model.ObjVal}"
    with open(path, "w") as f:
        json.dump(objective, f)
        json.dump(variables, f, indent=2)
    

def cg_solution_json(path, columns, zvars, get_index, T, objVal):

    Y = defaultdict(float)
    X = defaultdict(float)
    W = defaultdict(float)
    Q = defaultdict(float)
    Z = defaultdict(float)

    for t in T:
        for col in columns[t]:
            chi = col["chi_var"].X
            if abs(chi) < 1e-8:
                continue

            for varname, val in col["pattern"]:
                if varname.startswith("y["):
                    f = get_index(varname)[0]
                    Y[(f, t)] += chi * val

                elif varname.startswith("x["):
                    f, c = get_index(varname)
                    X[(f, c, t)] += chi * val

                elif varname.startswith("w["):
                    f, l = get_index(varname)
                    W[(f, l, t)] += chi * val

                elif varname.startswith("q["):
                    f = get_index(varname)[0]
                    Q[(f, t)] += chi * val          

    for zname, zvar in zvars.items():
        f, t = get_index(zname)
        val = zvar.X
        if abs(val) > 1e-8:
            Z[(f, t)] = val    

    solution = {}
    for (f, t), val in Y.items():
        if abs(val) > 1e-8:
            solution[f"y[{f},{t}]"] = val
    for (f, c, t), val in X.items():
        if abs(val) > 1e-8:
            solution[f"x[{f},{c},{t}]"] = val
    for (f, l, t), val in W.items():
        if abs(val) > 1e-8:
            solution[f"w[{f},{l},{t}]"] = val
    for (f, t), val in Q.items():
        if abs(val) > 1e-8:
            solution[f"q[{f},{t}]"] = val
    for (f, t), val in Z.items():
        if abs(val) > 1e-8:
            solution[f"z[{f},{t}]"] = val

    objective = f"Objective value: {objVal}"
    with open(path, "w") as f:
        json.dump(objective, f)
        json.dump(solution, f, indent=2)


def write_csv(path, rows, header=None):
    path = Path(path)
    if header or (not path.exists() or path.stat().st_size == 0):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            if rows:
                w.writerows(rows)
    else:
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if rows:
                w.writerows(rows)

def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def log_pricer_vars(RUN_DIR, iter_idx: int, t: int, nz_vars):
    # CSV dump of all non-zero variables
    rows = []
    for name, val in nz_vars.items():
        rows.append([iter_idx, t, name, f"{val:.4g}"])
    write_csv(f"{RUN_DIR}/pricer_nz_vars.csv", rows, header=["iter", "t", "var", "value"])

def log_column_steps(RUN_DIR, iter_idx: int, t: int, steps, c_cap, pricer_obj):
    # steps: list of dicts {constr, coeff, reason}
    path = f"{RUN_DIR}/column_iter{iter_idx}_t{t}.json"
    dump_json(path, {
        "iter": iter_idx,
        "t": t,
        "c_cap": c_cap,
        "pricer_objective": pricer_obj,
        "steps": steps,
    })
    logging.info(f"[{iter_idx}] t={t} column construction logged to {path}")

def log_master_vars(cData, model, phase="DRMP", nonzero_only= True, tol=1e-6):
    outdir = Path(str(cData.log_out_dir))
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"{cData.outfilename}_master_vars.csv"

    rows = []
    for v in model.getVars():
        if nonzero_only and abs(float(v.X)) <= tol:
            continue

        rows.append([phase, v.VarName, float(v.X)])
        header = ["phase", "name", "value"]

        write_csv(str(outpath), rows, header=header)
        logging.info("[LOG_MASTER_VARS]")

