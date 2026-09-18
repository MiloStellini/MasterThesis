"""
merge_seeds_drmp.py

Per una combinazione (istanza, frac) unisce le colonne del genetico gia' salvate
di TUTTI i seed in un unico DRMP, insieme alle colonne iniziali del pool CG
(stessa logica/soglia usata da CG_w_in_master.ga_only per una singola run), e
risolve il DRMP risultante.

E' l'equivalente di quello che veniva fatto per ogni singolo seed (ga_only), ma
mettendo tutte le colonne di tutti i seed nello stesso DRMP invece di risolverne
uno per seed.

Le colonne del genetico sono salvate su disco solo in forma ridotta (y, q): qui
si ricostruisce x con la stessa funzione deterministica usata durante i run
originali (x_build_from_yq in nsga_config.py), esattamente come in
reconstruct_ga_columns.py.

USO:
    python merge_seeds_drmp.py \
        --basedir <path/corberan_profiling_dataset/> \
        --instance amelia_N10_d25_T5 \
        --frac 0.25 \
        --ga-root <path/logs2/logs> \
        --outdir <cartella_output>/

Di default usa tutti i seed trovati sotto
    <ga-root>/<instance>/ga_cl30_gen30/frac0XX/seed*/
Passa --seeds 0,1,2,... per limitarti a un sottoinsieme.

Di default moltiplicatori r=f=a=m=1 (x1). Passa --mult r,f,a,m per altre
configurazioni (es. --mult 1,1,10,1 per x10).

OUTPUT (stessa convenzione usata finora, dentro --outdir):
    <outfilename>.log                 log testuale (istanza, colonne caricate, tempi)
    <outfilename>_gurobi_master.log   log Gurobi della costruzione del master
    <outfilename>_DRMP.log            log Gurobi della risoluzione del DRMP (obiettivo, tempo)
    <outfilename>_master_vars.csv     valore di tutte le variabili non nulle della soluzione
    metrics.json                      riepilogo: tempi, obiettivo, status, colonne incluse
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))

import helper as hlp
from corberan_data import CorberanData, PricerTimeColumn
from CG_w_in_master import CG_w_in_master
from DiscreteRMP import DiscreteRMP
from nsga_config import x_build_from_yq, columns_equal


def load_cdata(basedir: Path, instance: str) -> CorberanData:
    basedir_str = str(basedir)
    if not basedir_str.endswith(("/", "\\")):
        basedir_str += "/"
    cData = CorberanData.from_json_params(basedir_str, instance)
    if cData is None:
        raise FileNotFoundError(
            f"Impossibile caricare l'istanza '{instance}' da basedir '{basedir_str}' "
            f"(atteso un file '{instance}.json' li' dentro)."
        )
    return cData


def add_initial_cg_columns(cg_algo: CG_w_in_master, cols_path: Path, frac: float, cl: int):
    saved_cols = hlp.load_columns(str(cols_path))
    all_columns = defaultdict(list)
    n_added = 0
    for t, cols in saved_cols.items():
        for i, col in enumerate(cols):
            if col["name"].startswith("chi_heur") or i < frac * cl:
                y, q, x = hlp.rebuild_xyq_from_nonzero(col["nonzero_vars"], cg_algo.cData)
                c, obj_c = cg_algo.build_column(t, y, q, x)
                chi_var = cg_algo.master_model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, obj=obj_c, column=c, name=col["name"]
                )
                all_columns[t].append(
                    PricerTimeColumn(col["name"], col["cg_iter"], t, col["nonzero_vars"], chi_var)
                )
                n_added += 1
    cg_algo.master_model.update()
    return all_columns, n_added


def add_ga_columns_all_seeds(cg_algo: CG_w_in_master, all_columns, frac_dir: Path, seeds: list[int]):
    cData = cg_algo.cData
    nF = int(cData.nF)
    y_cols = [f"y_{f}" for f in range(nF)]
    q_cols = [f"q_{f}" for f in range(nF)]

    n_added, n_dupe, n_infeasible, n_missing = 0, 0, 0, 0

    for seed in seeds:
        seed_dir = frac_dir / f"seed{seed}"
        if not seed_dir.is_dir():
            logging.warning(f"[MERGE] seed dir non trovata: {seed_dir}")
            continue

        for t in cData.T:
            matches = sorted(seed_dir.glob(f"*_nsga_y_q_{t}.csv"))
            matches = [p for p in matches if "_full_columns" not in p.name]
            if not matches:
                logging.warning(f"[MERGE] CSV nsga_y_q mancante per seed={seed} t={t} in {seed_dir}")
                n_missing += 1
                continue

            df = pd.read_csv(matches[0])
            missing_cols = [c for c in y_cols + q_cols if c not in df.columns]
            if missing_cols:
                logging.warning(f"[MERGE] colonne mancanti in {matches[0].name}: {missing_cols[:4]}...")
                continue

            for _, row in df.iterrows():
                y = row[y_cols].to_numpy(dtype=float)
                q = row[q_cols].to_numpy(dtype=float)

                x, feasible = x_build_from_yq(y, q, cData, t)
                if not feasible:
                    n_infeasible += 1
                    continue

                nz_vars = {}
                for f in np.where(y > 1e-5)[0]:
                    nz_vars[f"y_{int(f)}_{t - 1}"] = float(y[f])
                for f in np.where(q > 1e-5)[0]:
                    nz_vars[f"q_{int(f)}_{t - 1}"] = float(q[f])
                a, b = np.where(x > 1e-5)
                for kk in range(len(a)):
                    nz_vars[f"x_{int(a[kk])}_{int(b[kk])}_{t - 1}"] = float(x[a[kk], b[kk]])

                if any(columns_equal(nz_vars, col.nonzero_vars) for col in all_columns[t]):
                    n_dupe += 1
                    continue

                c, obj_c = cg_algo.build_column(t, y, q, x)
                ind_id = int(row["id"]) if "id" in df.columns else len(all_columns[t])
                colname = f"chi_GA_seed{seed}_t{t}_ind{ind_id}"
                chi_var = cg_algo.master_model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, obj=obj_c, column=c, name=colname
                )
                all_columns[t].append(PricerTimeColumn(colname, -2, t, nz_vars, chi_var))
                n_added += 1

    cg_algo.master_model.update()
    return n_added, n_dupe, n_infeasible, n_missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basedir", required=True, help="cartella con i .json/.csv delle istanze Amelia")
    ap.add_argument("--instance", required=True, help="es. amelia_N10_d25_T5")
    ap.add_argument("--frac", type=float, required=True, help="frazione pool CG: 0.0, 0.25, 0.5, 0.75, 1.0")
    ap.add_argument("--ga-root", required=True,
                     help="cartella che contiene <instance>/ga_cl30_gen30/... e <instance>/pool_cl30/")
    ap.add_argument("--seeds", default=None,
                     help="lista seed separata da virgole (default: tutti quelli trovati)")
    ap.add_argument("--cl", type=int, default=30, help="nsga_min_cols_per_t / dimensione pool CG (default 30)")
    ap.add_argument("--mult", default="1,1,1,1", help="moltiplicatori r,f,a,m (default 1,1,1,1 = x1)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--timelimit", type=int, default=2000, help="time limit (s) per il DRMP")
    args = ap.parse_args()

    frac_pct = int(round(args.frac * 100))
    frac_dirname = f"frac{frac_pct:03d}"

    instance_dir = Path(args.ga_root) / args.instance
    frac_dir = instance_dir / "ga_cl30_gen30" / frac_dirname
    pool_dir = instance_dir / "pool_cl30"
    cols_path = pool_dir / "cg_cols.pkl"

    if not frac_dir.is_dir():
        print(f"ERRORE: cartella frac non trovata: {frac_dir}")
        sys.exit(1)
    if not cols_path.is_file():
        print(f"ERRORE: cg_cols.pkl non trovato: {cols_path}")
        sys.exit(1)

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = sorted(int(p.name.replace("seed", "")) for p in frac_dir.glob("seed*") if p.is_dir())
    if not seeds:
        print(f"ERRORE: nessun seed trovato sotto {frac_dir}")
        sys.exit(1)

    mult_values = [float(v) for v in args.mult.split(",")]
    multipliers = {"cr": mult_values[0], "cf": mult_values[1], "ca": mult_values[2], "cm": mult_values[3]}
    mult_label = "r%d_f%d_a%d_m%d" % (multipliers["cr"], multipliers["cf"], multipliers["ca"], multipliers["cm"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfilename = f"{args.instance}_ga_merged_{mult_label}_{frac_dirname}"

    logging.basicConfig(
        filename=str(outdir / f"{outfilename}.log"),
        filemode="w",
        format="{asctime}.s{msecs:.0f} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(f"[MERGE] istanza={args.instance} frac={args.frac} mult={mult_label} seed={seeds}")

    t0 = time.time()

    cData = load_cdata(Path(args.basedir), args.instance)
    cData.multipliers = multipliers
    cData.set_log_out_dir(str(outdir))
    cData.outfilename = outfilename
    cData.nsga_min_cols_per_t = args.cl
    cData.debug_info = False
    cData.nonzero_tolerance = 1e-5

    logging.info("data read with success")

    cg_algo = CG_w_in_master(cData, run_parallel=False)

    all_columns, n_init = add_initial_cg_columns(cg_algo, cols_path, args.frac, args.cl)
    logging.info(f"[MERGE] colonne iniziali CG/heur aggiunte: {n_init} (frac={args.frac}, cl={args.cl})")

    n_added, n_dupe, n_infeasible, n_missing = add_ga_columns_all_seeds(cg_algo, all_columns, frac_dir, seeds)
    reconstruct_time = time.time() - t0
    logging.info(
        f"[MERGE] colonne GA aggiunte={n_added} duplicate_scartate={n_dupe} "
        f"infeasible_scartate={n_infeasible} csv_mancanti={n_missing} (da {len(seeds)} seed)"
    )

    logging.info("[MERGE] DRMP start")
    drmp_start = time.time()
    drmp = DiscreteRMP(cg_algo.master_model, all_columns, cData)
    dsol = drmp.run(timelimit=args.timelimit)
    drmp_time = time.time() - drmp_start
    logging.info("[MERGE] DRMP end")

    obj = None
    status = None
    if dsol is not None:
        obj = float(getattr(dsol, "objval", getattr(dsol, "objVal", None)) or getattr(dsol, "ObjVal", None) or 0.0)
        status = str(getattr(dsol, "status", ""))

    total_time = time.time() - t0

    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "GA_MERGED_SEEDS",
            "dataset_prefix": outfilename,
            "instance": args.instance,
            "frac": args.frac,
            "mult": mult_label,
            "seeds_included": seeds,
            "n_seeds": len(seeds),
            "n_initial_cols": n_init,
            "n_ga_cols_added": n_added,
            "n_ga_cols_dupe_skipped": n_dupe,
            "n_ga_cols_infeasible_skipped": n_infeasible,
            "n_ga_csv_missing": n_missing,
            "reconstruct_time_sec": reconstruct_time,
            "drmp_time_sec": drmp_time,
            "total_time_sec": total_time,
            "drmp_obj": obj,
            "sol_status": status,
        }, f, indent=2)

    logging.info(f"[MERGE] done. obj={obj} total_time={total_time:.2f}s -> {metrics_path}")
    print(f"[DONE] {args.instance} frac={args.frac}: obj={obj} total_time={total_time:.2f}s "
          f"(init={n_init}, ga_added={n_added}, dupe={n_dupe}, infeasible={n_infeasible})")


if __name__ == "__main__":
    main()
