"""
reconstruct_ga_columns.py

Ricostruisce le colonne complete (y, q, x) prodotte da un singolo run del genetico
(NSGA-II) a partire dai CSV loggati da nsga.py (`..._nsga_y_q_<t>.csv`), che
contengono solo y (apertura) e q (capacita').

Per ogni individuo della popolazione finale del run:
  1. legge y, q dal CSV loggato
  2. ricarica i dati dell'istanza originale (stesso file .json/.csv usato dal run)
  3. ricostruisce x con la STESSA logica deterministica usata durante il run
     (x_build_from_yq, in nsga_config.py)
  4. ricalcola il costo per verifica incrociata col costo loggato

USO:
    # un singolo run
    python reconstruct_ga_columns.py --input <path/..._nsga_y_q_3.csv> --basedir <path/corberan_profiling_dataset/>

    # tutti i run sotto una cartella (ricorsivo)
    python reconstruct_ga_columns.py --input <cartella_logs2_o_logsmultx10> --basedir <path/corberan_profiling_dataset/> --recursive

Se --basedir non e' passato, lo script prova ad auto-individuarlo cercando una
cartella "corberan_profiling_dataset" a partire dalla root del progetto (di norma
sotto 01_code/legacy_corberan_checkout/corberan/).

Output: per ogni CSV di input <name>_nsga_y_q_<t>.csv, uno o due file accanto
all'originale:
  <name>_nsga_y_q_<t>_full_columns.csv   (una riga per individuo, x in forma sparsa)
  <name>_nsga_y_q_<t>_full_columns_x_long.csv   (opzionale, --long-x: una riga per
                                                  (individuo, facility, cliente) con
                                                  x != 0 -- puo' essere grande)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corberan_data import CorberanData
from nsga_config import x_build_from_yq

INSTANCE_RE = re.compile(r"amelia_N\d+_d\d+_T\d+")
FRAC_RE = re.compile(r"frac(\d{3})")
SEED_RE = re.compile(r"seed(\d+)")
FNAME_RE = re.compile(
    r"^(?P<prefix>.+?)_r(?P<r>\d+(?:\.\d+)?)_f(?P<f>\d+(?:\.\d+)?)"
    r"_a(?P<a>\d+(?:\.\d+)?)_m(?P<m>\d+(?:\.\d+)?)"
    r"_nsga_y_q_(?P<t>\d+)\.csv$"
)


def find_default_basedir(start: Path) -> Path | None:
    """Search upward/around `start` for a 'corberan_profiling_dataset' folder
    containing .json instance files (there is an extra nested duplicate level
    from the original .zip, e.g. .../corberan_profiling_dataset/corberan_profiling_dataset/)."""
    for root in [start] + list(start.parents):
        for cand in root.rglob("corberan_profiling_dataset"):
            if cand.is_dir():
                inner = cand / cand.name
                if inner.is_dir() and any(inner.glob("*.json")):
                    return inner
                if any(cand.glob("*.json")):
                    return cand
    return None


def parse_run_metadata(csv_path: Path):
    fname_match = FNAME_RE.match(csv_path.name)
    if not fname_match:
        return None

    full_str = str(csv_path)
    inst_match = INSTANCE_RE.search(full_str)
    frac_match = FRAC_RE.search(full_str)
    seed_match = SEED_RE.search(full_str)

    return {
        "instance": inst_match.group(0) if inst_match else None,
        "frac": int(frac_match.group(1)) if frac_match else None,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "mult_cr": float(fname_match.group("r")),
        "mult_cf": float(fname_match.group("f")),
        "mult_ca": float(fname_match.group("a")),
        "mult_cm": float(fname_match.group("m")),
        "t": int(fname_match.group("t")),
        "prefix": fname_match.group("prefix"),
    }


_CDATA_CACHE: dict[str, CorberanData] = {}


def load_cdata(basedir: Path, instance: str) -> CorberanData:
    key = str(basedir) + "|" + instance
    if key not in _CDATA_CACHE:
        basedir_str = str(basedir)
        if not basedir_str.endswith(("/", "\\")):
            basedir_str += "/"
        cData = CorberanData.from_json_params(basedir_str, instance)
        if cData is None:
            raise FileNotFoundError(
                f"Impossibile caricare l'istanza '{instance}' da basedir '{basedir_str}' "
                f"(atteso un file '{instance}.json' li' dentro)."
            )
        _CDATA_CACHE[key] = cData
    return _CDATA_CACHE[key]


def reconstruct_one_run(csv_path: Path, basedir: Path, top_k: int, scale: float, long_x: bool):
    meta = parse_run_metadata(csv_path)
    if meta is None:
        print(f"[SKIP] nome file non riconosciuto come run GA y_q: {csv_path.name}")
        return None
    if meta["instance"] is None:
        print(f"[SKIP] impossibile determinare l'istanza dal percorso: {csv_path}")
        return None

    cData = load_cdata(basedir, meta["instance"])
    cData.multipliers = {
        "cr": meta["mult_cr"],
        "cf": meta["mult_cf"],
        "ca": meta["mult_ca"],
        "cm": meta["mult_cm"],
    }

    df = pd.read_csv(csv_path)
    nF = cData.nF
    y_cols = [f"y_{f}" for f in range(nF)]
    q_cols = [f"q_{f}" for f in range(nF)]
    missing = [c for c in y_cols + q_cols if c not in df.columns]
    if missing:
        print(f"[SKIP] {csv_path.name}: colonne mancanti {missing[:4]}... "
              f"(nF atteso={nF}, il CSV ha {df.shape[1]} colonne totali)")
        return None

    t = meta["t"]

    out_rows = []
    long_x_rows = []

    for _, row in df.iterrows():
        y = row[y_cols].to_numpy(dtype=float)
        q = row[q_cols].to_numpy(dtype=float)

        x, feasible = x_build_from_yq(y, q, cData, t, top_k=top_k, scale=scale)

        y_bin = (y > 0.5).astype(float)
        cost_recomputed = float(cData.get_mult_cr() * np.dot(cData._cr, y_bin) +
                                 cData.get_mult_ca() * np.sum(cData.ca * x))
        if y_bin.sum() == 0:
            cost_recomputed += 1e5
        if not feasible:
            cost_recomputed = 1e6

        nz = np.argwhere(x > 1e-8)
        x_nonzero = [[int(f), int(c), round(float(x[f, c]), 6)] for f, c in nz]

        out_rows.append({
            "instance": meta["instance"],
            "mult_cr": meta["mult_cr"],
            "mult_cf": meta["mult_cf"],
            "mult_ca": meta["mult_ca"],
            "mult_cm": meta["mult_cm"],
            "frac": meta["frac"],
            "seed": meta["seed"],
            "t": t,
            "id": int(row["id"]) if "id" in df.columns else None,
            "cost_logged": float(row["cost"]) if "cost" in df.columns else None,
            "cost_recomputed": cost_recomputed,
            "feasible_x": feasible,
            "open_facilities": int(y_bin.sum()),
            "y": json.dumps(y.round(4).tolist()),
            "q": json.dumps(q.round(4).tolist()),
            "x_nonzero_f_c_val": json.dumps(x_nonzero),
        })

        if long_x:
            for f, c, val in x_nonzero:
                long_x_rows.append({
                    "instance": meta["instance"], "frac": meta["frac"], "seed": meta["seed"],
                    "t": t, "id": int(row["id"]) if "id" in df.columns else None,
                    "f": f, "c": c, "x": val,
                })

    out_df = pd.DataFrame(out_rows)
    out_path = csv_path.with_name(csv_path.stem + "_full_columns.csv")
    out_df.to_csv(out_path, index=False)

    if long_x and long_x_rows:
        long_path = csv_path.with_name(csv_path.stem + "_full_columns_x_long.csv")
        pd.DataFrame(long_x_rows).to_csv(long_path, index=False)

    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=str,
                     help="File '..._nsga_y_q_<t>.csv' oppure cartella da scandire")
    ap.add_argument("--basedir", type=str, default=None,
                     help="Cartella con i .json/.csv delle istanze Amelia "
                          "(default: auto-ricerca di 'corberan_profiling_dataset')")
    ap.add_argument("--recursive", action="store_true",
                     help="Se --input e' una cartella, cerca ricorsivamente")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1e6)
    ap.add_argument("--long-x", action="store_true",
                     help="Scrivi anche una tabella long-format per x (puo' essere grande)")
    args = ap.parse_args()

    input_path = Path(args.input)

    if args.basedir is not None:
        basedir = Path(args.basedir)
    else:
        basedir = find_default_basedir(Path(__file__).resolve())
        if basedir is None:
            print("ERRORE: --basedir non fornito e auto-ricerca fallita. "
                  "Passa esplicitamente --basedir <cartella con i .json delle istanze>.")
            sys.exit(1)
        print(f"[INFO] basedir auto-rilevato: {basedir}")

    if input_path.is_dir():
        pattern = "**/*_nsga_y_q_*.csv" if args.recursive else "*_nsga_y_q_*.csv"
        csv_files = sorted(p for p in input_path.glob(pattern) if "_full_columns" not in p.name)
    else:
        csv_files = [input_path]

    if not csv_files:
        print("Nessun file '..._nsga_y_q_<t>.csv' trovato.")
        sys.exit(1)

    print(f"[INFO] {len(csv_files)} run da processare")
    n_ok, n_skip = 0, 0
    for i, csv_path in enumerate(csv_files, 1):
        try:
            out_path = reconstruct_one_run(csv_path, basedir, args.top_k, args.scale, args.long_x)
        except Exception as e:
            print(f"[ERRORE] {csv_path}: {e}")
            n_skip += 1
            continue
        if out_path is None:
            n_skip += 1
        else:
            n_ok += 1
        if i % 50 == 0 or i == len(csv_files):
            print(f"[INFO] {i}/{len(csv_files)} processati ({n_ok} ok, {n_skip} skip)")

    print(f"[DONE] {n_ok} run ricostruiti, {n_skip} saltati.")


if __name__ == "__main__":
    main()
