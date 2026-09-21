"""
run_all_merged_drmp_x10.py

Come run_all_merged_drmp.py, ma per l'esperimento x10 (moltiplicatore costo
assegnazione = 10): punta a logsmultx10 invece di logs2, e usa il pool CG
costruito apposta con a=10 (logsmultx10/<istanza>/pool_cl30/cg_cols.pkl).

*** DA ESEGUIRE SUL SERVER UNIMI *** (richiede licenza Gurobi valida e i log
del genetico gia' presenti sotto GA_ROOT). Controlla/aggiorna i path in CONFIG
prima di lanciare - in particolare GA_ROOT: deve essere la cartella logsmultx10
sul server (verificala, potrebbe non chiamarsi esattamente come sotto).
"""

import subprocess
import sys
from pathlib import Path

# ====== CONFIG DA SETTARE (stesso spirito di run_all_ga.py) ======
PROJECT_ROOT = Path(r"/home/mstellini/corberan/MasterThesis")      # cartella con merge_seeds_drmp.py
MERGE_SCRIPT = PROJECT_ROOT / "merge_seeds_drmp.py"

BASEDIR = Path(r"/home/mstellini/corberan/corberan_profiling_dataset")
DATASET_LIST = Path(r"/home/mstellini/corberan/nomi_dataset.txt")

# Cartella che contiene, per ogni istanza, sia <istanza>/ga_cl30_gen30/... sia
# <istanza>/pool_cl30/cg_cols.pkl - qui la cartella x10 (verifica il nome esatto
# sul server con: find /home/mstellini/corberan -maxdepth 1 -type d).
GA_ROOT = Path(r"/home/mstellini/corberan/logsmultx10")

OUT_ROOT = Path(r"/home/mstellini/corberan/logs_merged_seeds_x10")

FRACS = [0.00, 0.25, 0.50, 0.75, 1.00]
MULT = "1,1,10,1"     # r,f,a,m - a=10: costo di assegnazione moltiplicato per 10
CL = 30
TIMELIMIT = 2000
# ===============================


def read_datasets(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def run(cmd):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = read_datasets(DATASET_LIST)

    combos = [(ds, frac) for ds in datasets for frac in FRACS]
    total = len(combos)
    n_ok, n_skip, n_err = 0, 0, 0

    for i, (ds, frac) in enumerate(combos, 1):
        frac_pct = int(round(frac * 100))
        outdir = OUT_ROOT / ds / f"frac{frac_pct:03d}" / "MERGED_ALL_SEEDS"
        outdir.mkdir(parents=True, exist_ok=True)

        metrics_path = outdir / "metrics.json"
        if metrics_path.exists():
            print(f"[{i}/{total}] SKIP (gia' fatto): {ds} frac={frac}")
            n_skip += 1
            continue

        cmd = [
            sys.executable, "-u", str(MERGE_SCRIPT),
            "--basedir", str(BASEDIR) + "/",
            "--instance", ds,
            "--frac", str(frac),
            "--ga-root", str(GA_ROOT),
            "--mult", MULT,
            "--cl", str(CL),
            "--timelimit", str(TIMELIMIT),
            "--outdir", str(outdir) + "/",
        ]
        print(f"[{i}/{total}] START: {ds} frac={frac}")
        try:
            run(cmd)
            n_ok += 1
        except subprocess.CalledProcessError as e:
            n_err += 1
            print(f"[{i}/{total}] ERRORE su {ds} frac={frac}: {e}", file=sys.stderr)

    print(f"\n=== FINE SWEEP: {n_ok} completati, {n_skip} saltati (gia' fatti), {n_err} in errore su {total} totali ===")


if __name__ == "__main__":
    main()
