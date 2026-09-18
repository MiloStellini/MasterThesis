"""
run_all_merged_drmp.py

Lancia merge_seeds_drmp.py per ogni dataset x ogni frac (stesso schema di
run_all_ga.py). Di default moltiplicatori x1 (r=f=a=m=1) - per x10 basta
cambiare MULT qui sotto e GA_ROOT se serve puntare a logsmultx10.

*** DA ESEGUIRE SUL SERVER UNIMI *** (richiede licenza Gurobi valida e i log
del genetico gia' presenti sotto GA_ROOT). Controlla/aggiorna i path in CONFIG
prima di lanciare.
"""

import subprocess
from pathlib import Path

# ====== CONFIG DA SETTARE (stesso spirito di run_all_ga.py) ======
PROJECT_ROOT = Path(r"/home/mstellini/corberan/MasterThesis")      # cartella con merge_seeds_drmp.py
MERGE_SCRIPT = PROJECT_ROOT / "merge_seeds_drmp.py"

BASEDIR = Path(r"/home/mstellini/corberan/corberan_profiling_dataset")
DATASET_LIST = Path(r"/home/mstellini/corberan/nomi_dataset.txt")

# Cartella che contiene, per ogni istanza, sia <istanza>/ga_cl30_gen30/... sia
# <istanza>/pool_cl30/cg_cols.pkl - cioe' la stessa struttura di logs2/logs sul
# server (quella usata da run_all_ga.py per l'esperimento x1).
GA_ROOT = Path(r"/home/mstellini/corberan/logs")

OUT_ROOT = Path(r"/home/mstellini/corberan/logs_merged_seeds_x1")

FRACS = [0.00, 0.25, 0.50, 0.75, 1.00]
MULT = "1,1,1,1"     # x1: r,f,a,m. Per x10 (assign): "1,1,10,1" e GA_ROOT su logsmultx10
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

    for ds in datasets:
        for frac in FRACS:
            frac_pct = int(round(frac * 100))
            outdir = OUT_ROOT / ds / f"frac{frac_pct:03d}" / "MERGED_ALL_SEEDS"
            outdir.mkdir(parents=True, exist_ok=True)

            metrics_path = outdir / "metrics.json"
            if metrics_path.exists():
                print(f"SKIP (gia' fatto): {ds} frac={frac}")
                continue

            cmd = [
                "python", "-u", str(MERGE_SCRIPT),
                "--basedir", str(BASEDIR) + "/",
                "--instance", ds,
                "--frac", str(frac),
                "--ga-root", str(GA_ROOT),
                "--mult", MULT,
                "--cl", str(CL),
                "--timelimit", str(TIMELIMIT),
                "--outdir", str(outdir) + "/",
            ]
            try:
                run(cmd)
            except subprocess.CalledProcessError as e:
                print(f"ERRORE su {ds} frac={frac}: {e}")


if __name__ == "__main__":
    main()
