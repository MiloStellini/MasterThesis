import os
import subprocess
from pathlib import Path

# ====== CONFIG DA SETTARE ======
PROJECT_ROOT = Path(r"C:\Users\milos\Desktop\tesi\corberan\algo\src")      # cartella dove sta main.py
MAIN_PY = PROJECT_ROOT / "main.py"

BASEDIR = Path(r"C:\Users\milos\Desktop\tesi\corberan\corberan_profiling_dataset\corberan_profiling_dataset")
DATASET_LIST = Path(r"C:\Users\milos\Desktop\tesi\nomi_dataset.txt")  # copia qui il txt allegato

OUT_ROOT = PROJECT_ROOT / r"logs\ga_logs"

ALG_CG = "cg_seq"      # oppure "cg_par"
ALG_GA = "ga_only"

POP_SIZE = 30
N_GEN = 30
TILIM = 1200

SEEDS = [0, 1, 2]
FRACS = [0.00, 0.25, 0.50, 0.75, 1.00]
# ===============================

def read_datasets(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def run(cmd):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = read_datasets(DATASET_LIST)

    for ds in datasets:
        ds_root = OUT_ROOT / ds
        ds_root.mkdir(parents=True, exist_ok=True)

        # ---- 1) CG_POOL una sola volta ----
        pool_dir = ds_root / f"pool_cl{POP_SIZE}"
        pool_dir.mkdir(parents=True, exist_ok=True)
        pool_npz = pool_dir / "cg_pool.npz"
        cols_pkl = pool_dir / "cg_cols.pkl"

        if not pool_npz.exists():
            cmd = [
                "python", "-u", str(MAIN_PY),
                "-b", str(BASEDIR) + os.sep,
                "-f", ds,
                "-o", str(pool_dir) + os.sep,
                "-a", ALG_CG,
                "-t", str(TILIM),
                "-pop", str(POP_SIZE),
                "-gen", str(N_GEN),
                "-cl", str(POP_SIZE),
                "--cgpool",
                "--poolfile", str(pool_npz),
                "--colsfile", str(cols_pkl)
            ]
            run(cmd)

        # ---- 2) GA_ONLY per frac x seed ----
        ga_root = ds_root / f"ga_cl{POP_SIZE}_gen{N_GEN}"
        ga_root.mkdir(parents=True, exist_ok=True)

        for frac in FRACS:
            frac_dir = ga_root / f"frac{int(frac*100):03d}"
            frac_dir.mkdir(parents=True, exist_ok=True)

            for seed in SEEDS:
                run_dir = frac_dir / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "python", "-u", str(MAIN_PY),
                    "-b", str(BASEDIR) + os.sep,
                    "-f", ds,
                    "-o", str(run_dir) + os.sep,
                    "-a", ALG_GA,
                    "-t", str(TILIM),
                    "-pop", str(POP_SIZE),
                    "-gen", str(N_GEN),
                    "--poolfile", str(pool_npz),
                    "--colsfile", str(cols_pkl),
                    "--poolfrac", str(frac),
                    "--seed", str(seed),
                ]
                run(cmd)

if __name__ == "__main__":
    main()
