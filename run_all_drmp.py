import os
import subprocess
from pathlib import Path

# ====== CONFIG DA SETTARE ======
PROJECT_ROOT = Path(r"C:\Users\milos\Desktop\tesi\git_codes")      # cartella dove sta main.py
MAIN_PY = PROJECT_ROOT / "main.py"

BASEDIR = Path(r"C:\Users\milos\Desktop\tesi\corberan\corberan_profiling_dataset\corberan_profiling_dataset")
DATASET_LIST = Path(r"C:\Users\milos\Desktop\tesi\nomi_dataset.txt")  # copia qui il txt allegato
COLS_ROOT = Path(r"C:\Users\milos\Desktop\tesi\logsmultx10\logsmultx10")
OUT_ROOT = Path(r"C:\Users\milos\Desktop\tesi\logsmultx10")

ALG = "drmp_only"
MULTIPLIERS = str("1,1,10,1")

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
        cols_dir = COLS_ROOT / ds / "pool_cl30"
        cols_pkl = cols_dir / "cg_cols.pkl"

        out_dir = OUT_ROOT / ds
        out_dir.mkdir(parents=True, exist_ok= True)

        cmd = [
            "python", "-u", str(MAIN_PY),
            "-b", str(BASEDIR) + os.sep,
            "-f", ds,
            "-o", str(out_dir) + os.sep,
            "-a", ALG,
            "-m", MULTIPLIERS,
            "--colsfile", str(cols_pkl)
        ]
        run(cmd)

if __name__ == "__main__":
    main()
