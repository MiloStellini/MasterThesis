import os
import subprocess
from pathlib import Path

# ====== CONFIG DA SETTARE ======
PROJECT_ROOT = Path(r"C:\Users\milos\Desktop\tesi\corberan\algo\src")
MAIN_PY = PROJECT_ROOT / "main.py"

BASEDIR = Path(r"C:\Users\milos\Desktop\tesi\corberan\corberan_profiling_dataset\corberan_profiling_dataset")
DATASET_LIST = Path(r"C:\Users\milos\Desktop\tesi\nomi_dataset.txt")

OUT_ROOT = PROJECT_ROOT / r"logsx10"

ALG_HEUR = "heur_only"
MULTIPLIERS = str("1,1,10,1")

# ===============================

def read_datasets(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def run(cmd):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = read_datasets(DATASET_LIST)

    for ds in datasets:

        cmd = [
            "python", "-u", str(MAIN_PY),
            "-b", str(BASEDIR) + os.sep,
            "-f", ds,
            "-o", str(OUT_ROOT),
            "-a", ALG_HEUR,
            "-m", MULTIPLIERS
        ]
        run(cmd)

if __name__ == "__main__":
    main()

