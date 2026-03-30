import os
import subprocess
from pathlib import Path

# ====== CONFIG DA SETTARE ======
PROJECT_ROOT = Path(r"/home/mstellini/corberan/MasterThesis")      # cartella dove sta main.py
MAIN_PY = PROJECT_ROOT / "main.py"

BASEDIR = Path(r"/home/mstellini/corberan/corberan_profiling_dataset")
DATASET_LIST = Path(r"/home/mstellini/corberan/nomi_dataset.txt")  # copia qui il txt allegato
OUT_ROOT = Path(r"/home/mstellini/corberan/logs/CG_final")

ALG = "cg_par"
MULTIPLIERS = str("1,1,1,1")

# Limite globale in secondi per l'intero run (incluso CG loop, pricing, ecc.)
GLOBAL_TILIM = 1200

# ===============================

def read_datasets(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def run(cmd, timeout: int):
    print("RUN:", " ".join(map(str, cmd)))
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,   # <-- limite globale sul processo figlio
        )
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Il run ha superato {timeout}s ed è stato terminato: {cmd}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Il run è terminato con errore (returncode={e.returncode}): {cmd}")

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = read_datasets(DATASET_LIST)

    for ds in datasets:
        out_dir = OUT_ROOT / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3", "-u", str(MAIN_PY),
            "-b", str(BASEDIR) + os.sep,
            "-f", ds,
            "-o", str(out_dir) + os.sep,
            "-a", ALG,
            "-m", MULTIPLIERS,
            "-t", str(GLOBAL_TILIM),   # passato anche internamente per i solver
        ]
        run(cmd, timeout=GLOBAL_TILIM)

if __name__ == "__main__":
    main()
