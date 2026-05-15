# KineticsForge V2 Kaggle one-cell run
# Attach `kineticsforge_v2_input.zip` as a Kaggle Dataset input, then run this cell.
import os, sys, json, shutil, subprocess, zipfile, pathlib, textwrap

ROOT = pathlib.Path("/kaggle/working/kineticsforge_v2")
INPUTS = list(pathlib.Path("/kaggle/input").rglob("kineticsforge_v2_input.zip"))
if not INPUTS:
    raise FileNotFoundError("Attach the KineticsForge V2 input zip as a Kaggle Dataset.")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(INPUTS[0], "r") as zf:
    zf.extractall(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

def run(cmd):
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

run([sys.executable, "-m", "pip", "install", "-q", "-r", "training/colab_kaggle/requirements_kaggle.txt"])
run([sys.executable, "-m", "validation.v2_readiness_report", "--project-root", ".", "--out", "data/cache/v2_readiness_report.json"])
run([sys.executable, "-m", "training.colab_kaggle.industrial_training_pipeline", "--project-root", ".", "--task", "all", "--profile", "smoke"])
run([sys.executable, "-m", "training.colab_kaggle.kaggle_colab_data_bootstrap", "--profile", "smoke"])
run([sys.executable, "-m", "training.colab_kaggle.kaggle_colab_train", "--task", "all", "--profile", "quick", "--device", "auto"])

print("\nArtifacts:")
for p in [
    "data/cache/v2_readiness_report.json",
    "training/colab_kaggle/runs",
    "checkpoints/metrics.json",
]:
    print(" -", ROOT / p)
