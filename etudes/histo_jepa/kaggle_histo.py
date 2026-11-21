"""
ÉTUDE histo_jepa sur KAGGLE, en sessions de fond (≤ 12 h) qui se relaient : kernels histo-a / histo-b, comme le bébé (kaggle_train.py).
Variables écrites en tête par kaggle_histo_push.sh : ETAPE (chrono, tout…), HEURES, BRANCHE, EXTRA (options en plus, p. ex. « --mini 2000 »).
Reprend le dossier histo_runs trouvé dans ses entrées (sortie de l'autre kernel) ; s'il n'y en a pas : départ de zéro.
Sorties : /kaggle/working/histo_runs (pre.pt, res/, pred/, resultats.md, courbe.png, submission.csv, journal.log). Rien n'est envoyé à la compétition.
"""
import glob, os, shutil, subprocess
ETAPE = os.environ.get("ETAPE", "tout"); HEURES = os.environ.get("HEURES", "11"); BRANCHE = os.environ.get("BRANCHE", "master"); EXTRA = os.environ.get("EXTRA", "")
REPO = "/tmp/jepa-physics-bench"; D = "/kaggle/working/histo_runs"
sh = lambda c: subprocess.run(c, shell=True, check=False)
sh(f"git clone -q -b {BRANCHE} https://github.com/frpatry/jepa-physics-bench {REPO}; git -C {REPO} log --oneline -1")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2")
vieux = [d for p in ("/kaggle/input/*/histo_runs", "/kaggle/input/*/*/histo_runs", "/kaggle/input/*/*/*/histo_runs") for d in glob.glob(p)]
if vieux: shutil.copytree(vieux[0], D, dirs_exist_ok=True); print("REPRISE depuis", vieux[0], sorted(os.listdir(D)), flush=True)
else: os.makedirs(D, exist_ok=True); print("DÉPART DE ZÉRO (aucun dossier histo_runs dans les entrées)", flush=True)
sh(f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u etudes/histo_jepa/histo_jepa.py {ETAPE} --out {D} --heures {HEURES} {EXTRA} 2>&1 | tee -a {D}/journal.log")
sh(f"ls -la {D}; du -sh {D}")
