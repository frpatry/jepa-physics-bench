"""
ENTRAÎNEMENT d'un bébé sur KAGGLE en sessions de fond T4 (≤ 12 h) qui se relaient — version générale de kaggle_bebe3c.py (17 nov. 2026, bébé v4 de zéro).
Variables écrites en tête par kaggle_train_push.sh : NOM (bebe4), RECETTE (run_bebe4.sh), HEURES, TOTAL, EXTRA.
Reprend la sauvegarde <NOM>.pt la plus avancée trouvée dans ses entrées (sortie de l'autre kernel) ; s'il n'y en a pas : départ de ZÉRO.
Sorties : /kaggle/working/jepa_runs (<NOM>.pt, instantanés, run_<NOM>.log) ; l'essentiel du journal est visible en direct.
"""
import glob, os, shutil, subprocess, torch
NOM = os.environ["NOM"]; RECETTE = os.environ["RECETTE"]; HEURES = os.environ.get("HEURES", "11"); TOTAL = os.environ.get("TOTAL", "110000"); EXTRA = os.environ.get("EXTRA", "")
REPO = "/tmp/jepa-physics-bench"; D = "/kaggle/working/jepa_runs"; os.makedirs(D, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, check=False)
sh(f"git clone -q https://github.com/frpatry/jepa-physics-bench {REPO}; git -C {REPO} log --oneline -1")
sh("pip -q install git+https://github.com/rbalestr-lab/lejepa && echo LEJEPA OK; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
best, it0 = None, -1
for f in glob.glob(f"/kaggle/input/**/{NOM}.pt", recursive=True):
    it = int(torch.load(f, map_location="cpu", weights_only=False)["state"]["it"]); print(f"entrée : {f} (pas {it})", flush=True)
    if it > it0: best, it0 = f, it
ck = f"{D}/{NOM}.pt"
if best is None: print("DÉPART DE ZÉRO (aucune sauvegarde dans les entrées)", flush=True)
else:
    shutil.copy(best, ck); print(f"REPRISE au pas {it0} depuis {best}", flush=True)
    for f in glob.glob(os.path.dirname(best) + f"/run_{NOM}.log") + glob.glob(os.path.dirname(best) + f"/{NOM}_*k.pt"): shutil.copy(f, D)   # le journal et les instantanés suivent
sh(f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout {HEURES}h bash {RECETTE} {ck} {TOTAL} --accum 4 --ckpt_every 500 {EXTRA} 2>&1 | tee -a {D}/run_{NOM}.log | "
   "grep --line-buffered -E '^ +pas +[0-9]*(000|500) |CRIT|objet|Error|error|Traceback|précision'")
sh(f"tail -c 2000 {D}/run_{NOM}.log; ls -la {D}")
if os.path.exists(ck): print("FIN pas", int(torch.load(ck, map_location="cpu", weights_only=False)["state"]["it"]), flush=True)
