"""
BÉBÉ v3c sur KAGGLE (16 nov. 2026 : plus d'unités Colab) — reprise de bebe3c.pt (70k sur Drive) jusqu'à 110k, par sessions de fond de ≤ 12 h sur T4.
Script autonome poussé par kaggle_bebe3c_push.sh (deux kernels a / b qui se relaient : chacun reprend la sauvegarde la plus avancée trouvée dans ses entrées —
le Dataset privé de départ ou la sortie de l'autre kernel). Sorties : /kaggle/working/jepa_runs (bebe3c.pt, instantanés, run_bebe3c.log).
Réglages par variables d'environnement écrites en tête par le script de poussée : HEURES, TOTAL, EXTRA.
"""
import glob, os, shutil, subprocess, torch
HEURES = os.environ.get("HEURES", "11"); TOTAL = os.environ.get("TOTAL", "110000"); EXTRA = os.environ.get("EXTRA", "")
REPO = "/tmp/jepa-physics-bench"; D = "/kaggle/working/jepa_runs"; os.makedirs(D, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, check=False)
sh(f"git clone -q https://github.com/frpatry/jepa-physics-bench {REPO}; git -C {REPO} log --oneline -1")
sh("pip -q install git+https://github.com/rbalestr-lab/lejepa && echo LEJEPA OK; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
best, it0 = None, -1                                                   # la sauvegarde la plus avancée parmi les entrées (lecture seule -> copie de travail)
for f in glob.glob("/kaggle/input/**/bebe3c.pt", recursive=True):
    it = int(torch.load(f, map_location="cpu", weights_only=False)["state"]["it"]); print(f"entrée : {f} (pas {it})", flush=True)
    if it > it0: best, it0 = f, it
assert best is not None, "aucun bebe3c.pt dans les entrées"
ck = f"{D}/bebe3c.pt"; shutil.copy(best, ck); print(f"REPRISE au pas {it0} depuis {best}", flush=True)
for f in glob.glob("/kaggle/input/**/run_bebe3c.log", recursive=True):   # le journal suit la sauvegarde
    if os.path.dirname(f) == os.path.dirname(best): shutil.copy(f, f"{D}/run_bebe3c.log")
sh(f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout {HEURES}h bash run_bebe3.sh {ck} {TOTAL} --motion_w 0 --roll_w 0 --solo_share 0 "
   f"--accum 4 --ckpt_every 500 {EXTRA} 2>&1 | tee -a {D}/run_bebe3c.log | grep --line-buffered -E '^ +pas +[0-9]*(000|500) |OBJET|objet|critère|Error|error|Traceback'")   # journal complet dans le fichier ; l'essentiel en direct
sh(f"tail -c 3000 {D}/run_bebe3c.log; ls -la {D}")
print("FIN pas", int(torch.load(ck, map_location="cpu", weights_only=False)["state"]["it"]), flush=True)
