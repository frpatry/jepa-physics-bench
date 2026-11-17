"""
PUSH-T sur KAGGLE avec un bébé gelé (17 nov. 2026) — même protocole que le test du bébé v2 (10 épisodes × 300 gestes, il planifie dans SON imagination par objet,
coût sur les cases « chose », pas de proposition ni de juge). Entrée : la sortie d'un kernel d'entraînement (bebe3c.pt le plus avancé de ses entrées, ou CKPT_PAS exact).
Sorties : /kaggle/working/pusht (run_pusht.log, GIF).   Poussé par kaggle_pusht_push.sh.
"""
import glob, os, subprocess, torch
PAS = int(os.environ.get("CKPT_PAS", "0")); EXTRA = os.environ.get("EXTRA", ""); HEURES = os.environ.get("HEURES", "6")
REPO = "/tmp/jepa-physics-bench"; D = "/kaggle/working/pusht"; os.makedirs(D, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, check=False)
sh(f"git clone -q https://github.com/frpatry/jepa-physics-bench {REPO}; git -C {REPO} log --oneline -1")
sh("pip -q install git+https://github.com/rbalestr-lab/lejepa && echo LEJEPA OK; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
C = {f: int(torch.load(f, map_location="cpu", weights_only=False)["state"]["it"]) for f in glob.glob("/kaggle/input/**/bebe3c.pt", recursive=True)}
print("entrées :", C, flush=True); ck = next((f for f, i in C.items() if i == PAS), None) or max(C, key=C.get); print(f"BÉBÉ GELÉ : {ck} (pas {C[ck]})", flush=True)
sh(f"cd {REPO} && timeout {HEURES}h python -u av_plan0.py --ckpt {ck} --objimag self --thing_cost 1 --task pusht --goal_img 1 --slide 1 --propose 0 --diag 0 "
   f"--episodes 10 --plan_steps 300 --plan_h 8 --pols= --reach_diag 1 --gif 6 --gif_frames 300 --film_dir {D} --gif_tag b3c {EXTRA} 2>&1 | tee {D}/run_pusht.log")
sh(f"ls -la {D}")
