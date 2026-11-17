"""
UNE COMMANDE du dépôt sur KAGGLE avec un bébé gelé (diagnostics courts ; 17 nov. 2026). Entrée : la sortie d'un kernel d'entraînement (bebe3c.pt).
CMD = commande à lancer depuis le dépôt, avec {ck} = la sauvegarde (CKPT_PAS exact, sinon la plus avancée) et {D} = dossier de sortie (/kaggle/working/out).
Poussé par kaggle_cmd_push.sh.
"""
import glob, os, subprocess, torch
PAS = int(os.environ.get("CKPT_PAS", "0")); CMD = os.environ["CMD"]; HEURES = os.environ.get("HEURES", "6")
REPO = "/tmp/jepa-physics-bench"; D = "/kaggle/working/out"; os.makedirs(D, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, check=False)
sh(f"git clone -q https://github.com/frpatry/jepa-physics-bench {REPO}; git -C {REPO} log --oneline -1")
sh("pip -q install git+https://github.com/rbalestr-lab/lejepa && echo LEJEPA OK; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
C = {f: int(torch.load(f, map_location="cpu", weights_only=False)["state"]["it"]) for f in glob.glob("/kaggle/input/**/bebe3c.pt", recursive=True)}
ck = next((f for f, i in C.items() if i == PAS), None) or max(C, key=C.get); print(f"BÉBÉ GELÉ : {ck} (pas {C[ck]})", flush=True)
sh(f"cd {REPO} && timeout {HEURES}h {CMD.format(ck=ck, D=D)} 2>&1 | tee {D}/run.log"); sh(f"ls -la {D}")
