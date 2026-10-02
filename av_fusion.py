"""
AV + V-JEPA 2 — ÉTAPE 1 : JEPA de FUSION vision (V-JEPA 2 gelé = cortex visuel) + OUÏE.

Étape 0 (av_vjepa2.py) : les latents V-JEPA 2 gelés voient nos disques (positions R² 0.92, chocs 89 %,
ratio de masses 0.41). Ici on entraîne, en JEPA pur (aucune étiquette), un transformer de FUSION
au-dessus : tokens visuels = features V-JEPA 2 (8 pas × 4×4, 1024 d, gelées), tokens audio = le son
des 2 frames de chaque pas (2 × a_sub × 2 canaux × 32 bandes). Même recette que av_jepa.py
(AVJEPA réutilisé tel quel) : masques tube / bloc / futur / son-depuis-image / bloc audio, dropout
de modalité, prédicteur attentionnel, SIGReg.

La question : le modèle apprend-il SEUL quel son appartient à quel disque ? (LIAGE)
  matériau par disque : 67 % = plafond « ensemble connu, attribution inconnue » ; >67 % = liage.
  Référence monde (vision parfaite + son, lecteur supervisé) : 99 %.
Plus : log-masse, ratio de masses (choc), impact. Conditions : init (fusion non entraînée) vs
JEPA entraîné ; entrée vision seule / vision+son / son seul.

  python av_fusion.py --n_train 6000 --steps 6000     # Colab : réutilise /content/av_vj2_feats.pt (sonde)
"""
import argparse, os, time
import numpy as np, torch
from av_jepa import gen_world, pretrain, evaluate, NB
from av_vjepa2 import encode_vjepa2

def build_tokens(Z, w, a_sub, amu=None, asd=None):
    """Z:(n,Tt,k,1024) V-JEPA 2 ; audio des frames 2t,2t+1 -> (n, Tt*k + Tt, 1024) fp16 (audio zéro-paddé)."""
    n, Tt, k, d = Z.shape
    A = w["A"].reshape(n, Tt, -1).astype(np.float32)                             # (n, Tt, 2*a_sub*2*NB)
    if amu is None: amu, asd = A.mean((0, 1)), A.std((0, 1)) + 1e-4
    A = (A - amu) / asd
    pad = np.zeros((n, Tt, d), np.float16); pad[..., :A.shape[-1]] = A
    tok = torch.cat([Z.reshape(n, Tt * k, d).half(), torch.from_numpy(pad)], 1)
    return tok, amu, asd

def per_step_labels(w, Tt):
    """impact par pas temporel V-JEPA 2 (= frames 2t, 2t+1)."""
    w = dict(w); w["IMP"] = w["IMP"][:, 0::2] | w["IMP"][:, 1::2]; return w

def feats(path, n, seed, a, dev):
    if os.path.exists(path):
        Z = torch.load(path)["Z"]
        if len(Z) >= n: print(f"  cache {path} : {tuple(Z.shape)}", flush=True); return Z[:n]
    w = gen_world(n, a.T, 32, seed=seed, a_sub=a.a_sub)
    Z = encode_vjepa2(w["X"], a, dev); torch.save(dict(Z=Z, args=vars(a)), path); print(f"  sauvé -> {path}")
    return Z

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=6000); p.add_argument("--n_probe", type=int, default=3000)
    p.add_argument("--T", type=int, default=16); p.add_argument("--a_sub", type=int, default=2)
    p.add_argument("--train_cache", type=str, default="/content/av_vj2_train.pt")
    p.add_argument("--probe_cache", type=str, default="/content/av_vj2_feats.pt")
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--pool", type=int, default=4); p.add_argument("--enc_bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=6000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--rw", type=float, default=1.0)
    p.add_argument("--n_masks", type=int, default=3); p.add_argument("--sig_on", type=str, default="token")
    p.add_argument("--d", type=int, default=256); p.add_argument("--nl", type=int, default=4)
    p.add_argument("--nh", type=int, default=8); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--probe_steps", type=int, default=2000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", type=str, default="va")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    rng = np.random.default_rng(a.seed)
    print("--- features V-JEPA 2 (gelé)", flush=True)
    Ztr = feats(a.train_cache, a.n_train, a.seed, a, dev)                        # monde d'entraînement : graine 0
    Zpr = feats(a.probe_cache, a.n_probe, 1000, a, dev)                          # jeu de sonde : graine 1000 (étape 0)
    wtr = gen_world(a.n_train, a.T, 32, seed=a.seed, a_sub=a.a_sub)
    wpr = per_step_labels(gen_world(a.n_probe, a.T, 32, seed=1000, a_sub=a.a_sub), Ztr.shape[1])
    tok, amu, asd = build_tokens(Ztr, wtr, a.a_sub); tokp, _, _ = build_tokens(Zpr, wpr, a.a_sub, amu, asd)
    Tt, k = Ztr.shape[1], Ztr.shape[2]; nv = Tt * k; nP = int(round(k ** 0.5))
    del Ztr, Zpr, wtr; wpr.pop("X")                                              # RAM Colab (12 Go)
    a.da = 2 * a.a_sub * 2 * NB; a.T_tok = Tt
    print(f"tokens : entraînement {tuple(tok.shape)}, sonde {tuple(tokp.shape)} | {Tt} pas × {k} visuels + {Tt} audio "
          f"| choc disque-disque {wpr['HIT'].mean():.0%} ({time.time() - t0:.0f}s)", flush=True)
    ntr = int(0.75 * a.n_probe); rows = []
    def run_evals(m, name, conds):
        for c in conds:
            r = evaluate(m, tokp, wpr, c, dev, a.probe_steps, ntr); rows.append((name, c, r))
            print(f"  {name:>6s} | entrée {c:>2s} | matériau {r['mat']:.0%} | log-masse R² {r['lm_all']:+.2f} "
                  f"| ratio masses (choc) R² {r['ratio_hit']:+.2f} | impact {r['imp_bacc']:.0%}", flush=True)
    print("--- fusion NON entraînée (ce que la sonde tire de V-JEPA 2 + son bruts)", flush=True)
    steps = a.steps; a.steps = 0
    run_evals(pretrain("va", tok, a, dev, nv, Tt, nP, rng), "init", ["v", "va", "a"]); a.steps = steps
    for arm in a.arms.split(","):
        print(f"--- JEPA de fusion bras {arm.upper()} ({a.steps} pas, auto-supervisé)", flush=True)
        m = pretrain(arm, tok, a, dev, nv, Tt, nP, rng)
        run_evals(m, arm.upper(), ["v"] if arm == "v" else ["v", "va", "a"])
    print("\n================ RÉSUMÉ (sondes gelées, jeu tenu à l'écart) ================")
    print(f"{'modèle':>6s} {'entrée':>6s} | {'matériau':>8s} | {'lmasse':>6s} | {'ratio choc':>10s} | {'impact':>6s}")
    for name, c, r in rows:
        print(f"{name:>6s} {c:>6s} | {r['mat']:8.0%} | {r['lm_all']:+6.2f} | {r['ratio_hit']:+10.2f} | {r['imp_bacc']:6.0%}")
    get = {(n, c): r for n, c, r in rows}
    if ("VA", "va") in get:
        i, v = get[("init", "va")], get[("VA", "va")]
        print(f"\nLIAGE son<->disque (matériau par disque ; 67 % = pas de liage, monde parfait 99 %) : "
              f"init {i['mat']:.0%} -> JEPA {v['mat']:.0%}")
        print(f"MASSE : log-masse {i['lm_all']:+.2f} -> {v['lm_all']:+.2f} ; ratio (choc) {i['ratio_hit']:+.2f} -> {v['ratio_hit']:+.2f}")
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
