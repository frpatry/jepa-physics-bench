"""
PISTE A — test d'âge « le son vient-il de là où ça bouge ? » (LOCALISATION) avec une BONNE vision.

av_test_check.py : avec la vision parfaite + son, localisation 96 % (1000 étiq.) ; nos encodeurs pixel
échouaient faute de voir où sont les disques. Ici la vision = V-JEPA 2 gelé (positions R² 0.92) et la
question : le JEPA de FUSION (auto-supervisé, prédit le son depuis l'image et inversement) apprend-il SEUL
le lien stéréo <-> position, au point de rendre la localisation facile avec PEU d'étiquettes ?

Représentations : raw (V-JEPA 2 + spectres), init (fusion aléatoire), jepa (fusion à cibles gelées).
Tests (lecteur transformer gelé, jeu de sonde tenu à l'écart) :
  LOCALISATION : son correct vs stéréo inversée — 100 / 300 / 1000 étiquettes
  ANTICIPATION : pas V-JEPA 2 0..3 (frames 0-7) seulement -> choc aux frames 8-9 ? — 2000 étiquettes

  python av_loc_vj2.py        # Colab : caches /content/av_vj2_train.pt + av_vj2_feats.pt (sinon recalculés)
"""
import argparse, time
import numpy as np, torch
from av_jepa import gen_world, NB
from av_fusion import build_tokens, feats, pretrain_frozen, FrozenTargetJEPA
from av_dev import fit_reader, represent, bacc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=6000); p.add_argument("--n_probe", type=int, default=3000)
    p.add_argument("--T", type=int, default=16); p.add_argument("--a_sub", type=int, default=2)
    p.add_argument("--train_cache", type=str, default="/content/av_vj2_train.pt")
    p.add_argument("--probe_cache", type=str, default="/content/av_vj2_feats.pt")
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--pool", type=int, default=4); p.add_argument("--enc_bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=6000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--n_masks", type=int, default=3)
    p.add_argument("--d", type=int, default=256); p.add_argument("--nl", type=int, default=4)
    p.add_argument("--nh", type=int, default=8); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--read_steps", type=int, default=2000); p.add_argument("--loc_budgets", type=str, default="100,300,1000")
    p.add_argument("--antic_labels", type=int, default=2000); p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--reps", type=str, default="raw,init,jepa"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time(); rng = np.random.default_rng(a.seed)
    Ztr = feats(a.train_cache, a.n_train, 0, a, dev); Zpr = feats(a.probe_cache, a.n_probe, 1000, a, dev)
    wtr = gen_world(a.n_train, a.T, 32, seed=0, a_sub=a.a_sub); wpr = gen_world(a.n_probe, a.T, 32, seed=1000, a_sub=a.a_sub)
    wtr.pop("X"); Xless = {k: v for k, v in wpr.items() if k != "X"}
    tok, st = build_tokens(Ztr, wtr, a.a_sub); tokp, _ = build_tokens(Zpr, Xless, a.a_sub, st)
    lab = (np.arange(a.n_probe) % 2).astype(np.int64)                  # 1 = stéréo inversée
    Asw = Xless["A"].copy(); Asw[lab == 1] = Asw[lab == 1][:, :, [1, 0, 3, 2]]
    toksw, _ = build_tokens(Zpr, dict(Xless, A=Asw), a.a_sub, st)
    Tt, k = Ztr.shape[1], Ztr.shape[2]; nv = Tt * k; nP = int(round(k ** 0.5)); a.da = 2 * a.a_sub * 2 * NB; del Ztr, Zpr
    N = nv + Tt; frame = np.concatenate([np.arange(nv) // k, np.arange(Tt)])
    y_loc = torch.from_numpy(lab); y_ant = torch.from_numpy((wpr["IMP"][:, 8] | wpr["IMP"][:, 9]).astype(np.int64))
    print(f"tokens {tuple(tok.shape)} | anticipation positifs {y_ant.float().mean():.0%} ({time.time() - t0:.0f}s)", flush=True)
    models = {}
    if "raw" in a.reps: models["raw"] = None
    if "init" in a.reps:
        torch.manual_seed(a.seed); models["init"] = FrozenTargetJEPA(tok.size(-1), a.da, nv, Tt, a.d, a.nl, a.nh, a.pred_layers).to(dev).eval()
    if "jepa" in a.reps:
        print(f"--- JEPA de fusion à cibles gelées ({a.steps} pas, AUCUNE étiquette)", flush=True)
        models["jepa"] = pretrain_frozen("va", tok, a, dev, nv, Tt, nP, rng)
    te = slice(a.n_probe - a.n_test, a.n_probe); rows = {}
    for name, m in models.items():
        R = represent(m, toksw, np.ones(N, bool), dev); r = {}
        for L in [int(x) for x in a.loc_budgets.split(",")]:
            pr = fit_reader(R[:L], y_loc[:L], R[te], "bin", dev, a.read_steps).argmax(-1); r[f"loc {L}"] = bacc(pr, y_loc[te])
        R = represent(m, tokp, frame <= 3, dev)                          # anticipation : seulement les pas 0..3 (aucune fuite)
        pr = fit_reader(R[:a.antic_labels], y_ant[:a.antic_labels], R[te], "bin", dev, a.read_steps).argmax(-1)
        r[f"antic {a.antic_labels}"] = bacc(pr, y_ant[te]); rows[name] = r
        print(f"  {name:>5s} | " + " | ".join(f"{k_} {v:.0%}" for k_, v in r.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    keys = list(next(iter(rows.values())).keys())
    print("\n===== LOCALISATION / ANTICIPATION avec vision V-JEPA 2 (lecteur gelé ; hasard 50 % ; plafond vision parfaite : loc 96 % à 1000) =====")
    print(f"{'repr':>5s} | " + " | ".join(f"{k_:>10s}" for k_ in keys))
    for name, r in rows.items(): print(f"{name:>5s} | " + " | ".join(f"{r[k_]:10.0%}" for k_ in keys))
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
