"""
ÉVALUATION ÉQUITABLE — le JEPA de fusion (cibles gelées, auto-supervisé) rend-il le LIAGE son<->disque
et la MASSE plus faciles à apprendre ? (efficacité-étiquettes : le vrai critère d'un apprentissage
auto-supervisé ; « l'enfant apprend-il plus vite grâce à ce qu'il a vécu ? »)

av_bind_check.py a montré que l'info est CALCULABLE depuis V-JEPA 2 + son (lecteur transformer : matériau
78 %, masse 0.59) mais que nos sondes 1-requête étaient trop faibles pour la lire. Ici, MÊME lecteur
transformer (2 couches), entraîné avec 300 / 1000 / 6000 étiquettes, sur 3 représentations :
  raw  : entrées gelées (V-JEPA 2 + spectres)
  init : fusion NON entraînée (contrôle : une transformation aléatoire n'aide pas)
  jepa : fusion entraînée en JEPA à cibles gelées (aucune étiquette)
Si le JEPA a appris le liage seul, son lecteur doit gagner surtout à PEU d'étiquettes.

  python av_fair_eval.py        # Colab : réutilise les caches V-JEPA 2 (sinon les recalcule)
"""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from av_jepa import gen_world, NB
from av_fusion import build_tokens, feats, pretrain_frozen, FrozenTargetJEPA
from av_bind_check import Reader, r2

@torch.no_grad()
def represent(m, tok, dev, bs=256):
    idx = torch.arange(tok.size(1), device=dev)
    return torch.cat([m.enc(tok[i:i + bs].to(dev).float(), idx.expand(len(tok[i:i + bs]), -1)).half().cpu()
                      for i in range(0, len(tok), bs)])

def read(Rtr, mtr, ytr, Rte, mte, yte, hit, nv, da, a, dev):
    """lecteur transformer supervisé (entrée image+son) -> (matériau, log-masse R², ratio choc R²)."""
    N = Rtr.size(1); keep = torch.ones(N, dtype=torch.bool, device=dev)
    ym, ys = ytr.mean(0), ytr.std(0); torch.manual_seed(0)
    m = Reader(Rtr.size(-1), da, nv, N).to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
    for it in range(a.read_steps):
        bi = torch.randint(0, len(Rtr), (min(a.bs, len(Rtr)),)); pm, pl = m(Rtr[bi].to(dev).float(), keep)
        loss = F.cross_entropy(pm.view(-1, 3), mtr[bi].to(dev).view(-1)) + F.mse_loss(pl, ((ytr[bi] - ym) / ys).to(dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(Rte), 256): outs.append([o.cpu() for o in m(Rte[i:i + 256].to(dev).float(), keep)])
    pm = torch.cat([o[0] for o in outs]); pl = torch.cat([o[1] for o in outs]) * ys + ym
    mat = float((pm.view(-1, 2, 3).argmax(-1) == mte).float().mean())
    return mat, (r2(pl[:, 0], yte[:, 0]) + r2(pl[:, 1], yte[:, 1])) / 2, r2(pl[hit, 2], yte[hit, 2])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=6000); p.add_argument("--n_probe", type=int, default=3000)
    p.add_argument("--T", type=int, default=16); p.add_argument("--a_sub", type=int, default=2)
    p.add_argument("--train_cache", type=str, default="/content/av_vj2_train.pt")
    p.add_argument("--probe_cache", type=str, default="/content/av_vj2_feats.pt")
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--pool", type=int, default=4); p.add_argument("--enc_bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=6000, help="pas du JEPA (auto-supervisé)")
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_masks", type=int, default=3)
    p.add_argument("--d", type=int, default=256); p.add_argument("--nl", type=int, default=4)
    p.add_argument("--nh", type=int, default=8); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--read_steps", type=int, default=3000); p.add_argument("--budgets", type=str, default="300,1000,6000")
    p.add_argument("--reps", type=str, default="raw,init,jepa"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    rng = np.random.default_rng(a.seed)
    Ztr = feats(a.train_cache, a.n_train, 0, a, dev); Zpr = feats(a.probe_cache, a.n_probe, 1000, a, dev)
    wtr = gen_world(a.n_train, a.T, 32, seed=0, a_sub=a.a_sub); wpr = gen_world(a.n_probe, a.T, 32, seed=1000, a_sub=a.a_sub)
    tok, st = build_tokens(Ztr, wtr, a.a_sub); tokp, _ = build_tokens(Zpr, wpr, a.a_sub, st)
    Tt, k = Ztr.shape[1], Ztr.shape[2]; nv = Tt * k; nP = int(round(k ** 0.5)); a.da = 2 * a.a_sub * 2 * NB
    del Ztr, Zpr
    def lab(w):
        lm = torch.tensor(w["LM"]); return torch.tensor(w["MAT"]).long(), torch.stack([lm[:, 0], lm[:, 1], lm[:, 0] - lm[:, 1]], 1)
    mtr, ytr = lab(wtr); mte, yte = lab(wpr); hit = torch.tensor(wpr["HIT"])
    print(f"tokens {tuple(tok.shape)} / {tuple(tokp.shape)} ({time.time() - t0:.0f}s)", flush=True)
    R = {}
    if "raw" in a.reps: R["raw"] = (tok, tokp, a.da)
    if "init" in a.reps:
        torch.manual_seed(a.seed); m0 = FrozenTargetJEPA(tok.size(-1), a.da, nv, Tt, a.d, a.nl, a.nh, a.pred_layers).to(dev).eval()
        R["init"] = (represent(m0, tok, dev), represent(m0, tokp, dev), a.d); del m0
    if "jepa" in a.reps:
        print(f"--- JEPA de fusion à cibles gelées ({a.steps} pas, AUCUNE étiquette)", flush=True)
        m = pretrain_frozen("va", tok, a, dev, nv, Tt, nP, rng)
        R["jepa"] = (represent(m, tok, dev), represent(m, tokp, dev), a.d); del m
    res = {}
    for L in [int(x) for x in a.budgets.split(",")]:
        for name, (Rtr, Rte, da) in R.items():
            r = read(Rtr[:L], mtr[:L], ytr[:L], Rte, mte, yte, hit, nv, da, a, dev); res[(name, L)] = r
            print(f"  {L:5d} étiquettes | {name:>4s} | matériau {r[0]:.0%} | log-masse R² {r[1]:+.2f} | ratio (choc) R² {r[2]:+.2f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    print("\n=========== EFFICACITÉ-ÉTIQUETTES (lecteur transformer, image+son, jeu tenu à l'écart) ===========")
    print("matériau : 67 % = pas de liage ; monde parfait 99 %")
    for L in [int(x) for x in a.budgets.split(",")]:
        print(f"  {L:5d} étiq. | " + " | ".join(f"{n} mat {res[(n, L)][0]:.0%} masse {res[(n, L)][1]:+.2f} ratio {res[(n, L)][2]:+.2f}"
                                             for n in R))
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
