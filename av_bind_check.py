"""
INSTRUMENT (pas une méthode) — le LIAGE son<->disque est-il CALCULABLE à partir des entrées gelées
V-JEPA 2 + son, telles que tokenisées pour la fusion ?

Un lecteur SUPERVISÉ expressif (transformer 2 couches, attention entre tous les tokens = peut faire
l'appariement relationnel « son à t + pano ↔ disque au mur à t ») est entraîné sur les étiquettes du
jeu d'entraînement et testé sur le jeu de sonde.
  matériau ≫ 67 % -> le liage est calculable : le blocage vient de la sonde (1 requête + linéaire)
                     ou du JEPA qui ne range pas le liage dans un token ;
  matériau ≈ 67 % -> la tokenisation l'empêche (pas V-JEPA 2 = 2 frames, résumé 4×4, son agrégé...).
Référence monde : état parfait + son -> 99 % (av_world_check.py).

  python av_bind_check.py           # Colab : réutilise /content/av_vj2_train.pt et av_vj2_feats.pt
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import gen_world, NB
from av_fusion import build_tokens, feats

class Reader(nn.Module):
    def __init__(s, dv, da, nv, ntok, d=256, nl=2, nh=8, p=0.1):
        super().__init__()
        s.nv, s.da = nv, da
        s.ev, s.ea = nn.Linear(dv, d), nn.Linear(da, d); s.pos = nn.Parameter(torch.zeros(1, ntok, d))
        s.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, nh, 2 * d, batch_first=True, dropout=p, activation="gelu")
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d); s.mat = nn.Linear(d, 6); s.lm = nn.Linear(d, 3)
    def forward(s, tok, keep):
        idx = torch.where(keep)[0]
        x = tok[:, idx]; isa = (idx >= s.nv).view(1, -1, 1)
        e = torch.where(isa, s.ea(x[..., :s.da]), s.ev(x)) + s.pos[:, idx]
        h = s.ln(s.tr(torch.cat([s.cls.expand(len(x), -1, -1), e], 1))[:, 0])
        return s.mat(h), s.lm(h)

def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-8))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=6000); p.add_argument("--n_probe", type=int, default=3000)
    p.add_argument("--T", type=int, default=16); p.add_argument("--a_sub", type=int, default=2)
    p.add_argument("--train_cache", type=str, default="/content/av_vj2_train.pt")
    p.add_argument("--probe_cache", type=str, default="/content/av_vj2_feats.pt")
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--pool", type=int, default=4); p.add_argument("--enc_bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--conds", type=str, default="va,a,v")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    Ztr = feats(a.train_cache, a.n_train, 0, a, dev); Zpr = feats(a.probe_cache, a.n_probe, 1000, a, dev)
    wtr = gen_world(a.n_train, a.T, 32, seed=0, a_sub=a.a_sub); wpr = gen_world(a.n_probe, a.T, 32, seed=1000, a_sub=a.a_sub)
    tok, st = build_tokens(Ztr, wtr, a.a_sub); tokp, _ = build_tokens(Zpr, wpr, a.a_sub, st)
    Tt, k = Ztr.shape[1], Ztr.shape[2]; nv = Tt * k; N = nv + Tt; da = 2 * a.a_sub * 2 * NB
    del Ztr, Zpr
    def lab(w):
        lm = torch.tensor(w["LM"]); return torch.tensor(w["MAT"]).long(), torch.stack([lm[:, 0], lm[:, 1], lm[:, 0] - lm[:, 1]], 1)
    mtr, ytr = lab(wtr); mte, yte = lab(wpr); ym, ys = ytr.mean(0), ytr.std(0); hit = torch.tensor(wpr["HIT"])
    print(f"tokens {tuple(tok.shape)} / {tuple(tokp.shape)} ({time.time() - t0:.0f}s)", flush=True)
    for cond in a.conds.split(","):
        keep = torch.zeros(N, dtype=torch.bool)
        if "v" in cond: keep[:nv] = True
        if "a" in cond: keep[nv:] = True
        keep = keep.to(dev); torch.manual_seed(0)
        m = Reader(tok.size(-1), da, nv, N).to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
        for it in range(1, a.steps + 1):
            bi = torch.randint(0, len(tok), (a.bs,)); pm, pl = m(tok[bi].to(dev).float(), keep)
            loss = F.cross_entropy(pm.view(-1, 3), mtr[bi].to(dev).view(-1)) * (1 if "a" in cond else 0) \
                + F.mse_loss(pl, ((ytr[bi] - ym) / ys).to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if it % (a.steps // 4) == 0: print(f"  [{cond}] step {it}  loss {loss.item():.3f}", flush=True)
        m.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(tokp), 256): outs.append([o.cpu() for o in m(tokp[i:i + 256].to(dev).float(), keep)])
        pm = torch.cat([o[0] for o in outs]); pl = torch.cat([o[1] for o in outs]) * ys + ym
        mat = float((pm.view(-1, 2, 3).argmax(-1) == mte).float().mean())
        lm = (r2(pl[:, 0], yte[:, 0]) + r2(pl[:, 1], yte[:, 1])) / 2; ratio = r2(pl[hit, 2], yte[hit, 2])
        print(f"LECTEUR SUPERVISÉ | entrée {cond:>2s} | matériau {mat:.0%} (67 % = pas de liage) | log-masse R² {lm:+.2f} "
              f"| ratio (choc) R² {ratio:+.2f}  ({time.time() - t0:.0f}s)", flush=True)
    print("Repères : monde parfait (état+son) 99 % / masse .96 / ratio .95 ; sondes 1-requête sur entrées brutes 66 % / .32 / −.15")

if __name__ == "__main__":
    main()
