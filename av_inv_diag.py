"""
DIAG « DEVINER SON GESTE » (instrument, quelques minutes GPU) — pendant le run phase 0, la tête de contingence reste à
R² ≈ 0 alors que le geste est LINÉAIREMENT lisible dans les pixels flous (ridge sur paires de frames : R² 0.91 en 0a).
Sur un instantané : l'info du geste est-elle dans le résumé VISUEL du bébé (encodeur en ligne et cible), et quelle tête
sait la lire ?
  - tête du run (m.inv) telle quelle ;
  - tête NEUVE même architecture, entraînée sur latents GELÉS ;
  - tête DIFFÉRENCE (z_t − z_{t−1} par patch -> MLP -> moyenne) ;
  - ridge exacte sur [z_{t−1}, z_t] aplatis ; ridge sur les PIXELS (repère).
Lecture : tête neuve OK -> l'apprentissage conjoint coince ; tout échoue alors que les pixels réussissent -> le résumé efface
le mouvement fin.

  python av_inv_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_0a_6k.pt --stage 0a
"""
import argparse, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_world0 import gen_world0
from av_phase0 import Baby0, InvHead, to_tok0, to_np, to_torch, view_at, T, H

def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

def ridge_r2(Ztr, ytr, Zte, yte, dev, lam=1.0):
    X, Xt = Ztr.to(dev).float(), Zte.to(dev).float(); mu = X.mean(0); X, Xt = X - mu, Xt - mu
    K = X @ X.T; al = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(K), device=dev), ytr.to(dev) - ytr.mean(0).to(dev))
    return r2((Xt @ (X.T @ al) + ytr.mean(0).to(dev)).cpu(), yte)

def fit(head, z0, z1, y, z0t, z1t, dev, steps=3000, bs=256):
    opt = torch.optim.AdamW(head.parameters(), 3e-4, weight_decay=0.01)
    for _ in range(steps):
        bi = torch.randint(0, len(y), (bs,))
        l = F.smooth_l1_loss(head(z0[bi].to(dev).float(), z1[bi].to(dev).float()), y[bi].to(dev)); opt.zero_grad(); l.backward(); opt.step()
    head.eval()
    with torch.no_grad(): return torch.cat([head(z0t[i:i + 512].to(dev).float(), z1t[i:i + 512].to(dev).float()).cpu() for i in range(0, len(z0t), 512)])

class DiffHead(nn.Module):
    def __init__(s, d, npf, h=256):
        super().__init__(); s.pos = nn.Parameter(torch.zeros(1, npf, 2 * d)); s.mlp = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.out = nn.Linear(h, 2)
    def forward(s, z0, z1): return s.out(s.mlp(torch.cat([z1 - z0, z1], -1) + s.pos).mean(1))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--stage", default="0a"); p.add_argument("--n", type=int, default=1500)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    P = cfg["P"]; nP = H // P; npf = nP * nP; nv = T * npf
    m = Baby0(cfg["din"], nv, npf, cfg["d"], cfg["nl"], cfg["nh"], cfg["pred_layers"], bool(cfg["sep"]), cfg.get("inv_head", "attn")).to(dev); m.load_state_dict(ck["m"]); m.eval()
    tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    w = to_np([gen_world0(a.n, a.stage, T, H, seed=4242)]); b = to_torch(w); v = view_at(a.stage, 1.0)
    print(f"instantané pas {ck['state']['it']} | monde {a.stage} | vue σ={v['sigma']:.1f} gris {v['gray']:.1f}", flush=True)
    feats = {"en ligne": [], "cible": []}
    with torch.no_grad():
        for i in range(0, a.n, 50):
            tok, _ = to_tok0({k: x[i:i + 50] for k, x in b.items()}, st, v, P, dev); ix = torch.arange(nv, device=dev).expand(len(tok), -1)
            feats["en ligne"].append(m.enc(tok[:, :nv], ix).view(len(tok), T, npf, -1).half().cpu())
            feats["cible"].append(tgt(tok[:, :nv], ix).view(len(tok), T, npf, -1).half().cpu())
    y = torch.from_numpy(w["CMD"][:, 1:] / 0.05).reshape(-1, 2).float(); k = int(0.8 * len(y))
    Xp = torch.from_numpy(w["X"]).float() / 255; pix = torch.cat([Xp[:, :-1].flatten(2), Xp[:, 1:].flatten(2)], -1).flatten(0, 1)
    sub = torch.randperm(k)[:4000]; te = torch.arange(k, len(y))
    print(f"  PIXELS nets (repère) ridge : R² {ridge_r2(pix[sub], y[sub], pix[te], y[te], dev):+.2f}", flush=True)
    for nm, Fz in feats.items():
        Z = torch.cat(Fz); z0, z1 = Z[:, :-1].flatten(0, 1), Z[:, 1:].flatten(0, 1)
        res = {}
        if nm == "en ligne":
            with torch.no_grad(): pr = torch.cat([m.inv(z0[i:i + 512].to(dev).float(), z1[i:i + 512].to(dev).float()).cpu() for i in range(k, len(y), 512)])
            res["tête du run"] = r2(pr, y[k:])
        zz = torch.cat([z0.flatten(1), z1.flatten(1)], -1); res["ridge"] = ridge_r2(zz[sub], y[sub], zz[te], y[te], dev)
        torch.manual_seed(0); res["tête neuve"] = r2(fit(InvHead(Z.size(-1), npf).to(dev), z0[:k], z1[:k], y[:k], z0[k:], z1[k:], dev), y[k:])
        torch.manual_seed(0); res["tête différence"] = r2(fit(DiffHead(Z.size(-1), npf).to(dev), z0[:k], z1[:k], y[:k], z0[k:], z1[k:], dev), y[k:])
        print(f"  encodeur {nm:>8s} : " + " | ".join(f"{k_} R² {v_:+.2f}" for k_, v_ in res.items()), flush=True)

if __name__ == "__main__":
    main()
