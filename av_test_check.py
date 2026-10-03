"""
INSTRUMENT — les tests « adaptés à l'âge » de av_dev.py sont-ils SOLUBLES ? (plafond avant conclusion)

Lecteur supervisé (transformer sur les frames) qui reçoit l'ÉTAT PARFAIT (positions + vitesses des disques,
aucun pixel) et le son, avec 1000 / 3000 / 10000 étiquettes :
  LOCALISATION : son correct vs stéréo inversée (gauche/droite)
  ANTICIPATION : frames 0..7 seulement -> choc aux frames 8-9 ?
+ diagnostic stéréo : la différence d'intensité gauche/droite (ILD) d'un choc dit-elle où il a eu lieu ?

  python av_test_check.py
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import gen_world

class Reader(nn.Module):
    def __init__(s, din, T, d=96, nl=3):
        super().__init__()
        s.emb = nn.Linear(din, d); s.pos = nn.Parameter(torch.zeros(1, T, d)); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.1)
        s.tr = nn.TransformerEncoder(layer, nl); s.out = nn.Linear(d, 2)
    def forward(s, x):
        h = s.emb(x) + s.pos[:, :x.size(1)]
        return s.out(s.tr(torch.cat([s.cls.expand(len(x), -1, -1), h], 1))[:, 0])

def feats(w, A, frames, use_state=True, use_audio=True):
    n = len(A); P = w["POS"].reshape(n, -1, 4); V = np.diff(w["POS"], axis=1, prepend=w["POS"][:, :1]).reshape(n, -1, 4) * 10
    parts = ([P, V] if use_state else []) + ([A.reshape(n, A.shape[1], -1)] if use_audio else [])
    return np.concatenate(parts, -1)[:, frames].astype(np.float32)

def fit(xtr, ytr, xte, yte, steps):
    mu, sd = xtr.mean((0, 1)), xtr.std((0, 1)) + 1e-4
    xtr, xte = torch.tensor((xtr - mu) / sd), torch.tensor((xte - mu) / sd); ytr, yte = torch.tensor(ytr), torch.tensor(yte)
    torch.manual_seed(0); m = Reader(xtr.size(-1), xtr.size(1)); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    for _ in range(steps):
        bi = torch.randint(0, len(xtr), (128,)); l = F.cross_entropy(m(xtr[bi]), ytr[bi]); opt.zero_grad(); l.backward(); opt.step()
    m.eval()
    with torch.no_grad(): p = m(xte).argmax(-1)
    return float(((p[yte == 1] == 1).float().mean() + (p[yte == 0] == 0).float().mean()) / 2)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=12000); p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--steps", type=int, default=2500); p.add_argument("--budgets", type=str, default="1000,3000,10000")
    a = p.parse_args(); t0 = time.time()
    w = gen_world(a.n + a.n_test, 16, 32, seed=1000, a_sub=2); n = a.n + a.n_test; A = w["A"]   # (n,16,4,32) lignes = (sous-fenêtre, canal)
    print(f"monde {n} séquences ({time.time() - t0:.0f}s)", flush=True)
    # --- diagnostic stéréo : ILD (log G - log D) sur les frames d'impact vs position x des disques
    L, R = A[:, :, 0::2].sum((2, 3)), A[:, :, 1::2].sum((2, 3)); ild = L - R
    imp = w["IMP"]; xm = w["POS"][..., 0].mean(-1)                      # x moyen des 2 disques (proxy)
    print(f"stéréo : corr(ILD, x moyen des disques) sur frames d'impact = {np.corrcoef(ild[imp], xm[imp])[0, 1]:+.2f} "
          f"(attendu < 0 : à gauche -> plus fort à gauche) | |ILD| moyen {np.abs(ild[imp]).mean():.2f} vs hors impact {np.abs(ild[~imp]).mean():.2f}")
    # --- localisation : moitié stéréo inversée
    lab = (np.arange(n) % 2).astype(np.int64); Asw = A.copy(); Asw[lab == 1] = A[lab == 1][:, :, [1, 0, 3, 2]]
    # --- anticipation
    antic = (w["IMP"][:, 8] | w["IMP"][:, 9]).astype(np.int64)
    tests = {
        "localisation (état+son)": (feats(w, Asw, slice(0, 16)), lab),
        "localisation (son seul)": (feats(w, Asw, slice(0, 16), use_state=False), lab),
        "anticipation (état+son, 0-7)": (feats(w, A, slice(0, 8)), antic),
        "anticipation (état seul, 0-7)": (feats(w, A, slice(0, 8), use_audio=False), antic),
    }
    te = slice(a.n, n)
    for name, (x, y) in tests.items():
        res = [f"{L_} étiq. {fit(x[:L_], y[:L_], x[te], y[te], a.steps):.0%}" for L_ in [int(b) for b in a.budgets.split(",")]]
        print(f"  {name:>30s} : " + " | ".join(res) + f"   ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
