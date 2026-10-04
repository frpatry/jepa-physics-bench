"""
VALIDATION DU MONDE (instrument de mesure, PAS un modèle du projet) — avant d'entraîner un JEPA,
on vérifie que l'information cherchée EXISTE et est atteignable dans le monde : un lecteur
supervisé reçoit la VISION PARFAITE (vraies positions + vitesses des disques, aucun pixel) et
le son, et doit retrouver les propriétés cachées.

  - matériau par disque : 67 % = plafond « ensemble connu, attribution inconnue » (pas de LIAGE
    son<->disque) ; > 67 % = le liage est possible dans ce monde.
  - ratio de masses (séquences avec choc) : déductible des trajectoires (conservation de la
    quantité de mouvement) -> la vision PEUT en principe le voir.
  - log-masse par disque.

Si un monde échoue ici, aucun JEPA ne réussira dessus : on ne lance pas Colab.

  python av_world_check.py --n 8000 --T 8 --a_sub 1          # monde v1
  python av_world_check.py --n 20000 --T 16 --a_sub 3        # candidat v2
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import gen_world

class Reader(nn.Module):
    """transformer sur les frames : token t = [positions, vitesses, son de la frame t]."""
    def __init__(s, din, T, d=96, nl=3):
        super().__init__()
        s.emb = nn.Linear(din, d); s.pos = nn.Parameter(torch.zeros(1, T, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.1)
        s.tr = nn.TransformerEncoder(layer, nl); s.mat = nn.Linear(d, 6); s.lm = nn.Linear(d, 3)
    def forward(s, x):
        h = s.tr(s.emb(x) + s.pos).mean(1)
        return s.mat(h), s.lm(h)

def feats(w, use_audio, use_state):
    n, T = w["POS"].shape[:2]
    P = w["POS"].reshape(n, T, 4); V = np.diff(w["POS"], axis=1, prepend=w["POS"][:, :1]).reshape(n, T, 4) * 10
    parts = []
    if use_state: parts += [P, V]
    if use_audio: parts.append(w["A"].reshape(n, T, -1))
    return np.concatenate(parts, -1).astype(np.float32)

def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def run(wtr, wte, use_audio, use_state, a, dev):
    xtr, xte = feats(wtr, use_audio, use_state), feats(wte, use_audio, use_state)
    mu, sd = xtr.mean((0, 1)), xtr.std((0, 1)) + 1e-4
    xtr, xte = torch.tensor((xtr - mu) / sd), torch.tensor((xte - mu) / sd)
    def ylm(w):
        lm = torch.tensor(w["LM"]); return torch.stack([lm[:, 0], lm[:, 1], lm[:, 0] - lm[:, 1]], 1)
    ytr, yte = ylm(wtr), ylm(wte); ym, ys = ytr.mean(0), ytr.std(0)
    mtr = torch.tensor(wtr["MAT"]).long()
    torch.manual_seed(0); m = Reader(xtr.size(-1), xtr.size(1)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    for it in range(a.steps):
        bi = torch.randint(0, len(xtr), (256,)); pm, pl = m(xtr[bi].to(dev))
        loss = F.cross_entropy(pm.view(-1, 3), mtr[bi].to(dev).view(-1)) * (1 if use_audio else 0) \
            + F.mse_loss(pl, ((ytr[bi] - ym) / ys).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): pm, pl = m(xte.to(dev))
    pm, pl = pm.cpu(), pl.cpu() * ys + ym; hit = torch.tensor(wte["HIT"])
    mat = float((pm.view(-1, 2, 3).argmax(-1) == torch.tensor(wte["MAT"])).float().mean())
    lm = (r2(pl[:, 0], yte[:, 0]) + r2(pl[:, 1], yte[:, 1])) / 2
    return mat, lm, r2(pl[hit, 2], yte[hit, 2])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000); p.add_argument("--n_test", type=int, default=3000)
    p.add_argument("--T", type=int, default=8); p.add_argument("--a_sub", type=int, default=1)
    p.add_argument("--r", type=float, default=0.12); p.add_argument("--smin", type=float, default=0.05)
    p.add_argument("--smax", type=float, default=0.11); p.add_argument("--hum", type=float, default=0.0); p.add_argument("--steps", type=int, default=3000)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time(); kw = dict(T=a.T, H=16, r=a.r, a_sub=a.a_sub, smin=a.smin, smax=a.smax, hum=a.hum)
    wtr, wte = gen_world(a.n, seed=0, **kw), gen_world(a.n_test, seed=999, **kw)
    nwall = wtr["IMP"][:, 1:].sum(1).mean()
    print(f"monde T={a.T} a_sub={a.a_sub} n={a.n} : choc disque-disque {wtr['HIT'].mean():.0%}, "
          f"frames avec impact/séq {nwall:.1f} ({time.time() - t0:.0f}s)", flush=True)
    for name, ua, us in [("état parfait seul", 0, 1), ("son seul", 1, 0), ("état parfait + son", 1, 1)]:
        mat, lm, ratio = run(wtr, wte, ua, us, a, dev)
        print(f"  {name:>20s} | matériau {mat:.0%} (67%=pas de liage) | log-masse R² {lm:+.2f} | ratio choc R² {ratio:+.2f}", flush=True)
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
