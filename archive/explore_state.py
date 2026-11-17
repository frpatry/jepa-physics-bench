"""
CONTRÔLE — actif vs passif sur l'ÉTAT PROPRE (positions vraies), SANS pixels.

But : ISOLER la question « la curiosité rend-elle l'apprentissage plus efficace ? » du
problème (dur, séparé) de la perception. Ici l'état = les positions des objets directement.
Si l'actif bat nettement le passif => le PRINCIPE de l'edge est validé ; la version pixels
(transposable) reviendra à y ajouter la perception.

Monde : N disques à MASSE cachée (par index). Action = (pousser disque i, direction).
Résultat : disque i se déplace de force/masse. Prédicteur = ensemble MLP (état,action)->état'.
Actif = action de désaccord max de l'ensemble (curiosité) ; passif = hasard. Métrique = R².

  python explore_state.py --N 800 --n_obj 5
"""
import argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DIRS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)

def gen_pos(rng, n, r):
    return rng.uniform(r, 1 - r, (n, 2)).astype(np.float32)     # état abstrait : positions libres (chevauchement ok)

def poke(pos, masses, i, d, force, r, noise=0.0, rng=None):
    p = pos.copy(); disp = force / masses[i] * DIRS[d]
    if noise > 0 and rng is not None: disp = disp + noise * rng.standard_normal(2).astype(np.float32)
    p[i] = np.clip(p[i] + disp, r, 1 - r); return p

class Dyn(nn.Module):
    def __init__(s, sdim, na):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(sdim + na, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, sdim))
    def forward(s, x, a): return x + s.net(torch.cat([x, a], -1))

@torch.no_grad()
def evaluate(ens, a, masses, dev):
    rng = np.random.default_rng(999); na = a.n_obj * 4; B, A, Y, I = [], [], [], []
    for _ in range(600):
        p = gen_pos(rng, a.n_obj, a.r); ai = rng.integers(na); i, dd = ai // 4, ai % 4
        B.append(p.flatten()); A.append(ai); Y.append(poke(p, masses, i, dd, a.force, a.r).flatten()); I.append(i)
    x = torch.tensor(np.stack(B)).to(dev); ac = F.one_hot(torch.tensor(A), na).float().to(dev)
    y = torch.tensor(np.stack(Y)).to(dev); pred = torch.stack([m(x, ac) for m in ens]).mean(0)
    rows = torch.arange(len(I), device=dev); I = torch.tensor(I, device=dev)          # objet POUSSÉ uniquement
    py = torch.stack([pred[rows, 2 * I], pred[rows, 2 * I + 1]], -1)                   # sa position prédite
    ty = torch.stack([y[rows, 2 * I], y[rows, 2 * I + 1]], -1)                         # sa vraie position
    ssr = (py - ty).pow(2).sum(0); sst = (ty - ty.mean(0)).pow(2).sum(0) + 1e-9
    return float((1 - ssr / sst).mean())

def run(mode, a, masses, dev):
    rng = np.random.default_rng(0 if mode == "passif" else 1); na = a.n_obj * 4; sdim = a.n_obj * 2
    ens = [Dyn(sdim, na).to(dev) for _ in range(a.K)]
    opt = torch.optim.Adam([p for m in ens for p in m.parameters()], 1e-3)
    Xs, Aa, Ys = [], [], []; curve = []
    for st in range(a.N):
        pos = gen_pos(rng, a.n_obj, a.r); s = pos.flatten()
        if mode == "actif" and len(Xs) > a.warmup:                 # curiosité = désaccord max de l'ensemble
            with torch.no_grad():
                x = torch.tensor(s).to(dev).expand(na, -1); A = torch.eye(na, device=dev)
                ai = int(torch.stack([m(x, A) for m in ens]).var(0).mean(-1).argmax())
        else:
            ai = int(rng.integers(na))
        i, dd = ai // 4, ai % 4
        Xs.append(s); Aa.append(ai); Ys.append(poke(pos, masses, i, dd, a.force, a.r, a.noise, rng).flatten())
        if len(Xs) >= a.bs:
            for _ in range(a.train_steps):
                idx = rng.integers(0, len(Xs), a.bs)
                x = torch.tensor(np.stack([Xs[j] for j in idx])).to(dev); y = torch.tensor(np.stack([Ys[j] for j in idx])).to(dev)
                ac = F.one_hot(torch.tensor([Aa[j] for j in idx]), na).float().to(dev)
                loss = sum(F.mse_loss(m(x, ac), y) for m in ens)
                opt.zero_grad(); loss.backward(); opt.step()
        if (st + 1) % a.eval_every == 0:
            r2 = evaluate(ens, a, masses, dev); curve.append((st + 1, r2))
            print(f"  [{mode}] interactions {st+1:4d}  R² {r2:.3f}", flush=True)
    return curve

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=400); p.add_argument("--n_obj", type=int, default=60)
    p.add_argument("--r", type=float, default=0.05); p.add_argument("--K", type=int, default=5)
    p.add_argument("--force", type=float, default=0.15); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.03, help="bruit d'observation -> plusieurs pokes nécessaires par objet")
    p.add_argument("--train_steps", type=int, default=20); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=20)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    masses = np.linspace(0.3, 1.0, a.n_obj).astype(np.float32)
    print(f"device={dev}  {a.n_obj} objets (masses cachées {masses.round(2)})  ÉTAT PROPRE (pas de pixels)", flush=True)
    print("\n--- PASSIF ---", flush=True); cp = run("passif", a, masses, dev)
    print("\n--- ACTIF ---", flush=True); ca = run("actif", a, masses, dev)
    print("\n=== ACTIF vs PASSIF (R² vs interactions) ===", flush=True)
    print(f"  {'interactions':>12} | {'passif':>8} | {'actif':>8}", flush=True)
    for (n, rp), (_, ra) in zip(cp, ca):
        print(f"  {n:>12} | {rp:>8.3f} | {ra:>8.3f}  {'<- actif devant' if ra > rp + 0.01 else ''}", flush=True)
    try:
        import matplotlib, os; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [n for n, _ in cp]
        plt.figure(figsize=(6, 4)); plt.plot(xs, [r for _, r in cp], "-o", label="passif (hasard)")
        plt.plot(xs, [r for _, r in ca], "-o", label="actif (curiosité)")
        plt.xlabel("interactions"); plt.ylabel("R² prédiction"); plt.legend(); plt.grid(alpha=.3)
        plt.title("Curiosité vs hasard (état propre)"); plt.tight_layout()
        out = "/content/explore_state.png" if os.path.isdir("/content") else "explore_state.png"
        plt.savefig(out); print(f"\nfigure -> {out}", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
