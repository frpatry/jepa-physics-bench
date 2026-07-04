"""
APPRENTISSAGE PAR L'ACTION — actif vs passif à données égales.

Hypothèse : un agent qui CHOISIT quoi pousser (là où son world model est le plus incertain)
apprend la physique PLUS VITE qu'un agent qui pousse au hasard, à nombre d'interactions égal.

Monde : N disques colorés, chacun de MASSE cachée (couleur = identité). Pousser un disque le
déplace de force/masse -> il faut avoir poussé chaque couleur pour prédire ses déplacements.

World model (end-to-end pixels, aucun objet codé en dur) :
  - encodeur CNN : image -> latent
  - ENSEMBLE de K prédicteurs : (latent, action) -> latent après (auto-supervisé, anti-collapse VICReg)
Action choisie par DÉSACCORD de l'ensemble (curiosité) — actif ; ou au hasard — passif.
Le désaccord évite le piège du bruit (les K sont d'accord sur "imprévisible" -> désaccord bas).

Métrique : on décode les positions (sonde) depuis le futur PRÉDIT -> R² vs vraies positions,
en fonction du nombre d'interactions. Courbe active sous/au-dessus de la passive = l'edge, mesuré.

  python explore.py --N 400 --n_obj 4 --H 32
"""
import argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

COLS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], np.float32)
DIRS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)

def render(pos, H, r, n):
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    img = np.zeros((H, H, 3), np.float32)
    for k in range(n):
        img[(xx - pos[k, 0]) ** 2 + (yy - pos[k, 1]) ** 2 < r * r] = COLS[k]
    return img

def gen_pos(rng, n, r):
    p = np.zeros((n, 2), np.float32)
    for k in range(n):
        for _ in range(100):
            c = rng.uniform(r, 1 - r, 2).astype(np.float32)
            if k == 0 or np.all(np.linalg.norm(p[:k] - c, axis=1) > 2.5 * r): p[k] = c; break
    return p

def poke(pos, masses, i, d, force, r):
    p = pos.copy()
    p[i] = np.clip(p[i] + force / masses[i] * DIRS[d], r, 1 - r)
    return p

class Enc(nn.Module):
    """CNN qui GARDE une grille spatiale 8x8 (pas de pooling global) -> la POSITION reste lisible."""
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                              nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                              nn.AdaptiveAvgPool2d(8), nn.Flatten(), nn.Linear(64 * 8 * 8, d))  # grille 8x8 conservée
    def forward(s, x): return s.net(x)

class Dyn(nn.Module):
    def __init__(s, d, na):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d + na, 128), nn.GELU(), nn.Linear(128, d))
    def forward(s, z, a): return z + s.net(torch.cat([z, a], -1))

def vicreg(z):                                                   # anti-collapse (variance + décorrélation)
    z = z - z.mean(0, keepdim=True); std = (z.var(0) + 1e-4).sqrt()
    cov = (z.T @ z) / (z.size(0) - 1); cov = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / z.size(1)
    return 25 * F.relu(1 - std).mean() + cov

def batch(imgs, dev): return torch.tensor(np.stack(imgs).transpose(0, 3, 1, 2)).to(dev)

@torch.no_grad()
def evaluate(enc, ens, a, dev, masses):
    """R² de prédiction des POSITIONS futures (sonde), comparable entre modèles."""
    rng = np.random.default_rng(999); na = a.n_obj * 4
    Bp, Ai, Ap = [], [], []
    for _ in range(400):
        p = gen_pos(rng, a.n_obj, a.r); ai = rng.integers(na); i, dd = ai // 4, ai % 4
        Bp.append(p); Ai.append(ai); Ap.append(poke(p, masses, i, dd, a.force, a.r))
    Bimg = batch([render(p, a.H, a.r, a.n_obj) for p in Bp], dev)
    Aimg = batch([render(p, a.H, a.r, a.n_obj) for p in Ap], dev)
    acts = F.one_hot(torch.tensor(Ai), na).float().to(dev)
    zb = enc(Bimg); za = enc(Aimg)
    pred = torch.stack([m(zb, acts) for m in ens]).mean(0)       # latent futur prédit
    pos_after = torch.tensor(np.stack(Ap).reshape(len(Ap), -1)).to(dev)  # vraies positions futures
    # sonde entraînée DIRECTEMENT sur les latents PRÉDITS (pas de décalage train/inférence) :
    # mesure honnêtement "le résultat est-il DANS la prédiction ?"
    ntr = 200; probe = nn.Sequential(nn.Linear(a.d, 128), nn.GELU(), nn.Linear(128, a.n_obj * 2)).to(dev)
    op = torch.optim.Adam(probe.parameters(), 1e-2)
    with torch.enable_grad():
        for _ in range(300):
            op.zero_grad(); F.mse_loss(probe(pred[:ntr]), pos_after[:ntr]).backward(); op.step()
    pp = probe(pred[ntr:]); gt = pos_after[ntr:]
    ssr = (pp - gt).pow(2).sum(0); sst = (gt - gt.mean(0)).pow(2).sum(0) + 1e-9
    return float((1 - ssr / sst).mean())

def run(mode, a, dev, masses):
    rng = np.random.default_rng(0 if mode == "passif" else 1); na = a.n_obj * 4
    enc = Enc(a.d).to(dev); ens = [Dyn(a.d, na).to(dev) for _ in range(a.K)]
    opt = torch.optim.Adam(list(enc.parameters()) + [p for m in ens for p in m.parameters()], 1e-3)
    Bb, Aa, Bf = [], [], []                                     # images avant, action, images après
    curve = []
    for st in range(a.N):
        pos = gen_pos(rng, a.n_obj, a.r); before = render(pos, a.H, a.r, a.n_obj)
        if mode == "actif" and len(Bb) > a.warmup:              # curiosité = désaccord max de l'ensemble
            with torch.no_grad():
                z = enc(batch([before], dev)).expand(na, -1); A = torch.eye(na, device=dev)
                disag = torch.stack([m(z, A) for m in ens]).var(0).mean(-1)
                ai = int(disag.argmax())
        else:
            ai = int(rng.integers(na))
        i, dd = ai // 4, ai % 4
        Bb.append(before); Aa.append(ai); Bf.append(render(poke(pos, masses, i, dd, a.force, a.r), a.H, a.r, a.n_obj))
        if len(Bb) >= a.bs:                                     # entraîne encodeur + ensemble sur le buffer
            for _ in range(a.train_steps):
                idx = rng.integers(0, len(Bb), a.bs)
                zb = enc(batch([Bb[j] for j in idx], dev)); za = enc(batch([Bf[j] for j in idx], dev))
                ac = F.one_hot(torch.tensor([Aa[j] for j in idx]), na).float().to(dev)
                loss = vicreg(za) + sum(F.smooth_l1_loss(m(zb, ac), za.detach()) for m in ens)
                opt.zero_grad(); loss.backward(); opt.step()
        if (st + 1) % a.eval_every == 0:
            r2 = evaluate(enc, ens, a, dev, masses); curve.append((st + 1, r2))
            print(f"  [{mode}] interactions {st+1:4d}  R² prédiction {r2:.3f}", flush=True)
    return curve

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=400); p.add_argument("--n_obj", type=int, default=4)
    p.add_argument("--H", type=int, default=32); p.add_argument("--r", type=float, default=0.10)
    p.add_argument("--d", type=int, default=128); p.add_argument("--K", type=int, default=4)
    p.add_argument("--force", type=float, default=0.15); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--train_steps", type=int, default=20); p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=25)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    masses = np.linspace(0.3, 1.0, a.n_obj).astype(np.float32)   # masse cachée par couleur
    print(f"device={dev}  {a.n_obj} objets (masses cachées {masses.round(2)})  H={a.H}", flush=True)
    print("\n--- PASSIF (pousse au hasard) ---", flush=True); cp = run("passif", a, dev, masses)
    print("\n--- ACTIF (pousse là où il est incertain) ---", flush=True); ca = run("actif", a, dev, masses)
    print("\n=== ACTIF vs PASSIF (R² de prédiction vs nb d'interactions) ===", flush=True)
    print(f"  {'interactions':>12} | {'passif':>8} | {'actif':>8}", flush=True)
    for (n, rp), (_, ra) in zip(cp, ca):
        print(f"  {n:>12} | {rp:>8.3f} | {ra:>8.3f}  {'<- actif devant' if ra > rp else ''}", flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [n for n, _ in cp]
        plt.figure(figsize=(6, 4)); plt.plot(xs, [r for _, r in cp], "-o", label="passif (hasard)")
        plt.plot(xs, [r for _, r in ca], "-o", label="actif (curiosité)")
        plt.xlabel("interactions"); plt.ylabel("R² prédiction physique"); plt.legend(); plt.grid(alpha=.3)
        plt.title("Apprentissage par l'action : actif vs passif"); plt.tight_layout()
        import os; out = "/content/explore.png" if os.path.isdir("/content") else "explore.png"
        plt.savefig(out); print(f"\nfigure -> {out}", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
