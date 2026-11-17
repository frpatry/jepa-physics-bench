"""
BAC À SABLE — NETTETÉ : où vit le flou, dans la SÉPARATION ou dans le RENDU ?

Retour aux disques (monde qu'on maîtrise, itération rapide) avec le cas DUR : des disques qui
SE TOUCHENT. On sépare enfin deux questions qu'on a toujours mélangées :

  1. SÉPARATION (les masques) : chaque pixel est-il clairement possédé par UN slot, même dans la
     zone de contact entre deux disques ? (c'est ce que font bien DINOv2/V-JEPA)
  2. RENDU (le RGB peint) : le disque peint est-il FRANC (bord net) ou baveux (dégradé) ?

Recette = la gagnante (peel + loss mix + reliquat const + inits par round + vary), inchangée.
On ne bidouille rien : on REGARDE, avec deux chiffres et une figure qui montre masques ET rendu.

Deux mesures, en clair :
  - séparation : parmi les pixels de CONTENU (non-fond), quelle FRACTION est franchement décidée
    (max des masques > 0.8) ? 1.0 = parfait, ~1/K = bouillie.
  - netteté du rendu : au bord des vrais disques, le rendu a-t-il un saut aussi FRANC que la vérité ?
    ratio = |gradient du rendu| / |gradient du vrai| sur la bande des bords. 1 = net, «1 = flou.

  python slots_sharp.py --n 5000 --n_obj 3 --H 48 --K 4 --steps 15000 --bs 64 --sep 1.0
"""
import argparse, math, os
import numpy as np, torch, torch.nn.functional as F
from slots import COLS, Model, mixture_nll, match_error

def gen_touch(n, H, n_obj, r, seed=0, vary=1, sep=1.0):
    """Comme gen(), mais séparation minimale réglable : sep=1.0 -> disques qui SE TOUCHENT
    (bords jointifs) ; sep<1.0 -> ils se CHEVAUCHENT ; sep=2.2 -> l'ancien monde bien espacé."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    X = np.zeros((n, H, H, 3), np.float32); P = np.zeros((n, n_obj, 2), np.float32)
    C = np.full(n, n_obj, np.int64)
    for i in range(n):
        if vary: C[i] = rng.integers(2, n_obj + 1)                        # au moins 2 (pour avoir des contacts)
        cols = rng.permutation(len(COLS))[:C[i]] if vary else np.arange(C[i])
        for k in range(C[i]):
            for _ in range(80):
                c = rng.uniform(r, 1 - r, 2).astype(np.float32)
                if k == 0 or np.all(np.linalg.norm(P[i, :k] - c, axis=1) > sep * r): P[i, k] = c; break
            X[i][(xx - P[i, k, 0]) ** 2 + (yy - P[i, k, 1]) ** 2 < r * r] = COLS[cols[k]]
    return X, P, C

def sharpness(recon, true):
    """Netteté du rendu : |grad(recon)| / |grad(true)| sur la bande des bords vrais.
    recon,true : (B,3,H,W). 1.0 = aussi net que la vérité, «1 = flou."""
    def grad(z):
        gx = (z[..., 1:, :] - z[..., :-1, :]).abs().sum(1)                # (B,H-1,W)
        gy = (z[..., :, 1:] - z[..., :, :-1]).abs().sum(1)
        g = torch.zeros(z.size(0), z.size(2), z.size(3), device=z.device)
        g[:, :-1, :] += gx; g[:, :, :-1] += gy
        return g
    gt, gr = grad(true), grad(recon)
    band = gt > 0.3                                                       # là où la vérité a un vrai bord
    return (gr[band].mean() / (gt[band].mean() + 1e-8)).item()

def decided_content(masks, true, thr=0.8):
    """Séparation : parmi les pixels de contenu (non-fond), fraction franchement décidée.
    masks:(B,K,H,W) somme=1 ; true:(B,3,H,W)."""
    content = true.sum(1) > 0.1                                           # non-noir
    mx = masks.max(1).values                                             # (B,H,W) : possession du pixel
    return (mx[content] > thr).float().mean().item()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000); p.add_argument("--n_obj", type=int, default=3)
    p.add_argument("--H", type=int, default=48); p.add_argument("--r", type=float, default=0.15)
    p.add_argument("--K", type=int, default=4); p.add_argument("--D", type=int, default=64)
    p.add_argument("--slot_dim", type=int, default=12); p.add_argument("--dec_w", type=int, default=32)
    p.add_argument("--iters", type=int, default=3); p.add_argument("--sig", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=15000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-4); p.add_argument("--sep", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, P, C = gen_touch(a.n, a.H, a.n_obj, a.r, vary=1, sep=a.sep)
    Xt = torch.tensor(X.transpose(0, 3, 1, 2))
    print(f"device={dev}  {a.n} images  disques qui se touchent (sep={a.sep}×r)  K={a.K}  sig={a.sig}", flush=True)
    m = Model(a.H, a.K, a.D, slot_dim=a.slot_dim, dec=a.dec_w and "sbd", dec_w=a.dec_w,
              iters=a.iters, init="learned", mode="peel", bg="const").to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    Xe, Pe, Ce = gen_touch(200, a.H, a.n_obj, a.r, seed=7, vary=1, sep=a.sep)
    xe = torch.tensor(Xe.transpose(0, 3, 1, 2)).to(dev)
    for st in range(a.steps):
        bi = np.random.randint(0, a.n, a.bs); img = Xt[bi].to(dev)
        recon, masks, _, rgbs = m(img)
        loss = mixture_nll(img, rgbs, masks, a.sig)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if st % 1000 == 0:
            with torch.no_grad():
                rec, mk, _, _ = m(xe)
                err = match_error(mk, Pe, a.H, Ce)
                sep = decided_content(mk, xe); shp = sharpness(rec, xe)
            print(f"  step {st:5d}  recon {loss.item():.3f}  err-pos {err:.3f}  |  SÉPARATION {sep:.2f}"
                  f"  NETTETÉ-rendu {shp:.2f}  (1=net, bas=flou)", flush=True)
    # ===== figure décisive : masques (séparation) ET rendu, sur des paires qui se touchent =====
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Xv, Pv, Cv = gen_touch(4, a.H, a.n_obj, a.r, seed=3, vary=1, sep=a.sep)
        v = torch.tensor(Xv.transpose(0, 3, 1, 2)).to(dev)
        with torch.no_grad():
            rec, mk, _, rgbs = m(v)
        ncol = 3 + a.K
        fig, ax = plt.subplots(4, ncol, figsize=(2 * ncol, 8))
        for i in range(4):
            ax[i, 0].imshow(Xv[i].clip(0, 1)); ax[i, 0].set_title("image" if i == 0 else "")
            ax[i, 1].imshow(rec[i].permute(1, 2, 0).cpu().clip(0, 1)); ax[i, 1].set_title("RENDU (recon)" if i == 0 else "")
            # bords : vrai (vert) vs rendu (rouge) superposés -> voir le flou du rendu
            gt = (np.abs(np.diff(Xv[i].sum(-1), axis=0, prepend=0)) +
                  np.abs(np.diff(Xv[i].sum(-1), axis=1, prepend=0))) > 0.3
            rc = rec[i].permute(1, 2, 0).cpu().numpy()
            grc = (np.abs(np.diff(rc.sum(-1), axis=0, prepend=0)) +
                   np.abs(np.diff(rc.sum(-1), axis=1, prepend=0)))
            edge = np.zeros((a.H, a.H, 3), np.float32); edge[gt] = [0, 1, 0]; edge[:, :, 0] = np.clip(grc, 0, 1)
            ax[i, 2].imshow(edge); ax[i, 2].set_title("bords vrai(V)/rendu(R)" if i == 0 else "")
            for kk in range(a.K):
                ax[i, 3 + kk].imshow(mk[i, kk].cpu(), cmap="viridis", vmin=0, vmax=1)
                nom = "reliquat" if kk == a.K - 1 else f"masque {kk}"
                ax[i, 3 + kk].set_title(nom if i == 0 else "")
            for j in range(ncol): ax[i, j].axis("off")
        with torch.no_grad():
            sep = decided_content(mk, v); shp = sharpness(rec, v)
        fig.suptitle(f"disques qui se touchent — SÉPARATION {sep:.2f} (1=net)   NETTETÉ-rendu {shp:.2f} (1=net)")
        out = "/content/slots_sharp.png" if os.path.isdir("/content") else "slots_sharp.png"
        plt.tight_layout(); plt.savefig(out)
        print(f"\nfigure -> {out}  (masques = séparation ; recon+bords = rendu)", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:80], flush=True)

if __name__ == "__main__":
    main()
