"""
DYNAMIQUE SUR LES SLOTS — le world model object-centré (SAVi-lite sur la recette gagnante de slots.py).

La fondation statique est acquise (slots.py : err 0.048, 1 slot/objet, comptage émergent).
Ici on pose la dynamique DESSUS : le monde bouge (disques avec vitesses + rebonds murs), le modèle
voit les frames t0 et t1, et doit IMAGINER t2 — qui n'est jamais encodée (prédiction honnête,
contrairement au mur n°2 où l'encodeur bidirectionnel trichait en voyant le futur).

Mécanique :
  t0 : épluchage (peel) -> slots S0                    [perception, recette slots.py]
  t1 : épluchage initialisé par S0 -> slots S1          [binding temporel à la SAVi : le round j
                                                         repart de « son » objet -> suivi gratuit]
  t2 : Ŝ2 = S1 + g([S1, S1-S0])                        [dynamique par slot, MLP PARTAGÉ : chaque
                                                         objet évolue indépendamment (factorisation)]
       décodage de Ŝ2 SEUL (aucune image) = futur imaginé, comparé au vrai t2.

Mesures honnêtes : err t0 (perception), err t2 IMAGINÉ vs err t2 COPIE (baseline sans dynamique :
les masques de t1 face aux positions de t2). Si imaginé << copie, g a appris le mouvement.

  python slots_dyn.py --n 5000 --n_obj 4 --H 48 --K 5 --steps 20000 --bs 64
"""
import argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from slots import COLS, PosEmbed, SlotAttention, mixture_nll, match_error

def gen_seq(n, H, n_obj, r, T=3, seed=0, vary=1, smin=0.04, smax=0.12):
    """Séquences de T frames : disques à vitesse constante + rebonds murs.
    vary=1 : nombre d'objets 1..n_obj + couleurs au hasard (la leçon du run 12)."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    X = np.zeros((n, T, H, H, 3), np.float32); P = np.zeros((n, T, n_obj, 2), np.float32)
    C = np.full(n, n_obj, np.int64)
    for i in range(n):
        if vary: C[i] = rng.integers(1, n_obj + 1)
        cols = rng.permutation(len(COLS))[:C[i]] if vary else np.arange(C[i])
        for k in range(C[i]):
            for _ in range(50):
                c = rng.uniform(r, 1 - r, 2).astype(np.float32)
                if k == 0 or np.all(np.linalg.norm(P[i, 0, :k] - c, axis=1) > 2.2 * r): P[i, 0, k] = c; break
        sp = rng.uniform(smin, smax, C[i]).astype(np.float32)
        th = rng.uniform(0, 2 * np.pi, C[i]).astype(np.float32)
        V = np.stack([sp * np.cos(th), sp * np.sin(th)], -1)
        for t in range(T):
            for k in range(C[i]):
                if t > 0:
                    p = P[i, t - 1, k] + V[k]
                    for d in range(2):                                     # rebond mur
                        if p[d] < r: p[d] = 2 * r - p[d]; V[k, d] = -V[k, d]
                        if p[d] > 1 - r: p[d] = 2 * (1 - r) - p[d]; V[k, d] = -V[k, d]
                    P[i, t, k] = p
                X[i, t][(xx - P[i, t, k, 0]) ** 2 + (yy - P[i, t, k, 1]) ** 2 < r * r] = COLS[cols[k]]
    return X, P, C

class DynModel(nn.Module):
    def __init__(s, H, K, D=64, res=24, slot_dim=16, dec_w=32, iters=3):
        super().__init__(); s.res, s.K, s.D, s.slot_dim, s.H = res, K, D, slot_dim, H
        s.enc = nn.Sequential(nn.Conv2d(3, D, 5, 1, 2), nn.ReLU(), nn.Conv2d(D, D, 5, 2, 2), nn.ReLU(),
                              nn.Conv2d(D, D, 5, 1, 2), nn.ReLU())
        s.pe = PosEmbed(D, res); s.mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        s.sa = SlotAttention(2, D, iters, "learned")                       # le cerneur objet/reste
        s.peel_init = nn.Parameter(torch.randn(1, K - 1, 2, D) * 0.5)      # rôles distincts par round
        s.down = nn.Linear(D, slot_dim)                                    # goulot 16d
        s.pe_d = PosEmbed(slot_dim, H)
        s.dec = nn.Sequential(nn.Conv2d(slot_dim, dec_w, 1), nn.ReLU(),    # SBD 1x1 faible
                              nn.Conv2d(dec_w, dec_w, 1), nn.ReLU(), nn.Conv2d(dec_w, 4, 1))
        s.bg = nn.Parameter(torch.zeros(1, 3, 1, 1))                       # reliquat = couleur unie
        s.bg_logit = nn.Parameter(torch.zeros(1))                          # compositing du futur imaginé
        s.g = nn.Sequential(nn.Linear(2 * D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))  # dynamique par slot
    def feats(s, img):
        B = img.size(0); f = s.enc(img).permute(0, 2, 3, 1)
        return s.mlp(s.pe(f).reshape(B, s.res * s.res, s.D))
    def decode_one(s, sl):                                                 # sl:(B,slot_dim) -> rgb, logit alpha
        B = sl.size(0)
        x = sl.reshape(B, s.slot_dim, 1, 1).expand(-1, -1, s.H, s.H).permute(0, 2, 3, 1)
        x = s.pe_d(x).permute(0, 3, 1, 2); out = s.dec(x)
        return out[:, :3], out[:, 3:4]
    def peel(s, f, init=None):
        """Épluchage (recette slots.py). init:(B,K-1,D) = slots objets du pas précédent
        (corrector à la SAVi : le round j repart de « son » objet -> binding temporel)."""
        B, H = f.size(0), s.H
        scope = torch.ones(B, 1, H, H, device=f.device)
        masks, rgbs, S = [], [], []
        for j in range(s.K - 1):
            sc = F.adaptive_avg_pool2d(scope, s.res).reshape(B, s.res * s.res, 1)
            if init is None: s0 = s.peel_init[:, j].expand(B, -1, -1)
            else: s0 = torch.stack([init[:, j], s.peel_init[:, j, 1].expand(B, -1)], 1)
            slots2, _ = s.sa(f, scope=sc, slots0=s0)
            S.append(slots2[:, 0])
            rgb, alog = s.decode_one(s.down(slots2).reshape(B * 2, s.slot_dim))
            rgb = rgb.reshape(B, 2, 3, H, H); alog = alog.reshape(B, 2, 1, H, H)
            a = torch.softmax(alog, 1)
            masks.append(scope * a[:, 0]); rgbs.append(rgb[:, 0]); scope = scope * a[:, 1]
        masks.append(scope); rgbs.append(s.bg.expand(B, 3, H, H))
        return torch.stack(masks, 1), torch.stack(rgbs, 1), torch.stack(S, 1)  # (B,K,1,H,H) (B,K,3,H,H) (B,K-1,D)
    def imagine(s, S):
        """Décode le futur depuis les slots PRÉDITS seuls (aucune image) : chaque slot peint sa part,
        compétition softmax par pixel + logit fond appris."""
        B, Km1, _ = S.shape; H = s.H
        rgb, alog = s.decode_one(s.down(S).reshape(B * Km1, s.slot_dim))
        rgb = rgb.reshape(B, Km1, 3, H, H); alog = alog.reshape(B, Km1, 1, H, H)
        logits = torch.cat([alog, s.bg_logit.reshape(1, 1, 1, 1, 1).expand(B, 1, 1, H, H)], 1)
        a = torch.softmax(logits, 1)
        rgbs = torch.cat([rgb, s.bg.expand(B, 3, H, H).unsqueeze(1)], 1)
        return a, rgbs                                                     # (B,K,1,H,H) (B,K,3,H,H)
    def forward(s, x0, x1):
        m0, r0, S0 = s.peel(s.feats(x0))
        m1, r1, S1 = s.peel(s.feats(x1), init=S0)                          # suivi : repart des slots de t0
        Sh = S1 + s.g(torch.cat([S1, S1 - S0], -1))                        # résiduel ; vitesse = S1-S0
        m2, r2 = s.imagine(Sh)
        return (m0, r0), (m1, r1), (m2, r2)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000); p.add_argument("--n_obj", type=int, default=4)
    p.add_argument("--H", type=int, default=48); p.add_argument("--r", type=float, default=0.15)
    p.add_argument("--K", type=int, default=5); p.add_argument("--D", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-4); p.add_argument("--iters", type=int, default=3)
    p.add_argument("--slot_dim", type=int, default=16); p.add_argument("--dec_w", type=int, default=32)
    p.add_argument("--sig", type=float, default=0.1); p.add_argument("--vary", type=int, default=1)
    p.add_argument("--smin", type=float, default=0.04); p.add_argument("--smax", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, P, C = gen_seq(a.n, a.H, a.n_obj, a.r, seed=a.seed, vary=a.vary, smin=a.smin, smax=a.smax)
    Xt = torch.tensor(X.transpose(0, 1, 4, 2, 3))                          # (n,T,3,H,H)
    nobj = f"1..{a.n_obj} objets (variable)" if a.vary else f"{a.n_obj} objets"
    print(f"device={dev}  {a.n} séquences de 3 frames  {nobj}  K={a.K}  seed={a.seed}"
          f"  (voit t0,t1 -> IMAGINE t2, jamais encodé)", flush=True)
    m = DynModel(a.H, a.K, a.D, slot_dim=a.slot_dim, dec_w=a.dec_w, iters=a.iters).to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    Xe, Pe, Ce = gen_seq(200, a.H, a.n_obj, a.r, seed=7, vary=a.vary, smin=a.smin, smax=a.smax)
    Xet = torch.tensor(Xe.transpose(0, 1, 4, 2, 3)).to(dev)
    for st in range(a.steps):
        bi = np.random.randint(0, a.n, a.bs)
        x0, x1, x2 = Xt[bi, 0].to(dev), Xt[bi, 1].to(dev), Xt[bi, 2].to(dev)
        (m0, r0), (m1, r1), (m2, r2) = m(x0, x1)
        loss = (mixture_nll(x0, r0, m0[:, :, 0], a.sig) + mixture_nll(x1, r1, m1[:, :, 0], a.sig)
                + mixture_nll(x2, r2, m2[:, :, 0], a.sig))                 # t2 : futur imaginé vs vrai
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if st % 500 == 0:
            with torch.no_grad():
                (tm0, tr0), (tm1, _), (tm2, tr2) = m(Xet[:, 0], Xet[:, 1])
                err0 = match_error(tm0[:, :, 0], Pe[:, 0], a.H, Ce)        # perception t0
                errC = match_error(tm1[:, :, 0], Pe[:, 2], a.H, Ce)        # baseline COPIE (t1 -> t2 sans bouger)
                errP = match_error(tm2[:, :, 0], Pe[:, 2], a.H, Ce)        # futur IMAGINÉ vs vrai t2
                mse2 = F.mse_loss((tr2 * tm2).sum(1), Xet[:, 2]).item()
            print(f"  step {st:5d}  perception t0 {err0:.3f}  |  t2 imaginé {errP:.3f}  vs copie {errC:.3f}"
                  f"  (imaginé<copie = g a appris le mouvement)  mse t2 {mse2:.4f}", flush=True)
    # figure : t0, t1, vrai t2, t2 imaginé + masques prédits
    try:
        import matplotlib, os; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with torch.no_grad():
            Xv, Pv, Cv = gen_seq(4, a.H, a.n_obj, a.r, seed=3, vary=a.vary, smin=a.smin, smax=a.smax)
            Xvt = torch.tensor(Xv.transpose(0, 1, 4, 2, 3)).to(dev)
            _, _, (m2, r2) = m(Xvt[:, 0], Xvt[:, 1])
            recon2 = (r2 * m2).sum(1)
        fig, ax = plt.subplots(4, a.K + 4, figsize=(2 * (a.K + 4), 8))
        titles = ["t0", "t1", "t2 (vrai)", "t2 IMAGINÉ"]
        for i in range(4):
            for t in range(3): ax[i, t].imshow(Xv[i, t].clip(0, 1)); ax[i, t].set_title(titles[t] if i == 0 else "")
            ax[i, 3].imshow(recon2[i].permute(1, 2, 0).cpu().clip(0, 1)); ax[i, 3].set_title(titles[3] if i == 0 else "")
            for k in range(a.K):
                ax[i, k + 4].imshow(m2[i, k, 0].cpu(), cmap="viridis")
                ax[i, k + 4].set_title(f"slot {k} prédit" if i == 0 else "")
            for j in range(a.K + 4): ax[i, j].axis("off")
        out = "/content/slots_dyn.png" if os.path.isdir("/content") else "slots_dyn.png"
        plt.tight_layout(); plt.savefig(out); print(f"\nfigure -> {out}  (t2 imaginé doit matcher t2 vrai)", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
