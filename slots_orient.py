"""
BAC À SABLE — ORIENTATION : la recette voit-elle COMMENT une forme est tournée ?

Trou jamais testé (insight utilisateur) : un cercle n'a pas d'orientation, donc « position 0.021
sur les disques » ne dit RIEN sur la perception d'une forme tournée. Le T est le premier cas où
l'angle compte, et on l'a rencontré directement sur le vrai Push-T (basse résolution, faible
contraste, plein d'objets). Ici on l'isole dans le régime FACILE : UNE forme, fond noir, fort
contraste — et on fait varier la RÉSOLUTION.

Forme = T (branches + tige, orientation sans ambiguïté) ou cercle (témoin, pas d'orientation).
Recette gagnante inchangée (peel + mix + reliquat const). Position/couleur/angle au hasard.

Trois sorties :
  1. figure : la reconstruction ressemble-t-elle à un vrai T (branches) ou à une boule ?
  2. netteté du rendu (comme le bac à sable précédent).
  3. LE décisif : une petite SONDE lit l'angle depuis le slot -> erreur d'angle en degrés.
     basse = l'orientation est perçue ; ~90° (baseline) = elle ne l'est pas.

Balayer la résolution :  --H 48   puis  --H 64   puis  --H 96

  python slots_orient.py --shape T --H 64 --steps 12000 --probe_steps 4000
"""
import argparse, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from slots import COLS, Model, mixture_nll, match_error
from slots_sharp import sharpness

def perceive(m, img):
    """Extrait les slots-OBJET (B,K-1,D) du peel — réplique fidèle du front-end de Model.forward
    (le Model de slots.py ne les expose pas ; on ne touche pas au fichier validé)."""
    B, H = img.size(0), img.size(2)
    f = m.enc(img).permute(0, 2, 3, 1)
    f = m.pe(f).reshape(B, m.res * m.res, m.D); f = m.mlp(f)
    scope = torch.ones(B, 1, H, H, device=img.device); S = []
    for j in range(m.K - 1):
        sc = F.adaptive_avg_pool2d(scope, m.res).reshape(B, m.res * m.res, 1)
        slots2, _ = m.sa(f, scope=sc, slots0=m.peel_init[:, j].expand(B, -1, -1))
        S.append(slots2[:, 0])                                            # le slot OBJET de ce round
        _, alog = m.decode_one(m.down(slots2).reshape(B * 2, m.slot_dim), H)
        a = torch.softmax(alog.reshape(B, 2, 1, H, H), dim=1)
        scope = scope * a[:, 1]
    return torch.stack(S, 1)                                              # (B,K-1,D)

def gen_oriented(n, H, seed=0, shape="T", scale=0.16):
    """UNE forme par image, fond noir, position + couleur + angle au hasard.
    T : crossbar (haut) + tige (bas), orientation nette. cercle : témoin (angle ignoré)."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    X = np.zeros((n, H, H, 3), np.float32); P = np.zeros((n, 1, 2), np.float32); A = np.zeros(n, np.float32)
    a, th, voff, slen = 0.85 * scale, 0.22 * scale, 0.5 * scale, 0.9 * scale
    m = 0.30                                                              # marge pour que la forme tournée reste dans le cadre
    for i in range(n):
        c = rng.uniform(m, 1 - m, 2).astype(np.float32); ang = rng.uniform(0, 2 * np.pi)
        P[i, 0] = c; A[i] = ang; col = COLS[rng.integers(0, len(COLS))]
        du = xx - c[0]; dv = yy - c[1]
        u = du * np.cos(ang) + dv * np.sin(ang); v = -du * np.sin(ang) + dv * np.cos(ang)
        if shape == "circle":
            mask = du * du + dv * dv < scale * scale
        else:                                                            # T : crossbar en haut + tige en bas
            crossbar = (np.abs(u) < a) & (np.abs(v - voff) < th)
            stem = (np.abs(u) < th) & (v > -slen) & (v < voff)
            mask = crossbar | stem
        X[i][mask] = col
    return X, P, A

class PoseProbe(nn.Module):
    """Lit (x, y, sinθ, cosθ) depuis le(s) slot(s)-objet(s). Pooling attentif -> MLP."""
    def __init__(s, D, hid=128):
        super().__init__()
        s.q = nn.Parameter(torch.randn(1, 1, D) * 0.5)
        s.att = nn.MultiheadAttention(D, 4, batch_first=True)
        s.mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, hid), nn.ReLU(),
                              nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 4))
    def forward(s, S):
        q = s.q.expand(S.size(0), -1, -1); pooled, _ = s.att(q, S, S)
        o = s.mlp(pooled[:, 0])
        return torch.sigmoid(o[:, :2]), o[:, 2:] / (o[:, 2:].norm(dim=-1, keepdim=True) + 1e-8)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=["T", "circle"], default="T")
    p.add_argument("--n", type=int, default=6000); p.add_argument("--H", type=int, default=64)
    p.add_argument("--K", type=int, default=2); p.add_argument("--D", type=int, default=64)
    p.add_argument("--slot_dim", type=int, default=12); p.add_argument("--dec_w", type=int, default=48)
    p.add_argument("--iters", type=int, default=3); p.add_argument("--sig", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=12000); p.add_argument("--probe_steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, P, A = gen_oriented(a.n, a.H, shape=a.shape)
    Xt = torch.tensor(X.transpose(0, 3, 1, 2))
    print(f"device={dev}  forme={a.shape}  H={a.H} (grille {a.H//2})  K={a.K}  sig={a.sig}  {a.n} images", flush=True)
    m = Model(a.H, a.K, a.D, res=a.H // 2, slot_dim=a.slot_dim, dec="sbd", dec_w=a.dec_w,
              iters=a.iters, init="learned", mode="peel", bg="const").to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    Xe, Pe, Ae = gen_oriented(300, a.H, seed=7, shape=a.shape)
    xe = torch.tensor(Xe.transpose(0, 3, 1, 2)).to(dev)
    for st in range(a.steps):                                            # 1) PERCEPTION (recette gagnante)
        bi = np.random.randint(0, a.n, a.bs); img = Xt[bi].to(dev)
        recon, masks, _, rgbs = m(img)
        loss = mixture_nll(img, rgbs, masks, a.sig)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if st % 2000 == 0:
            with torch.no_grad():
                rec, mk, _, _ = m(xe); err = match_error(mk, Pe, a.H, np.ones(len(Pe), int))
                shp = sharpness(rec, xe)
            print(f"  step {st:5d}  recon {loss.item():.3f}  pos {err:.3f}  netteté-rendu {shp:.2f}", flush=True)
    m.eval()
    for q in m.parameters(): q.requires_grad_(False)
    # 2) SONDE d'angle sur les slots gelés
    probe = PoseProbe(a.D).to(dev); popt = torch.optim.Adam(probe.parameters(), 5e-4)
    At = torch.tensor(A).to(dev); Pt = torch.tensor(P[:, 0]).to(dev)
    Aet = torch.tensor(Ae).to(dev); Pet = torch.tensor(Pe[:, 0]).to(dev)
    print(f"--- sonde d'angle ({a.probe_steps} steps) ---", flush=True)
    for st in range(a.probe_steps):
        bi = np.random.randint(0, a.n, a.bs); img = Xt[bi].to(dev)
        with torch.no_grad(): S = perceive(m, img)
        xy, ang = probe(S.detach())
        tgt_ang = torch.stack([torch.sin(At[bi]), torch.cos(At[bi])], -1)
        loss = ((xy - Pt[bi]) ** 2).sum(-1).mean() + 0.5 * ((ang - tgt_ang) ** 2).sum(-1).mean()
        popt.zero_grad(); loss.backward(); popt.step()
        if st % 1000 == 0:
            with torch.no_grad():
                Se = perceive(m, xe); xy, ang = probe(Se)
                pe = (xy - Pet).norm(dim=-1).mean().item()
                sc = torch.stack([torch.sin(Aet), torch.cos(Aet)], -1)
                ae = torch.acos((ang * sc).sum(-1).clamp(-1, 1)) * 180 / math.pi
                mang = torch.atan2(torch.sin(Aet).mean(), torch.cos(Aet).mean())
                base = (torch.acos((torch.stack([torch.sin(mang), torch.cos(mang)]) * sc).sum(-1).clamp(-1, 1))
                        * 180 / math.pi).mean().item()
            tag = "(cercle : angle sans objet)" if a.shape == "circle" else ""
            print(f"  probe {st:5d}  pos {pe:.3f}  |  ANGLE méd {ae.median():.0f}° moy {ae.mean():.0f}°"
                  f"  (baseline {base:.0f}°) {tag}", flush=True)
    # 3) figures : recon (T ou boule ?) + nuage angle
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with torch.no_grad():
            rec, mk, _, _ = m(xe[:4])
            Se = perceive(m, xe); _, ang = probe(Se)
        fig, ax = plt.subplots(2, 4, figsize=(14, 7))
        for i in range(4):
            ax[0, i].imshow(Xe[i].clip(0, 1)); ax[0, i].set_title("vrai" if i == 0 else "")
            ax[0, i].axis("off")
        for i in range(3):
            ax[1, i].imshow(rec[i].permute(1, 2, 0).cpu().clip(0, 1)); ax[1, i].set_title("RECON" if i == 0 else "")
            ax[1, i].axis("off")
        at = (torch.atan2(torch.sin(Aet), torch.cos(Aet)) * 180 / math.pi).cpu()
        ap = (torch.atan2(ang[:, 0], ang[:, 1]) * 180 / math.pi).cpu()
        ax[1, 3].scatter(at, ap, s=8, alpha=0.5); ax[1, 3].plot([-180, 180], [-180, 180], "r--")
        ax[1, 3].set_title("angle vrai vs lu"); ax[1, 3].set_xlabel("vrai°"); ax[1, 3].set_ylabel("lu°")
        fig.suptitle(f"forme={a.shape}  H={a.H}  netteté={sharpness(rec, xe[:4]):.2f}")
        out = f"/content/slots_orient_{a.shape}_{a.H}.png" if os.path.isdir("/content") else f"slots_orient_{a.shape}_{a.H}.png"
        plt.tight_layout(); plt.savefig(out); print(f"\nfigure -> {out}", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:70], flush=True)

if __name__ == "__main__":
    main()
