"""
MONDE D'OBJETS — world model LeJEPA qui comprend les OBJETS et leur DYNAMIQUE.

Étape 1 (compréhension, ce fichier) : N objets colorés bougent dans une boîte, rebondissent
sur les murs et SE PERCUTENT (collision élastique, masse égale). Le world model V-JEPA
(LeJEPA / SIGReg, sans EMA ni étiquettes) regarde ces vidéos, puis on MESURE ce qu'il a compris :

  - SONDE ÉTAT     : peut-on lire position + vitesse de chaque objet dans le latent ?
                     (couleur = identité, donc "où est le rouge / où va-t-il" est bien défini)
  - ANTICIPATION   : depuis la 1re moitié, prédit-il une COLLISION objet-objet à venir ?
                     vs hasard, vs ORACLE supervisé (= "l'info est-elle là ?").

La vérité-terrain (positions, vitesses, collisions) ne sert QU'À mesurer, jamais à entraîner le SSL.

  python objects.py --n 2500 --T 12 --H 48 --n_obj 3 --ssl_steps 3000
"""
import argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vjepa import patchify, VJEPA, tube_masks, temporal_mask

COLORS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]], np.float32)

# ------------------------------------------------------------------ environnement (numpy pur)
def gen_clips(n, T, n_obj, H, r=0.09, dt=0.045, seed=0):
    """Renvoie frames (n,T,H,H,3), states (n,T,n_obj,4)=[x,y,vx,vy], coll (n,T) = collision ce pas."""
    rng = np.random.default_rng(seed)
    cols = COLORS[:n_obj]
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H            # grille pour le rendu
    frames = np.zeros((n, T, H, H, 3), np.float32)
    states = np.zeros((n, T, n_obj, 4), np.float32)
    coll = np.zeros((n, T), np.float32)
    for i in range(n):
        pos = np.zeros((n_obj, 2), np.float32)                    # init sans chevauchement
        for k in range(n_obj):
            for _ in range(200):
                p = rng.uniform(r, 1 - r, 2)
                if k == 0 or np.all(np.linalg.norm(pos[:k] - p, axis=1) > 2.2 * r): pos[k] = p; break
        ang = rng.uniform(0, 2 * np.pi, n_obj); spd = rng.uniform(1.4, 2.8, n_obj)
        vel = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32) * spd[:, None]
        for t in range(T):
            states[i, t, :, :2] = pos; states[i, t, :, 2:] = vel
            img = np.zeros((H, H, 3), np.float32)
            for k in range(n_obj):
                m = (xx - pos[k, 0]) ** 2 + (yy - pos[k, 1]) ** 2 < r * r
                img[m] = cols[k]
            frames[i, t] = img
            pos = pos + vel * dt                                  # avance
            for dim in (0, 1):                                    # rebond sur les murs
                lo = pos[:, dim] < r; vel[lo, dim] = np.abs(vel[lo, dim]); pos[lo, dim] = r
                hi = pos[:, dim] > 1 - r; vel[hi, dim] = -np.abs(vel[hi, dim]); pos[hi, dim] = 1 - r
            hit = False                                           # collisions élastiques paire à paire
            for a in range(n_obj):
                for b in range(a + 1, n_obj):
                    dv = pos[a] - pos[b]; dist = np.linalg.norm(dv)
                    if 1e-6 < dist < 2 * r:
                        nrm = dv / dist; rel = np.dot(vel[a] - vel[b], nrm)
                        if rel < 0:                               # ils se rapprochent
                            vel[a] -= rel * nrm; vel[b] += rel * nrm          # masse égale : échange composante normale
                            ov = 2 * r - dist; pos[a] += nrm * ov / 2; pos[b] -= nrm * ov / 2
                            hit = True
            coll[i, t] = float(hit)
    return frames, states, coll

# ------------------------------------------------------------------ sondes (encodeur GELÉ)
def per_frame_latent(m, Xt, T, npf, dev, bs=16):
    """(.,ntok,obs) -> descripteur PAR FRAME (.,T,2d) = moyenne+max sur les patches d'une frame.
    Le max garde l'objet LOCALISÉ (la position) au lieu de la diluer dans la moyenne."""
    out = []
    for i in range(0, len(Xt), bs):
        z = m.tokens(Xt[i:i+bs].to(dev))                         # (b,ntok,d)
        zf = z.reshape(z.size(0), T, npf, z.size(-1))            # (b,T,npf,d)
        out.append(torch.cat([zf.mean(2), zf.amax(2)], -1).cpu())  # (b,T,2d)
    return torch.cat(out)

def r2(pred, tgt):
    ss_res = (pred - tgt).pow(2).sum(0); ss_tot = (tgt - tgt.mean(0)).pow(2).sum(0) + 1e-9
    return (1 - ss_res / ss_tot).mean().item()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2500); p.add_argument("--T", type=int, default=12)
    p.add_argument("--H", type=int, default=48); p.add_argument("--patch", type=int, default=8)
    p.add_argument("--n_obj", type=int, default=3)
    p.add_argument("--d_model", type=int, default=256); p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--reg_w", type=float, default=1.0); p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--n_mask", type=int, default=2)
    p.add_argument("--ssl_steps", type=int, default=3000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    nP = a.H // a.patch; npf = nP * nP; ntok = a.T * npf; d = a.d_model
    print(f"device={dev}  monde {a.n_obj} objets  H={a.H} -> grille {nP}x{nP}, {ntok} tokens/clip", flush=True)
    F0, S, C = gen_clips(a.n, a.T, a.n_obj, a.H, seed=0)
    Xp = patchify(F0, a.patch); obs = Xp.shape[2]
    Xt = torch.tensor(Xp); g = torch.Generator().manual_seed(1); pm = torch.randperm(a.n, generator=g)
    nt = int(0.8 * a.n); tr, te = pm[:nt].numpy(), pm[nt:].numpy()
    print(f"{a.n} clips | collisions dans {int((C.sum(1) > 0).sum())} clips", flush=True)

    # ---- LeJEPA SSL (tubelets spatiaux + masquage temporel), SANS étiquettes ----
    torch.manual_seed(0); m = VJEPA(obs, d, ntok, a.n_layer, a.n_head, a.reg_w, a.pred_layers).to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr); rng = np.random.default_rng(0)
    for st in range(a.ssl_steps):
        bi = tr[np.random.randint(0, len(tr), a.bs)]; o = Xt[bi].to(dev)
        if np.random.rand() < 0.5:
            masks = [mk.to(dev) for mk in tube_masks(a.bs, a.T, nP, a.mask_ratio, a.n_mask, rng)]
        else:
            t0 = np.random.randint(2, a.T - 1); masks = [temporal_mask(a.bs, a.T, nP, t0, dev)]
        loss = m(o, masks); opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0: print(f"  [SSL] step {st} loss {loss.item():.3f}", flush=True)
    for prm in m.parameters(): prm.requires_grad = False

    # ---- SONDE 1 : état (position + vitesse de chaque objet) depuis le latent par frame ----
    Z = per_frame_latent(m, Xt, a.T, npf, dev)                   # (n,T,2d)
    fd = Z.size(-1)
    St = torch.tensor(S.reshape(a.n, a.T, -1))                   # (n,T,n_obj*4)
    def flat(idx): return Z[idx].reshape(-1, fd).to(dev), St[idx].reshape(-1, a.n_obj * 4).to(dev)
    Ztr, Ytr = flat(tr); Zte, Yte = flat(te)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
    head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Linear(256, a.n_obj * 4)).to(dev)
    oh = torch.optim.Adam(head.parameters(), 1e-3)
    for _ in range(800):
        oh.zero_grad(); F.mse_loss(head(Ztr), (Ytr - mu) / sd).backward(); oh.step()
    with torch.no_grad(): pr = head(Zte) * sd + mu
    pos_r2 = r2(pr.reshape(-1, a.n_obj, 4)[..., :2].reshape(len(pr), -1), Yte.reshape(-1, a.n_obj, 4)[..., :2].reshape(len(pr), -1))
    vel_r2 = r2(pr.reshape(-1, a.n_obj, 4)[..., 2:].reshape(len(pr), -1), Yte.reshape(-1, a.n_obj, 4)[..., 2:].reshape(len(pr), -1))

    # ---- SONDE 2 : anticipation de collision (1re moitié -> collision dans la 2e ?) ----
    t0 = a.T // 2
    lab = torch.tensor((C[:, t0+1:].sum(1) > 0).astype(np.int64))             # collision après t0 ?
    @torch.no_grad()
    def ctx_fut_feat(idx):                                       # contexte observé + futur imaginé
        out = []
        for i in range(0, len(idx), 16):
            o = Xt[idx[i:i+16]].to(dev); msk = temporal_mask(o.size(0), a.T, nP, t0, dev)
            ctx, fut = m.predict_targets(o, msk); out.append(torch.cat([ctx, fut.mean(1)], -1).cpu())
        return torch.cat(out)
    def clf_acc(Ftr, ytr, Fte, yte):
        clf = nn.Sequential(nn.Linear(Ftr.size(1), 128), nn.GELU(), nn.Linear(128, 2)).to(dev)
        op = torch.optim.Adam(clf.parameters(), 1e-3); n1 = int(ytr.sum())
        w = torch.tensor([len(ytr)/(2*(len(ytr)-n1)+1), len(ytr)/(2*n1+1)], device=dev)  # classes équilibrées
        for _ in range(800):
            op.zero_grad(); F.cross_entropy(clf(Ftr.to(dev)), ytr.to(dev), weight=w).backward(); op.step()
        with torch.no_grad(): return (clf(Fte.to(dev)).argmax(-1).cpu() == yte).float().mean().item()
    acc_wm = clf_acc(ctx_fut_feat(tr), lab[tr], ctx_fut_feat(te), lab[te])
    # ORACLE supervisé : mêmes frames 0..t0 brutes -> "l'info est-elle là ?"
    raw = torch.tensor(F0[:, :t0+1].reshape(a.n, -1))
    acc_oracle = clf_acc(raw[tr], lab[tr], raw[te], lab[te])

    print(f"\n=== COMPRÉHENSION DU MONDE D'OBJETS ===", flush=True)
    print(f"  SONDE ÉTAT (R², 1.0=parfait) : position={pos_r2:.2f}  vitesse={vel_r2:.2f}", flush=True)
    print(f"  ANTICIPATION COLLISION (acc) : world model={acc_wm:.2f}  | oracle={acc_oracle:.2f}  | hasard={1-lab.float().mean():.2f}", flush=True)

if __name__ == "__main__":
    main()
