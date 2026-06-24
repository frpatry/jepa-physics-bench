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
    p.add_argument("--n_obj", type=int, default=3); p.add_argument("--dt", type=float, default=0.045)
    p.add_argument("--pred_k", type=int, default=2, help="horizon de prédiction (frames en avant) : court = apprenable")
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
    F0, S, C = gen_clips(a.n, a.T, a.n_obj, a.H, dt=a.dt, seed=0)
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
            t0 = a.T - 1 - np.random.randint(1, a.pred_k + 1)    # masque seulement les pred_k dernières (horizon court)
            masks = [temporal_mask(a.bs, a.T, nP, t0, dev)]
        loss = m(o, masks); opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0: print(f"  [SSL] step {st} loss {loss.item():.3f}", flush=True)
    for prm in m.parameters(): prm.requires_grad = False

    # ---- décodeur de POSITION (sonde gelée) : latent par frame -> positions des objets ----
    Z = per_frame_latent(m, Xt, a.T, npf, dev)                   # (n,T,2d) latents propres
    fd = Z.size(-1)
    P = torch.tensor(S[..., :2].reshape(a.n, a.T, -1)).float()   # (n,T,n_obj*2) positions vraies
    Vv = torch.tensor(S[..., 2:].reshape(a.n, a.T, -1)).float()  # (n,T,n_obj*2) vitesses vraies
    Ptr = P[tr].reshape(-1, a.n_obj * 2).to(dev); pmu, psd = Ptr.mean(0), Ptr.std(0) + 1e-6
    def train_dec(Zf, Pf):                                       # entraîne un décodeur latent->position
        d2 = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Linear(256, a.n_obj * 2)).to(dev)
        op = torch.optim.Adam(d2.parameters(), 1e-3)
        for _ in range(1000):
            op.zero_grad(); F.mse_loss(d2(Zf), (Pf - pmu) / psd).backward(); op.step()
        return lambda z: d2(z.to(dev)) * psd + pmu
    decode = train_dec(Z[tr].reshape(-1, fd).to(dev), Ptr)       # décodeur appris sur latents PROPRES
    pos_now = r2(decode(Z[te].reshape(-1, fd)), P[te].reshape(-1, a.n_obj * 2).to(dev))   # sait-il OÙ (frames vues)

    # ---- PRÉDICTION DU FUTUR : comprend-il la DYNAMIQUE ? (prédire les pred_k frames suivantes) ----
    t0 = a.T - 1 - a.pred_k; nf = a.pred_k                       # contexte 0..t0, prédire t0+1..T-1 (horizon court)
    @torch.no_grad()
    def imagine(idx):                                           # descripteurs futurs IMAGINÉS (.,nf,2d)
        out = []
        for i in range(0, len(idx), 16):
            o = Xt[idx[i:i+16]].to(dev); msk = temporal_mask(o.size(0), a.T, nP, t0, dev)
            _, fut = m.predict_targets(o, msk)                  # (b, nf*npf, d)
            zf = fut.reshape(fut.size(0), nf, npf, fut.size(-1))
            out.append(torch.cat([zf.mean(2), zf.amax(2)], -1).cpu())   # (b,nf,2d)
        return torch.cat(out)
    # comparaison juste (même décodeur "propre") : imaginer vs figer la derniere vue vs plafond
    gt = P[te][:, t0+1:].to(dev).reshape(-1, a.n_obj * 2)        # positions futures vraies
    Zi_te = imagine(te)                                          # futurs IMAGINÉS (test)
    r2_wm = r2(decode(Zi_te.reshape(-1, fd)), gt)              # futur imaginé, décodeur propre
    frozen = Z[te][:, t0:t0+1].expand(-1, nf, -1).reshape(-1, fd)  # on FIGE le dernier latent observé
    r2_frozen = r2(decode(frozen), gt)                          # = "rien ne bouge"
    r2_ceil = r2(decode(Z[te][:, t0+1:].reshape(-1, fd)), gt)   # plafond : décodage du futur RÉEL (limite décodeur)
    # DIAGNOSTIC : ré-apprendre le décodeur SUR les latents imaginés -> le futur imaginé est-il LISIBLE ?
    dec_i = train_dec(imagine(tr).reshape(-1, fd).to(dev), P[tr][:, t0+1:].reshape(-1, a.n_obj * 2).to(dev))
    r2_wm_relu = r2(dec_i(Zi_te.reshape(-1, fd)), gt)
    # repere physique (etat VRAI) : ce qu'une extrapolation ligne droite atteindrait
    p0 = P[te][:, t0:t0+1].to(dev); v0 = Vv[te][:, t0:t0+1].to(dev)
    ks = torch.arange(1, nf + 1).view(1, nf, 1).float().to(dev)
    r2_constv = r2((p0 + v0 * a.dt * ks).reshape(-1, a.n_obj * 2), gt)

    print(f"\n=== COMPRÉHENSION DU MONDE D'OBJETS ===", flush=True)
    print(f"  SAIT OÙ sont les objets (R² position, frames vues) : {pos_now:.2f}", flush=True)
    print(f"  PRÉDIT LE FUTUR : imaginé={r2_wm:.2f}  vs  figé(rien ne bouge)={r2_frozen:.2f}  | plafond(futur réel)={r2_ceil:.2f}", flush=True)
    print(f"  DIAGNOSTIC futur imaginé LISIBLE (décodeur ré-appris dessus) : {r2_wm_relu:.2f}", flush=True)
    print(f"  (repère physique, état vrai : ligne droite atteindrait {r2_constv:.2f})", flush=True)

if __name__ == "__main__":
    main()
