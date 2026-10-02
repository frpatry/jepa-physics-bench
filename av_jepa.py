"""
AV-JEPA — le SON enrichit-il le latent d'un JEPA ?  (étape 1 de l'avenue « bébé » : vision + ouïe)

Monde jouet : 2 disques dans une boîte, chocs élastiques AVEC MASSES INÉGALES, et deux propriétés
CACHÉES par disque, indépendantes de sa couleur :
  - MASSE (log-uniforme 1/3..3)  -> la vision peut la déduire en partie (ratio, via la dynamique
                                     d'un choc disque-disque) ; le son la porte directement
                                     (amplitude ∝ impulsion, hauteur ∝ masse^-1/3 si --pitch_mass 1).
  - MATÉRIAU (bois/métal/verre)  -> AUCUNE trace visuelle ; seul le son le porte (timbre, durée de
                                     résonance). = CONTRÔLE NÉGATIF : en vision seule, la sonde doit
                                     rester au hasard (33 %) — sinon il y a une fuite.
Son = synthèse MODALE d'impact (somme de sinusoïdes amorties), stéréo panoramique selon x du choc,
chocs disque-disque ET murs, bruit de fond ; une fenêtre audio par frame -> spectrogramme log
(32 bandes × 2 canaux) = 1 token audio par frame. --occl w : bande grise opaque au centre (on
entend les chocs qu'on ne voit pas).

Modèle : JEPA multimodal (encodeur-contexte sur tokens visibles, prédicteur attentionnel, cibles =
latents de l'encodeur sur le clip propre, anti-collapse SIGReg officiel — recette vjepa.py).
Masques : tube vision, bloc temporel vision, futur, et pour le bras VA : prédire le son depuis
l'image (et un bloc audio depuis le reste). Dropout de modalité : l'encodeur VA sait aussi
tourner en VISION SEULE.

Bras (même données, mêmes pas, même archi) :
  V  : ne voit jamais le son.
  VA : vision + son à l'entraînement.
Éval (encodeur GELÉ, sondes attentives) dans plusieurs conditions d'entrée :
  V|vision, VA|vision (le son comme PROFESSEUR : entraîné avec, testé sans), VA|vision+son,
  VA|son seul, + encodeurs NON entraînés (ce que la sonde tire des entrées brutes).
Sondes : matériau (acc), log-masse (R² tout / avec choc), ratio de masses (R² avec choc =
ce que la vision peut en principe déduire), impact par frame (exactitude équilibrée).

  python av_jepa.py --n 8000 --steps 6000            # Colab GPU
  python av_jepa.py --n 300 --n_probe 200 --steps 30 --probe_steps 30 --d 32   # smoke local
"""
import argparse, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vjepa import sigreg, patchify, tube_masks, Predictor, AttentiveProbe, _idx, _gather

PAL = np.array([[1, .2, .2], [.2, 1, .2], [.3, .4, 1], [1, 1, .2], [1, .3, 1], [.2, 1, 1]], np.float32)
MATS = ("bois", "métal", "verre")
F_MAT = np.array([180., 520., 1150.])                                   # fondamentale par matériau
PARTIALS = [np.array([1., 2.7, 5.2]), np.array([1., 2.76, 5.40, 8.93]), np.array([1., 1.6, 2.9])]
TAU = np.array([0.02, 0.45, 0.12])                                      # résonance (s) : sec / long / moyen
SR, SPF, NB = 8000, 512, 32                                             # 64 ms d'audio par frame

def band_matrix():
    """pooling rfft(512) -> NB bandes log-espacées 100..4000 Hz (bande vide -> bin le plus proche)."""
    fr = np.fft.rfftfreq(SPF, 1 / SR); e = np.geomspace(100, SR / 2, NB + 1)
    W = np.zeros((NB, len(fr)), np.float32)
    for b in range(NB):
        on = (fr >= e[b]) & (fr < e[b + 1])
        if on.any(): W[b, on] = 1 / on.sum()
        else: W[b, np.argmin(abs(fr - math.sqrt(e[b] * e[b + 1])))] = 1
    return W

def gen_world(n, T=8, H=32, r=0.12, seed=0, pitch_mass=1, occl=0.0, p_aim=0.6, smin=0.05, smax=0.11):
    rng = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:H, 0:H].astype(np.float32) + 0.5) / H
    X = np.zeros((n, T, H, H, 3), np.float32); A = np.zeros((n, T, 2, NB), np.float32)
    MAT = rng.integers(0, 3, (n, 2)); LM = rng.uniform(np.log(1 / 3), np.log(3), (n, 2)).astype(np.float32)
    IMP = np.zeros((n, T), bool); HIT = np.zeros(n, bool)
    win = np.hanning(SPF).astype(np.float32); W = band_matrix(); tt_all = np.arange(T * SPF) / SR
    for i in range(n):
        cols = PAL[np.sort(rng.choice(len(PAL), 2, replace=False))]    # objet 0 = plus petit indice palette
        m = np.exp(LM[i]); P = np.zeros((2, 2), np.float32)
        P[0] = rng.uniform(r, 1 - r, 2)
        for _ in range(100):
            P[1] = rng.uniform(r, 1 - r, 2)
            if np.linalg.norm(P[1] - P[0]) > 2.5 * r: break
        sp = rng.uniform(smin, smax, 2)
        if rng.random() < p_aim:                                        # visent l'un vers l'autre -> chocs fréquents
            d = P[1] - P[0]; a0 = math.atan2(d[1], d[0])
            th = np.array([a0, a0 + math.pi]) + rng.normal(0, 0.35, 2)
        else: th = rng.uniform(0, 2 * math.pi, 2)
        V = np.stack([sp * np.cos(th), sp * np.sin(th)], -1).astype(np.float32)
        ev = []                                                         # (t, objet, impulsion, x du choc)
        for t in range(T):
            if t > 0:
                P = P + V
                for k in range(2):
                    for dd in range(2):
                        if P[k, dd] < r or P[k, dd] > 1 - r:
                            J = 2 * m[k] * abs(V[k, dd]); V[k, dd] = -V[k, dd]
                            P[k, dd] = 2 * r - P[k, dd] if P[k, dd] < r else 2 * (1 - r) - P[k, dd]
                            ev.append((t, k, J, P[k, 0]))
                dv = P[0] - P[1]; dist = float(np.linalg.norm(dv))
                if 1e-6 < dist < 2 * r:
                    nv = dv / dist; s_ = float((V[0] - V[1]) @ nv)
                    if s_ < 0:
                        J = -2 * s_ * m[0] * m[1] / (m[0] + m[1])
                        V[0] += J / m[0] * nv; V[1] -= J / m[1] * nv; HIT[i] = True
                        xc = float((P[0, 0] + P[1, 0]) / 2); ev += [(t, 0, J, xc), (t, 1, J, xc)]
                    push = (2 * r - dist) / 2
                    P[0] = np.clip(P[0] + push * nv, r, 1 - r); P[1] = np.clip(P[1] - push * nv, r, 1 - r)
            img = np.zeros((H, H, 3), np.float32)
            for k in range(2):
                al = np.clip((r - np.sqrt((xx - P[k, 0]) ** 2 + (yy - P[k, 1]) ** 2)) * H + 0.5, 0, 1)[..., None]
                img = img * (1 - al) + cols[k] * al
            if occl > 0: img[:, (xx[0] > 0.5 - occl / 2) & (xx[0] < 0.5 + occl / 2)] = 0.5
            X[i, t] = img
        sig = rng.normal(0, 0.005, (2, T * SPF)).astype(np.float32)    # bruit de fond
        for (t, k, J, xc) in ev:
            IMP[i, t] = True
            on = t * SPF + int(rng.integers(0, SPF // 2)); tt = tt_all[:T * SPF - on]
            f0 = F_MAT[MAT[i, k]] * (m[k] ** (-1 / 3) if pitch_mass else 1.0); amp = min(J / 0.15, 4.0)
            s = np.zeros_like(tt)
            for j, rr in enumerate(PARTIALS[MAT[i, k]]):
                if f0 * rr < 0.95 * SR / 2:
                    s += np.exp(-tt / (TAU[MAT[i, k]] / (1 + j))) * np.sin(2 * math.pi * f0 * rr * tt + rng.uniform(0, 6.28)) / (1 + j)
            xc = float(np.clip(xc, 0, 1))
            sig[0, on:] += amp * math.sqrt(1 - xc) * s; sig[1, on:] += amp * math.sqrt(xc) * s
        fr = sig.reshape(2, T, SPF) * win
        A[i] = np.log1p(np.einsum("bf,ctf->tcb", W, np.abs(np.fft.rfft(fr, axis=-1)) ** 2))
    return dict(X=X, A=A, MAT=MAT, LM=LM, IMP=IMP, HIT=HIT)

# ---------------------------------------------------------------- tokens : vision (patches) + audio (1/frame)
def make_tokens(w, P, amu=None, asd=None):
    """-> (n, T*npf + T, P*P*3) fp16 ; l'audio (2*NB dims) est zéro-paddé dans la même largeur."""
    Tv = patchify(w["X"], P); n, T = w["A"].shape[:2]
    a = w["A"].reshape(n, T, -1)
    if amu is None: amu, asd = a.mean((0, 1)), a.std((0, 1)) + 1e-4
    a = (a - amu) / asd
    pad = np.zeros((n, T, Tv.shape[-1]), np.float32); pad[..., :a.shape[-1]] = a
    return torch.from_numpy(np.concatenate([Tv, pad], 1).astype(np.float16)), amu, asd

class AVEncoder(nn.Module):
    """ViT sur un sous-ensemble de tokens ; la modalité est déduite de l'indice (>= nv -> audio)."""
    def __init__(s, dv, da, d, ntok, nv, nl, nh):
        super().__init__()
        s.ev, s.ea = nn.Linear(dv, d), nn.Linear(da, d); s.mod = nn.Embedding(2, d); s.pos = nn.Embedding(ntok, d)
        s.nv, s.da = nv, da
        layer = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d)
    def forward(s, tok, idx):
        isa = idx >= s.nv
        e = torch.where(isa.unsqueeze(-1), s.ea(tok[..., :s.da]), s.ev(tok))
        return s.ln(s.tr(e + s.mod(isa.long()) + s.pos(idx)))

class AVJEPA(nn.Module):
    def __init__(s, dv, da, nv, T, d, nl, nh, pred_layers):
        super().__init__()
        s.nv, s.T, s.ntok = nv, T, nv + T
        s.enc = AVEncoder(dv, da, d, s.ntok, nv, nl, nh); s.pred = Predictor(d, s.ntok, pred_layers, nh)
    def loss(s, o, present, pairs):
        """present:(B,N) tokens du passage-cible propre ; pairs = [(ctx_mask, tgt_mask)] (B,N) bool."""
        B, N, _ = o.shape; pidx = _idx(present)
        z = s.enc(_gather(o, pidx), pidx)
        zf = torch.zeros(B, N, z.size(-1), device=o.device, dtype=z.dtype)
        zf = zf.scatter(1, pidx.unsqueeze(-1).expand(-1, -1, z.size(-1)), z)
        pl = 0.0
        for cm, tm in pairs:
            cidx, tidx = _idx(cm), _idx(tm)
            pl = pl + F.smooth_l1_loss(s.pred(s.enc(_gather(o, cidx), cidx), cidx, tidx), _gather(zf, tidx))
        return pl / len(pairs), sigreg(z.reshape(-1, z.size(-1)))

def sample_pairs(arm, B, T, nP, nv, rng, n_masks, p_drop=0.3):
    """masques de pré-entraînement. vision = tokens [0,nv) frame-major ; audio = [nv, nv+T)."""
    npf = nP * nP; N = nv + T
    isv = np.zeros(N, bool); isv[:nv] = True; isa = ~isv
    frame = np.concatenate([np.arange(nv) // npf, np.arange(T)])
    present = isv | (isa if arm == "va" else False)
    strat = ["tube", "vblock", "future"] + (["a_from_v", "ablock"] if arm == "va" else [])
    pairs = []
    for _ in range(n_masks):
        st = strat[rng.integers(len(strat))]
        tg = np.zeros((B, N), bool)
        if st == "tube":
            tg[:, :nv] = tube_masks(B, T, nP, 0.5, 1, rng)[0].numpy()
        elif st in ("vblock", "ablock"):
            L = 3
            for b in range(B):
                t0 = rng.integers(0, T - L + 1); blk = (frame >= t0) & (frame < t0 + L)
                tg[b] = blk & (isv if st == "vblock" else isa)
        elif st == "future":
            t0 = rng.integers(2, T - 1); tg[:] = present & (frame > t0)
            cm = np.broadcast_to(present & (frame <= t0), (B, N)).copy(); pairs.append((cm, tg)); continue
        elif st == "a_from_v":
            tg[:] = isa; pairs.append((np.broadcast_to(isv, (B, N)).copy(), tg)); continue
        cm = present & ~tg
        if arm == "va" and st != "ablock" and rng.random() < p_drop: cm &= isv  # dropout de modalité
        pairs.append((cm, tg))
    return present, pairs

class SupModel(nn.Module):
    """PLAFOND SUPERVISÉ : même encodeur, entraîné bout-à-bout sur les étiquettes cachées
    (masse, impact ; + matériau si le son est en entrée). Dit si l'info est LISIBLE dans ces entrées
    à cette résolution — indépendamment de l'objectif JEPA."""
    def __init__(s, dv, da, nv, T, d, nl, nh):
        super().__init__()
        s.nv, s.T = nv, T
        s.enc = AVEncoder(dv, da, d, nv + T, nv, nl, nh)
        s.h_mat, s.h_lm, s.h_imp = AttentiveProbe(d, 6), AttentiveProbe(d, 3), AttentiveProbe(d, 2)

def train_sup(arm, tok, w, a, dev, nv, T):
    torch.manual_seed(a.seed)
    m = SupModel(tok.size(-1), 2 * NB, nv, T, a.d, a.nl, a.nh).to(dev)
    keep = np.zeros(nv + T, bool); keep[:nv] = True
    if arm == "sup_va": keep[nv:] = True
    idx = torch.from_numpy(np.where(keep)[0]).to(dev)
    lm = torch.from_numpy(w["LM"]); y = torch.stack([lm[:, 0], lm[:, 1], lm[:, 0] - lm[:, 1]], 1)
    y = ((y - y.mean(0)) / y.std(0)).to(dev); mat = torch.from_numpy(w["MAT"]).long().to(dev)
    imp = torch.from_numpy(w["IMP"][:, 1:]).long().to(dev)
    fpos = imp.float().mean().clamp(0.01, 0.99); cw = torch.stack([1 / (1 - fpos), 1 / fpos])
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05); t0 = time.time()
    for it in range(1, a.steps + 1):
        bi = torch.randint(0, len(tok), (a.bs,)); o = tok[bi].to(dev).float(); ix = idx.expand(a.bs, -1)
        Z = m.enc(_gather(o, ix), ix); bd = bi.to(dev)
        l_lm = F.mse_loss(m.h_lm(Z), y[bd])
        l_imp = F.cross_entropy(m.h_imp(frame_tokens(Z, keep, nv, T).flatten(0, 1)), imp[bd].flatten(), weight=cw)
        l_mat = F.cross_entropy(m.h_mat(Z).view(-1, 3), mat[bd].view(-1)) if arm == "sup_va" else torch.zeros((), device=dev)
        loss = l_lm + l_imp + l_mat
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if it % max(1, a.steps // 10) == 0 or it == 1:
            print(f"  [{arm}] step {it:5d}  masse {l_lm.item():.3f}  impact {l_imp.item():.3f}  matériau {l_mat.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return m.eval()

def pretrain(arm, tok, a, dev, nv, T, nP, rng):
    torch.manual_seed(a.seed)
    m = AVJEPA(tok.size(-1), 2 * NB, nv, T, a.d, a.nl, a.nh, a.pred_layers).to(dev)
    if a.steps == 0: return m.eval()
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05); t0 = time.time()
    for it in range(1, a.steps + 1):
        bi = torch.randint(0, len(tok), (a.bs,))
        o = tok[bi].to(dev).float()
        present, pairs = sample_pairs(arm, a.bs, T, nP, nv, rng, a.n_masks)
        tt = lambda x: torch.from_numpy(x).to(dev)
        pl, sr = m.loss(o, tt(np.broadcast_to(present, (a.bs, len(present))).copy()), [(tt(c), tt(g)) for c, g in pairs])
        loss = pl + a.rw * sr
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if it % max(1, a.steps // 10) == 0 or it == 1:
            print(f"  [{arm}] step {it:5d}  pred {pl.item():.4f}  sigreg {sr.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return m.eval()

# ---------------------------------------------------------------- évaluation : sondes attentives sur encodeur gelé
@torch.no_grad()
def encode(m, tok, cond, dev, bs=256):
    nv, T = m.nv, m.T; N = nv + T
    keep = np.zeros(N, bool)
    if "v" in cond: keep[:nv] = True
    if "a" in cond: keep[nv:] = True
    idx = torch.from_numpy(np.where(keep)[0]).to(dev)
    out = []
    for i in range(0, len(tok), bs):
        o = tok[i:i + bs].to(dev).float(); ix = idx.expand(len(o), -1)
        out.append(m.enc(_gather(o, ix), ix).half().cpu())
    return torch.cat(out), keep

def frame_tokens(Z, keep, nv, T):
    """Z:(n,k,d) latents des tokens gardés -> (n,T-1,kf,d) : tokens de la frame t (patches + audio t), t=1..T-1."""
    kept = np.where(keep)[0]; npf = nv // T
    frame = np.where(kept < nv, kept // npf, kept - nv)
    return torch.stack([Z[:, torch.from_numpy(np.where(frame == t)[0]).to(Z.device)] for t in range(1, T)], 1)

def fit_probe(Ttr, ytr, Tte, nout, task, dev, steps, bs=128, lr=1e-3):
    pr = AttentiveProbe(Ttr.size(-1), nout).to(dev); opt = torch.optim.AdamW(pr.parameters(), lr, weight_decay=1e-2)
    if task == "imp":                                                   # classes déséquilibrées -> poids inverses
        fpos = ytr.float().mean().clamp(0.01, 0.99); cw = torch.stack([1 / (1 - fpos), 1 / fpos]).to(dev)
    for _ in range(steps):
        bi = torch.randint(0, len(Ttr), (bs,)); out = pr(Ttr[bi].to(dev).float()); y = ytr[bi].to(dev)
        if task == "mat": l = F.cross_entropy(out.view(-1, 3), y.view(-1))
        elif task == "imp": l = F.cross_entropy(out, y, weight=cw)
        else: l = F.mse_loss(out, y)
        opt.zero_grad(); l.backward(); opt.step()
    pr.eval()
    with torch.no_grad(): return torch.cat([pr(Tte[i:i + 512].to(dev).float()).cpu() for i in range(0, len(Tte), 512)])

def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-8))

def evaluate(m, tok, w, cond, dev, steps, ntr):
    Z, keep = encode(m, tok, cond, dev); n = len(Z); tr, te = slice(0, ntr), slice(ntr, n)
    res = {}
    mat = torch.from_numpy(w["MAT"]).long()
    p = fit_probe(Z[tr], mat[tr], Z[te], 6, "mat", dev, steps)
    res["mat"] = float((p.view(-1, 2, 3).argmax(-1) == mat[te]).float().mean())
    lm = torch.from_numpy(w["LM"]); y = torch.stack([lm[:, 0], lm[:, 1], lm[:, 0] - lm[:, 1]], 1)
    mu, sd = y[tr].mean(0), y[tr].std(0); p = fit_probe(Z[tr], (y[tr] - mu) / sd, Z[te], 3, "reg", dev, steps) * sd + mu
    hit = torch.from_numpy(w["HIT"])[te]
    res["lm_all"] = (r2(p[:, 0], y[te, 0]) + r2(p[:, 1], y[te, 1])) / 2
    res["lm_hit"] = (r2(p[hit, 0], y[te][hit, 0]) + r2(p[hit, 1], y[te][hit, 1])) / 2
    res["ratio_hit"] = r2(p[hit, 2], y[te][hit, 2])
    # impact par frame : tokens de la frame t (patches de t + token audio t si présent), frames 1..T-1
    Zf = frame_tokens(Z, keep, m.nv, m.T)                                # (n,T-1,kf,d)
    imp = torch.from_numpy(w["IMP"][:, 1:]).long()
    Ztr, Zte = Zf[tr].flatten(0, 1), Zf[te].flatten(0, 1); ytr, yte = imp[tr].flatten(), imp[te].flatten()
    p = fit_probe(Ztr, ytr, Zte, 2, "imp", dev, steps).argmax(-1)
    res["imp_bacc"] = float(((p[yte == 1] == 1).float().mean() + (p[yte == 0] == 0).float().mean()) / 2)
    return res

def fig_world(w, path, k=2):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = w["X"].shape[1]; fig, ax = plt.subplots(2 * k, T, figsize=(1.6 * T, 3.4 * k))
    for e in range(k):
        i = int(np.where(w["HIT"])[0][e]) if w["HIT"].any() else e
        for t in range(T):
            ax[2 * e, t].imshow(w["X"][i, t]); ax[2 * e, t].axis("off")
            ax[2 * e, t].set_title(("IMPACT " if w["IMP"][i, t] else "") + f"t{t}", fontsize=8)
            ax[2 * e + 1, t].imshow(w["A"][i, t], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=w["A"].max()); ax[2 * e + 1, t].axis("off")
        ax[2 * e, 0].text(-0.1, 0.5, f"{MATS[w['MAT'][i, 0]]}/{MATS[w['MAT'][i, 1]]}\nm={np.exp(w['LM'][i]).round(2)}",
                          transform=ax[2 * e, 0].transAxes, ha="right", va="center", fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=110); print(f"figure -> {path}")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000); p.add_argument("--n_probe", type=int, default=4000)
    p.add_argument("--T", type=int, default=8); p.add_argument("--H", type=int, default=32); p.add_argument("--P", type=int, default=8)
    p.add_argument("--r", type=float, default=0.12); p.add_argument("--occl", type=float, default=0.0)
    p.add_argument("--pitch_mass", type=int, default=1)
    p.add_argument("--steps", type=int, default=6000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--rw", type=float, default=1.0)
    p.add_argument("--n_masks", type=int, default=3)
    p.add_argument("--d", type=int, default=128); p.add_argument("--nl", type=int, default=4)
    p.add_argument("--nh", type=int, default=4); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--probe_steps", type=int, default=1500); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", type=str, default="v,va"); p.add_argument("--init_baseline", type=int, default=1)
    p.add_argument("--fig", type=str, default="av_world.png"); p.add_argument("--out", type=str, default="av_jepa_results.json")
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(a.seed)
    t0 = time.time()
    wtr = gen_world(a.n, a.T, a.H, a.r, a.seed, a.pitch_mass, a.occl)
    wpr = gen_world(a.n_probe, a.T, a.H, a.r, a.seed + 1000, a.pitch_mass, a.occl)
    print(f"monde : {a.n}+{a.n_probe} séquences en {time.time() - t0:.0f}s | choc disque-disque {wtr['HIT'].mean():.0%} "
          f"| frames avec impact {wtr['IMP'][:, 1:].mean():.0%} | occl {a.occl}", flush=True)
    if a.fig: fig_world(wpr, a.fig)
    tok, amu, asd = make_tokens(wtr, a.P); tokp, _, _ = make_tokens(wpr, a.P, amu, asd)
    nP = a.H // a.P; nv = a.T * nP * nP; ntr = int(0.75 * a.n_probe)
    rows = []
    def run_evals(m, name, conds):
        for c in conds:
            r = evaluate(m, tokp, wpr, c, dev, a.probe_steps, ntr); rows.append((name, c, r))
            print(f"  {name:>7s} | entrée {c:>2s} | matériau {r['mat']:.0%} | log-masse R² {r['lm_all']:+.2f} "
                  f"(avec choc {r['lm_hit']:+.2f}) | ratio masses R² {r['ratio_hit']:+.2f} | impact {r['imp_bacc']:.0%}", flush=True)
    if a.init_baseline:
        print("--- encodeur NON entraîné (ce que la sonde tire des entrées brutes)")
        steps = a.steps; a.steps = 0; run_evals(pretrain("va", tok, a, dev, nv, a.T, nP, rng), "init", ["v", "va", "a"]); a.steps = steps
    for arm in a.arms.split(","):
        if arm.startswith("sup"):
            print(f"--- PLAFOND SUPERVISÉ bras {arm.upper()} ({a.steps} pas, étiquettes du jeu d'entraînement)", flush=True)
            m = train_sup(arm, tok, wtr, a, dev, nv, a.T)
            run_evals(m, arm.upper(), ["v"] if arm == "sup_v" else ["va"])
            continue
        print(f"--- pré-entraînement JEPA bras {arm.upper()} ({a.steps} pas)", flush=True)
        m = pretrain(arm, tok, a, dev, nv, a.T, nP, rng)
        run_evals(m, arm.upper(), ["v"] if arm == "v" else ["v", "va", "a"])
    print("\n================ RÉSUMÉ (sondes sur encodeur gelé, jeu tenu à l'écart) ================")
    print(f"{'modèle':>7s} {'entrée':>6s} | {'matériau':>8s} | {'lmasse':>6s} | {'lm choc':>7s} | {'ratio choc':>10s} | {'impact':>6s}")
    for name, c, r in rows:
        print(f"{name:>7s} {c:>6s} | {r['mat']:8.0%} | {r['lm_all']:+6.2f} | {r['lm_hit']:+7.2f} | {r['ratio_hit']:+10.2f} | {r['imp_bacc']:6.0%}")
    get = {(n, c): r for n, c, r in rows}
    if ("V", "v") in get and ("VA", "v") in get:
        v, va = get[("V", "v")], get[("VA", "v")]
        print(f"\nCONTRÔLE NÉGATIF matériau en vision seule (hasard 33%) : V {v['mat']:.0%}, VA {va['mat']:.0%}")
        print(f"Q1 SON = PROFESSEUR (VA|vision vs V|vision) : ratio choc {va['ratio_hit']:+.2f} vs {v['ratio_hit']:+.2f} ; "
              f"impact {va['imp_bacc']:.0%} vs {v['imp_bacc']:.0%}")
    if ("SUP_V", "v") in get:
        sv = get[("SUP_V", "v")]
        print(f"PLAFOND SUPERVISÉ vision : log-masse {sv['lm_all']:+.2f}, ratio choc {sv['ratio_hit']:+.2f}, impact {sv['imp_bacc']:.0%} "
              f"-> si élevé : l'info EST lisible à cette résolution, c'est l'objectif JEPA qui ne la capte pas")
    if ("SUP_VA", "va") in get:
        s2 = get[("SUP_VA", "va")]
        print(f"PLAFOND SUPERVISÉ vision+son : matériau {s2['mat']:.0%}, ratio choc {s2['ratio_hit']:+.2f} (= liage son<->disque possible ?)")
    if ("VA", "va") in get:
        b = get[("VA", "va")]
        print(f"Q2 SON À L'INFÉRENCE (VA|vision+son) : matériau {b['mat']:.0%}, log-masse {b['lm_all']:+.2f}, impact {b['imp_bacc']:.0%}")
    if a.out:
        json.dump(dict(args=vars(a), rows=[dict(model=n, input=c, **r) for n, c, r in rows]), open(a.out, "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
