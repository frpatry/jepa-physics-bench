"""
PHASE 2 « bébé » — vision + ouïe + TOUCHER + ACTION (recette V-JEPA 2-AC, 100 % auto-supervisé).

Monde v3 : les 2 disques à masse/matériau cachés + une MAIN (carré blanc, masse 1) pilotée par
l'action (vitesse désirée). La main pousse les disques (choc à 2 corps : l'impulsion dépend de la masse
du disque) ; le TOUCHER = impulsion ressentie par la main (vecteur, norme) + RÉSISTANCE (impulsion /
vitesse d'approche ≈ masse effective : « c'est lourd à pousser ») ; le disque touché
sonne (son matériau). Politique « jeu de bébé » : la main va vers un disque, le pousse, change de cible.
Insight phase 1 : l'ACTION résout le LIAGE par construction — je sais quel disque je pousse, donc la
force ressentie et le son lui appartiennent.

Modèle : PRÉDICTEUR conditionné par l'action, bloc-causal (V-JEPA 2-AC) sur
  vision = latents V-JEPA 2 GELÉS (8 pas × 4×4), son (spectres), toucher, action.
Il prédit le PAS SUIVANT : latents V-JEPA 2 (cibles gelées -> pas de collapse), son, toucher.
Variantes (mêmes données/pas) : V+action | V+son+action | V+son+TOUCHER+action.

Évaluation SANS étiquette — erreur de prédiction des latents visuels du pas suivant vs COPIE :
  tous les pas | pas avec POUSSÉE | poussée d'un disque DÉJÀ touché (inférence de masse EN CONTEXTE :
  avoir senti le disque doit aider à prédire comment il réagira à la prochaine poussée).

  python av_act.py --n_train 6000 --n_test 2000      # Colab (encode V-JEPA 2 une fois, caches /content)
  python av_act.py --encoder dev --hum 0.15 --ctx 5   # PHASE 2 SUR NOTRE BÉBÉ : encodeur v4 GELÉ au meilleur moment
                                                     # (pas 20k, surprise 87 %), monde v3 + bourdonnement, pas = 1 frame
"""
import argparse, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import PAL, NB, SPF, SR, band_matrix, render_audio, hum_signal
from av_vjepa2 import encode_vjepa2

M_HAND, R_HAND = 1.0, 0.07
SWAP = [2, 3, 0, 1, 4, 5]                                               # échange disque 0 <-> disque 1 (main inchangée)

def gen_world_act(n, T=16, H=32, r=0.12, seed=0, a_sub=2, pitch_mass=1, vmax=0.1, damp=0.96, hum=0.0):
    rng = np.random.default_rng(seed); Ls = SPF // a_sub
    yy, xx = (np.mgrid[0:H, 0:H].astype(np.float32) + 0.5) / H
    X = np.zeros((n, T, H, H, 3), np.float32); A = np.zeros((n, T, a_sub * 2, NB), np.float32)
    POS = np.zeros((n, T, 2, 2), np.float32); HAND = np.zeros((n, T, 2), np.float32)
    ACT = np.zeros((n, T, 2), np.float32); TOUCH = np.zeros((n, T, 4), np.float32)
    WHO = -np.ones((n, T), np.int64)                                    # disque poussé dans (t-1, t]
    MAT = rng.integers(0, 3, (n, 2)); LM = rng.uniform(np.log(1 / 3), np.log(3), (n, 2)).astype(np.float32)
    IMP = np.zeros((n, T), bool); COL = np.zeros((n, 2, 3), np.float32)
    win = np.hanning(Ls).astype(np.float32); W = band_matrix(Ls); tt_all = np.arange(T * SPF) / SR
    for i in range(n):
        cols = PAL[np.sort(rng.choice(len(PAL), 2, replace=False))]; m = np.exp(LM[i]); COL[i] = cols
        P = np.zeros((2, 2), np.float32); P[0] = rng.uniform(r, 1 - r, 2)
        for _ in range(100):
            P[1] = rng.uniform(r, 1 - r, 2)
            if np.linalg.norm(P[1] - P[0]) > 2.5 * r: break
        V = rng.normal(0, 0.01, (2, 2)).astype(np.float32)               # disques quasi immobiles
        for _ in range(100):
            Hp = rng.uniform(R_HAND, 1 - R_HAND, 2).astype(np.float32)
            dd_ = np.linalg.norm(P - Hp, axis=1)                         # main près d'un disque (pas dessus)
            if np.all(dd_ > r + R_HAND + 0.05) and dd_.min() < 0.4: break
        ev = []; tgt, left = None, 0
        for t in range(T):
            if t > 0:
                # --- politique « jeu de bébé » : viser un disque (et le dépasser pour le pousser) ou un point
                if left <= 0:
                    left = rng.integers(3, 7)
                    if rng.random() < 0.8:
                        k = rng.integers(2); d = P[k] - Hp; tgt = P[k] + 0.15 * d / (np.linalg.norm(d) + 1e-6)
                    else: tgt = rng.uniform(R_HAND, 1 - R_HAND, 2)
                left -= 1
                d = tgt - Hp; a = vmax * d / max(np.linalg.norm(d), vmax) + rng.normal(0, 0.015, 2)
                a = np.clip(a, -vmax, vmax).astype(np.float32); ACT[i, t - 1] = a   # action t-1 : frame t-1 -> t
                Vh = a.copy(); P0, V0 = P.copy(), V.copy()
                Hp = np.clip(Hp + Vh, R_HAND, 1 - R_HAND); P = P + V
                for k in range(2):                                      # murs
                    for dd in range(2):
                        if P[k, dd] < r or P[k, dd] > 1 - r:
                            wall = r if P[k, dd] < r else 1 - r
                            fr_ = float(np.clip((wall - P0[k, dd]) / (V[k, dd] + 1e-9), 0, 0.999))
                            J = 2 * m[k] * abs(V[k, dd]); V[k, dd] = -V[k, dd]
                            P[k, dd] = 2 * r - P[k, dd] if P[k, dd] < r else 2 * (1 - r) - P[k, dd]
                            ev.append((t - 1 + fr_, k, J, P[k, 0]))
                for k in range(2):                                      # MAIN -> disque (choc à 2 corps)
                    dv = P[k] - Hp; dist = float(np.linalg.norm(dv))
                    if 1e-6 < dist < r + R_HAND:
                        nv = dv / dist; s_ = float((V[k] - Vh) @ nv)
                        if s_ < 0:
                            J = -2 * s_ * M_HAND * m[k] / (M_HAND + m[k])
                            V[k] += J / m[k] * nv; Vh -= J / M_HAND * nv
                            TOUCH[i, t, :2] += -J * nv; TOUCH[i, t, 2] += J; TOUCH[i, t, 3] = J / (-s_ + 0.01); WHO[i, t] = k
                            ev.append((t - 0.5, k, J, float((P[k, 0] + Hp[0]) / 2)))
                        P[k] = np.clip(Hp + nv * (r + R_HAND), r, 1 - r)
                dv = P[0] - P[1]; dist = float(np.linalg.norm(dv))     # disque <-> disque
                if 1e-6 < dist < 2 * r:
                    nv = dv / dist; s_ = float((V[0] - V[1]) @ nv)
                    if s_ < 0:
                        J = -2 * s_ * m[0] * m[1] / (m[0] + m[1]); V[0] += J / m[0] * nv; V[1] -= J / m[1] * nv
                        xc = float((P[0, 0] + P[1, 0]) / 2); ev += [(t - 0.5, 0, J, xc), (t - 0.5, 1, J, xc)]
                    push = (2 * r - dist) / 2
                    P[0] = np.clip(P[0] + push * nv, r, 1 - r); P[1] = np.clip(P[1] - push * nv, r, 1 - r)
                V *= damp                                               # frottement : il faut pousser
            img = np.zeros((H, H, 3), np.float32)
            for k in range(2):
                al = np.clip((r - np.sqrt((xx - P[k, 0]) ** 2 + (yy - P[k, 1]) ** 2)) * H + 0.5, 0, 1)[..., None]
                img = img * (1 - al) + cols[k] * al
            hx = np.clip((R_HAND * 0.85 - np.maximum(abs(xx - Hp[0]), abs(yy - Hp[1]))) * H + 0.5, 0, 1)[..., None]
            img = img * (1 - hx) + hx                                   # main = carré blanc
            X[i, t] = img; POS[i, t] = P; HAND[i, t] = Hp
        extra = hum_signal(POS[i], MAT[i], m, T, hum, pitch_mass) if hum > 0 else None   # monde v4 : chaque disque bourdonne
        A[i] = render_audio(ev, MAT[i], m, T, a_sub, rng, W, win, tt_all, pitch_mass, IMP[i], extra)
    return dict(X=X, A=A, MAT=MAT, LM=LM, IMP=IMP, POS=POS, HAND=HAND, ACT=ACT, TOUCH=TOUCH, WHO=WHO, COL=COL)

# ---------------------------------------------------------------- tokens par pas V-JEPA 2 (2 frames)
def step_inputs(w, Tt):
    """audio (n,Tt,2*a_sub*2*NB), toucher (n,Tt,8), action (n,Tt,4) alignés sur les pas V-JEPA 2.
    action du pas s = transitions 2s+1 -> 2s+2 et 2s+2 -> 2s+3 (ce qui mène au pas s+1)."""
    n, T = w["ACT"].shape[:2]; f = T // Tt                             # frames par pas (2 : V-JEPA 2 ; 1 : notre encodeur)
    A = w["A"].reshape(n, Tt, -1); Tch = w["TOUCH"].reshape(n, Tt, -1)
    act = np.concatenate([w["ACT"], np.zeros((n, f, 2), np.float32)], 1)       # ACT[t] : t -> t+1
    Act = np.stack([np.concatenate([act[:, f * s + f - 1 + j] for j in range(f)], -1) for s in range(Tt)], 1)
    return A.astype(np.float32), Tch.astype(np.float32), Act.astype(np.float32)

# ---------------------------------------------------------------- NOTRE encodeur bébé GELÉ (phase 1 -> phase 2)
@torch.no_grad()
def encode_dev(w, a, dev, P=4, d=192, nl=6, nh=6, pred_layers=3, bs=64):
    """encodeur CIBLE (EMA) d'un instantané av_dev_long, appliqué FRAME PAR FRAME (aucune fuite du futur dans
    le latent du pas t, comme l'encodeur image de V-JEPA 2-AC) : 64 patches -> moyenne 2×2 = 4×4 tokens + le
    token audio de la frame -> (n, T, 17, d)."""
    import copy
    from av_jepa import gen_world
    from av_dev import DevJEPA
    from av_dev_long import to_tokens, stereo, T as T0, H as H0
    nP = H0 // P; npf = nP * nP; nv = T0 * npf; da = 2 * 2 * NB; W = max(P * P * 3, da)
    w0 = gen_world(2000, T0, H0, seed=0, a_sub=2, hum=a.hum); A0 = stereo(torch.from_numpy(w0["A"])).reshape(2000, T0, -1)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev)); del w0     # mêmes stats que le pré-entraînement
    m = DevJEPA(W, da, nv, T0, d, nl, nh, pred_layers).to(dev); enc = copy.deepcopy(m.enc)
    enc.load_state_dict(torch.load(a.enc_ckpt, map_location=dev, weights_only=False)["tgt"]); enc.eval(); del m
    X = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)); A = torch.from_numpy(w["A"]); n = len(X); out = []
    for i in range(0, n, bs):
        tok = to_tokens(X[i:i + bs].to(dev), A[i:i + bs].to(dev), P, st); B = len(tok); zs = []
        for t in range(T0):
            if a.dev_ctx == "causal":           # PASSÉ seulement : frames 0..t (comme à l'entraînement, des frames visibles), jamais le futur
                idx = torch.cat([torch.arange((t + 1) * npf), nv + torch.arange(t + 1)]).to(dev).expand(B, -1)
                zz = enc(torch.gather(tok, 1, idx.unsqueeze(-1).expand(-1, -1, tok.size(-1))), idx)
                z = torch.cat([zz[:, t * npf:(t + 1) * npf], zz[:, -1:]], 1)                         # tokens de la frame t
            else:                               # frame seule
                idx = torch.cat([torch.arange(t * npf, (t + 1) * npf), torch.tensor([nv + t])]).to(dev).expand(B, -1)
                z = enc(torch.gather(tok, 1, idx.unsqueeze(-1).expand(-1, -1, tok.size(-1))), idx)   # (B, 65, d)
            zv = z[:, :npf] if a.dev_pool == 1 else \
                F.avg_pool2d(z[:, :npf].reshape(B, nP, nP, d).permute(0, 3, 1, 2), a.dev_pool).flatten(2).transpose(1, 2)   # (B, (8/pool)², d)
            zs.append(torch.cat([zv, z[:, npf:]], 1))
        out.append(torch.stack(zs, 1).half().cpu())
    print(f"  encodeur bébé gelé ({a.enc_ckpt}) -> {tuple(out[0].shape[1:])} par séquence", flush=True)
    return torch.cat(out)

class ACPredictor(nn.Module):
    """bloc-causal : les tokens du pas s voient les pas ≤ s ; ils prédisent le pas s+1."""
    def __init__(s, k, Tt, dv, da, dt, dact, d, nl, nh, mods, residual=True):
        super().__init__()
        s.k, s.Tt, s.mods, s.residual = k, Tt, mods, residual
        s.ev, s.ea, s.et, s.eact = nn.Linear(dv, d), nn.Linear(da, d), nn.Linear(dt, d), nn.Linear(dact, d)
        s.m = k + ("a" in mods) + ("t" in mods) + 1
        s.pos = nn.Parameter(torch.zeros(1, Tt, s.m, d)); nn.init.normal_(s.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, nh, 2 * d, batch_first=True, dropout=0.0, activation="gelu")
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d)
        s.hv, s.ha, s.ht = nn.Linear(d, dv), nn.Linear(d, da), nn.Linear(d, dt)
        if residual: nn.init.zeros_(s.hv.weight); nn.init.zeros_(s.hv.bias)   # part de la COPIE, n'apprend que le changement
        step = torch.arange(Tt).repeat_interleave(s.m)
        s.register_buffer("mask", step[None, :] > step[:, None])           # True = interdit (futur)
    def forward(s, V, A, Tch, Act):
        B = V.size(0); parts = [s.ev(V)]
        if "a" in s.mods: parts.append(s.ea(A).unsqueeze(2))
        if "t" in s.mods: parts.append(s.et(Tch).unsqueeze(2))
        parts.append(s.eact(Act).unsqueeze(2))
        x = (torch.cat(parts, 2) + s.pos).reshape(B, s.Tt * s.m, -1)
        h = s.ln(s.tr(x, mask=s.mask)).reshape(B, s.Tt, s.m, -1)
        dV = s.hv(h[:, :, :s.k]); out = {"v": V + dV if s.residual else dV}; j = s.k   # résiduel : ẑ(s+1) = z(s) + Δ
        if "a" in s.mods: out["a"] = s.ha(h[:, :, j]); j += 1
        if "t" in s.mods: out["t"] = s.ht(h[:, :, j])
        return out, h

def loss_fn(out, V, A, Tch, targets="v"):
    """targets 'v' : toutes les variantes prédisent la MÊME cible (latents visuels) -> comparaison
    équitable (son/toucher = entrées seulement) ; 'own' : chaque variante prédit aussi ses modalités."""
    l = F.smooth_l1_loss(out["v"][:, :-1], V[:, 1:])
    if targets == "own":
        if "a" in out: l = l + F.smooth_l1_loss(out["a"][:, :-1], A[:, 1:])
        if "t" in out: l = l + F.smooth_l1_loss(out["t"][:, :-1], Tch[:, 1:])
    return l

class PosReadout(nn.Module):
    """lecteur GELÉ latents V-JEPA 2 d'un pas -> positions (2 disques + main). Entraîné sur les VRAIS
    latents, puis appliqué aux latents PRÉDITS : erreur en pixels = espace décodé « certifié »."""
    def __init__(s, dv, d=256):
        super().__init__()
        s.proj = nn.Linear(dv, d); s.pos = nn.Parameter(torch.zeros(1, 80, d)); s.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        s.att = nn.MultiheadAttention(d, 4, batch_first=True); s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 6))
    def forward(s, v):                                                  # v (B, k, dv)
        h = s.proj(v) + s.pos[:, :v.size(1)]
        return s.out(s.att(s.q.expand(len(v), -1, -1), h, h)[0][:, 0])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=6000); p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--T", type=int, default=16); p.add_argument("--a_sub", type=int, default=2)
    p.add_argument("--train_cache", type=str, default="/content/av_act_train.pt")
    p.add_argument("--test_cache", type=str, default="/content/av_act_test.pt")
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--pool", type=int, default=4); p.add_argument("--enc_bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--d", type=int, default=256)
    p.add_argument("--nl", type=int, default=4); p.add_argument("--nh", type=int, default=8)
    p.add_argument("--variants", type=str, default="v,va,vat"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fig", type=str, default="av_act_world.png")
    p.add_argument("--targets", type=str, default="v", choices=["v", "own"])
    p.add_argument("--ctx", type=int, default=3, help="dernier pas de contexte observé avant le rollout")
    p.add_argument("--sens_drop", type=float, default=0.5, help="p de couper son/toucher sur un suffixe (entraînement)")
    p.add_argument("--residual", type=int, default=1, help="ẑ(s+1) = z(s) + Δ, tête zéro-init (leçon pusht_vjepa2 : sinon collé à la moyenne)")
    p.add_argument("--encoder", type=str, default="vjepa2", choices=["vjepa2", "dev"], help="dev = NOTRE JEPA bébé GELÉ (av_dev_long, ex. v4 pas 20k)")
    p.add_argument("--enc_ckpt", type=str, default="/content/drive/MyDrive/jepa_runs/av_dev_v4_20k.pt")
    p.add_argument("--eval_bs", type=int, default=64)
    p.add_argument("--perm_ro", type=int, default=1, help="lecteur de positions invariant à l'ordre des 2 disques (sinon il doit deviner « qui est le disque 0 » par la couleur)")
    p.add_argument("--dev_ctx", type=str, default="frame", choices=["frame", "causal"], help="encoder chaque frame seule, ou avec tout son PASSÉ")
    p.add_argument("--dev_pool", type=int, default=2, help="regroupement des 8×8 patches de notre encodeur (1 = aucun : position fine)")
    p.add_argument("--hum", type=float, default=0.0, help="monde v4 : bourdonnement continu des disques (0.15 = comme le pré-entraînement)")
    p.add_argument("--ro_steps", type=int, default=3000, help="pas du lecteur de positions")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    wtr = gen_world_act(a.n_train, a.T, seed=a.seed, a_sub=a.a_sub, hum=a.hum)
    wte = gen_world_act(a.n_test, a.T, seed=a.seed + 5000, a_sub=a.a_sub, hum=a.hum)
    cont = (wtr["WHO"] >= 0).any(1).mean()
    print(f"monde v3 : {a.n_train}+{a.n_test} séquences | {cont:.0%} avec poussée | poussées/séq "
          f"{(wtr['WHO'] >= 0).sum(1).mean():.1f} ({time.time() - t0:.0f}s)", flush=True)
    if a.fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, a.T, figsize=(1.3 * a.T, 3))
        for e in range(2):
            for t in range(a.T):
                ax[e, t].imshow(wte["X"][e, t]); ax[e, t].axis("off")
                ax[e, t].set_title(("P" if wte["WHO"][e, t] >= 0 else "") + f"{t}", fontsize=7)
        plt.tight_layout(); plt.savefig(a.fig, dpi=100); plt.close(); print(f"figure -> {a.fig}")
    def feats(path, w):
        if os.path.exists(path):
            Z = torch.load(path)["Z"]
            if len(Z) == len(w["X"]): print(f"  cache {path}", flush=True); return Z
        Z = encode_vjepa2(w["X"], a, dev) if a.encoder == "vjepa2" else encode_dev(w, a, dev)
        torch.save(dict(Z=Z), path); return Z
    Ztr, Zte = feats(a.train_cache, wtr), feats(a.test_cache, wte)
    Tt, k, dv = Ztr.shape[1], Ztr.shape[2], Ztr.shape[3]; f = a.T // Tt   # frames par pas
    vmu, vsd = Ztr[:2000].float().mean((0, 1, 2)), Ztr[:2000].float().std((0, 1, 2)) + 1e-4
    Vtr = ((Ztr.float() - vmu) / vsd).half(); Vte = ((Zte.float() - vmu) / vsd).half(); del Ztr, Zte
    Atr, Ttr, Ctr = step_inputs(wtr, Tt); Ate, Tte_, Cte = step_inputs(wte, Tt)
    st = {n_: (x.mean((0, 1)), x.std((0, 1)) + 1e-4) for n_, x in [("a", Atr), ("t", Ttr), ("c", Ctr)]}
    nz = lambda x, n_: torch.from_numpy((x - st[n_][0]) / st[n_][1])
    Atr, Ttr, Ctr, Ate, Tte_, Cte = nz(Atr, "a"), nz(Ttr, "t"), nz(Ctr, "c"), nz(Ate, "a"), nz(Tte_, "t"), nz(Cte, "c")
    def mass_tok(w):                                                    # ORACLE (contrôle) : vraies log-masses + couleurs
        x = np.concatenate([w["LM"] / 0.63, w["COL"].reshape(len(w["LM"]), 6)], -1).astype(np.float32)
        return torch.from_numpy(np.repeat(x[:, None], Tt, 1))
    Mtr, Mte = mass_tok(wtr), mass_tok(wte)
    # sous-ensembles d'éval : pas s+1 (frames 2s+2, 2s+3) avec poussée ; disque déjà touché avant ?
    who = wte["WHO"]; n_te = len(who)
    push = np.zeros((n_te, Tt - 1), bool); retouch = np.zeros((n_te, Tt - 1), bool)
    for i in range(n_te):
        for s_ in range(Tt - 1):
            ks = {int(x) for x in who[i, f * (s_ + 1):f * (s_ + 2)] if x >= 0}
            if ks:
                push[i, s_] = True
                before = {int(x) for x in who[i, :f * (s_ + 1)] if x >= 0}
                retouch[i, s_] = bool(ks & before)
    first = push & ~retouch
    copy_err = (Vte[:, :-1].float() - Vte[:, 1:].float()).abs().mean((2, 3)).numpy()      # (n, Tt-1)
    print(f"tokens : {Tt} pas × ({k} visuels + son + toucher + action) | pas avec poussée {push.mean():.0%} "
          f"(1re fois {first.mean():.0%}, disque déjà touché {retouch.mean():.0%})", flush=True)
    # --- lecteur de positions (vrais latents) -> métrique décodée en pixels
    def pos_target(w):                                                  # état en fin de pas s (dernière frame du pas)
        P = w["POS"][:, f - 1::f].reshape(len(w["POS"]), Tt, 4); Hh = w["HAND"][:, f - 1::f]
        return torch.from_numpy(np.concatenate([P, Hh], -1))           # (n, Tt, 6)
    Ptr, Pte = pos_target(wtr), pos_target(wte); torch.manual_seed(0)
    ro = PosReadout(dv).to(dev); opt = torch.optim.AdamW(ro.parameters(), 1e-3, weight_decay=1e-2)
    for it in range(a.ro_steps):
        bi = torch.randint(0, len(Vtr), (128,)); sj = torch.randint(0, Tt, (128,))
        pr_, y_ = ro(Vtr[bi, sj].to(dev).float()), Ptr[bi, sj].to(dev)
        loss = F.mse_loss(pr_, y_) if not a.perm_ro else torch.minimum(((pr_ - y_) ** 2).mean(-1), ((pr_[:, SWAP] - y_) ** 2).mean(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ro.eval()
    def readout(Vs):                                                   # (n, S, k, dv) -> (n, S, 6)
        with torch.no_grad():
            return torch.cat([ro(Vs[i:i + 512].flatten(0, 1).to(dev).float()).cpu().view(-1, Vs.size(1), 6)
                              for i in range(0, len(Vs), 512)])
    true_next = Pte[:, 1:]                                             # (n, Tt-1, 6)
    def align(pred, true):                                             # lecteur INVARIANT à l'ordre des disques : meilleure attribution
        if not a.perm_ro: return pred
        sw = pred[..., SWAP]; better = ((sw - true)[..., :4] ** 2).sum(-1) < ((pred - true)[..., :4] ** 2).sum(-1)
        return torch.where(better[..., None], sw, pred)
    def disk_err(pred):                                                # erreur px des disques POUSSÉS au pas s+1
        e = (align(pred, true_next)[..., :4] - true_next[..., :4]).view(n_te, Tt - 1, 2, 2).norm(dim=-1) * 32   # (n, Tt-1, 2)
        pushed = np.zeros((n_te, Tt - 1, 2), bool)
        for i in range(n_te):
            for s_ in range(Tt - 1):
                for x in who[i, f * (s_ + 1):f * (s_ + 2)]:
                    if x >= 0: pushed[i, s_, x] = True
        return e.numpy(), pushed
    ceil = readout(Vte[:, 1:]); e_ceil, pushed = disk_err(ceil)
    e_copy, _ = disk_err(readout(Vte[:, :-1]))
    def px(e, msk): return float(e[msk].mean())
    # --- ROLLOUT : contexte = pas 0..c (on a vu/entendu/senti), puis H pas IMAGINÉS avec les actions prévues ;
    #     son et toucher du futur inconnus (zéro) ; les latents prédits sont réinjectés comme vision.
    c, H = a.ctx, Tt - 1 - a.ctx
    def rollout(m, C):
        out_all = []
        with torch.no_grad():
            for i in range(0, n_te, a.eval_bs):
                sl = slice(i, i + a.eval_bs); V = Vte[sl].to(dev).float().clone()
                A_, T_ = Ate[sl].to(dev).clone(), Tte_[sl].to(dev).clone(); A_[:, c + 1:] = 0; T_[:, c + 1:] = 0
                for h in range(1, H + 1):
                    o, _ = m(V, A_, T_, C[sl].to(dev)); V[:, c + h] = o["v"][:, c + h - 1]
                out_all.append(V[:, c + 1:].half().cpu())
        return readout(torch.cat(out_all))                              # (n, H, 6)
    tr_fut = Pte[:, c + 1:]                                             # vraies positions futures (n, H, 6)
    def roll_err(pred):                                                 # (n, H, 2) px par disque
        return ((align(pred, tr_fut)[..., :4] - tr_fut[..., :4]).view(n_te, H, 2, 2).norm(dim=-1) * 32).numpy()
    lm_te = wte["LM"]; touched = np.zeros((n_te, 2), bool); pushed_fut = np.zeros((n_te, 2), bool)
    for i in range(n_te):
        for x in who[i, :f * (c + 1)]:
            if x >= 0: touched[i, x] = True                             # senti PENDANT le contexte
        for x in who[i, f * (c + 1):]:
            if x >= 0: pushed_fut[i, x] = True                          # poussé pendant le futur imaginé
    extreme = np.abs(lm_te) > 0.7                                       # masse < 0.5 ou > 2
    rollouts = {"copie": readout(Vte[:, c:c + 1].expand(-1, H, -1, -1).contiguous()), "vrai futur": readout(Vte[:, c + 1:])}
    pf = pushed & first[..., None]; pr = pushed & retouch[..., None]
    print(f"lecteur de positions (vrais latents) : disques poussés {px(e_ceil, pushed):.2f} px (plafond) ; copie {px(e_copy, pushed):.2f} px", flush=True)
    dec = [("lecture vrai futur", e_ceil), ("copie", e_copy)]
    rows = [("copie", copy_err)]
    for mods in a.variants.split(","):
        torch.manual_seed(a.seed)
        CtrV, CteV = (torch.cat([Ctr, Mtr], -1), torch.cat([Cte, Mte], -1)) if "m" in mods else (Ctr, Cte)   # oracle -> token action
        m = ACPredictor(k, Tt, dv, Atr.size(-1), Ttr.size(-1), CtrV.size(-1), a.d, a.nl, a.nh, mods, bool(a.residual)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05); tt0 = time.time()
        for it in range(1, a.steps + 1):
            bi = torch.randint(0, len(Vtr), (a.bs,))
            V, A_, T_, C_ = Vtr[bi].to(dev).float(), Atr[bi].to(dev), Ttr[bi].to(dev), CtrV[bi].to(dev)
            if a.sens_drop > 0:                                         # suffixe sans son/toucher (= futur imaginé)
                cut = torch.randint(1, Tt + 1, (a.bs, 1), device=dev)
                keep = (torch.arange(Tt, device=dev)[None] < cut) | (torch.rand(a.bs, 1, device=dev) > a.sens_drop)
                A_in, T_in = A_ * keep[..., None], T_ * keep[..., None]
            else: A_in, T_in = A_, T_
            out, _ = m(V, A_in, T_in, C_); loss = loss_fn(out, V, A_, T_, a.targets)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if it % (a.steps // 5) == 0: print(f"  [{mods}] step {it}  loss {loss.item():.4f}  ({time.time() - tt0:.0f}s)", flush=True)
        m.eval(); errs = []; preds = []
        with torch.no_grad():
            for i in range(0, n_te, a.eval_bs):
                sl = slice(i, i + a.eval_bs); V = Vte[sl].to(dev).float()
                out, _ = m(V, Ate[sl].to(dev), Tte_[sl].to(dev), CteV[sl].to(dev))
                errs.append((out["v"][:, :-1] - V[:, 1:]).abs().mean((2, 3)).cpu()); preds.append(out["v"][:, :-1].half().cpu())
        rows.append((mods.upper(), torch.cat(errs).numpy()))
        e_dec, _ = disk_err(readout(torch.cat(preds))); dec.append((mods.upper(), e_dec)); del preds
        rollouts[mods.upper()] = rollout(m, CteV)
        e = rows[-1][1]
        print(f"  {mods.upper():>4s} | erreur latents pas suivant : tous {e.mean():.4f} | poussée {e[push].mean():.4f} "
              f"| 1re poussée {e[first].mean():.4f} | disque déjà touché {e[retouch].mean():.4f}", flush=True)
    print("\n========== PRÉDICTION DU PAS SUIVANT (latents V-JEPA 2, jeu tenu à l'écart ; plus bas = mieux) ==========")
    print(f"{'modèle':>6s} | {'tous':>7s} | {'poussée':>7s} | {'1re pouss.':>10s} | {'déjà touché':>11s}")
    for name, e in rows:
        print(f"{name:>6s} | {e.mean():7.4f} | {e[push].mean():7.4f} | {e[first].mean():10.4f} | {e[retouch].mean():11.4f}")
    print(f"\n===== ROLLOUT {H} PAS IMAGINÉS après {c + 1} pas de contexte (px disques POUSSÉS dans le futur, horizon final) =====")
    groups = [("tous poussés", pushed_fut), ("déjà touché (contexte)", pushed_fut & touched),
              ("jamais touché", pushed_fut & ~touched), ("déjà touché + masse extrême", pushed_fut & touched & extreme)]
    print(f"{'modèle':>11s} | " + " | ".join(f"{g:>27s}" for g, _ in groups) + f" | {'tous disques h=1..H':>20s}")
    for name, pr_ in rollouts.items():
        e = roll_err(pr_)
        print(f"{name:>11s} | " + " | ".join(f"{e[:, -1][msk].mean():27.2f}" for _, msk in groups)
              + " | " + " ".join(f"{e[:, h].mean():.2f}" for h in range(H)))
    print(f"(effectifs : " + ", ".join(f"{g} {msk.sum()}" for g, msk in groups) + ")")
    print("\n===== POSITION PRÉDITE DES DISQUES POUSSÉS (px, lecteur gelé sur latents prédits ; plus bas = mieux) =====")
    print(f"{'modèle':>18s} | {'poussés':>7s} | {'1re poussée':>11s} | {'déjà touché':>11s}")
    for name, e in dec:
        print(f"{name:>18s} | {px(e, pushed):7.2f} | {px(e, pf):11.2f} | {px(e, pr):11.2f}")
    g = {n_: e for n_, e in rows}
    if "VAT" in g and "VA" in g:
        print(f"\nTOUCHER (VAT vs VA) sur disque déjà touché : {g['VA'][retouch].mean():.4f} -> {g['VAT'][retouch].mean():.4f} "
              f"({100 * (1 - g['VAT'][retouch].mean() / g['VA'][retouch].mean()):+.1f} %) ; 1re poussée : "
              f"{100 * (1 - g['VAT'][first].mean() / g['VA'][first].mean()):+.1f} %")
    if "VA" in g and "V" in g:
        print(f"SON (VA vs V) sur poussée : {100 * (1 - g['VA'][push].mean() / g['V'][push].mean()):+.1f} %")
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
