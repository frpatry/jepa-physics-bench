"""
MONDE v5 « VARIÉ » (phase 1 du bébé) — leçon du diag phase 2 : un encodeur qui n'a vu que 2 disques
toujours en mouvement ne représente bien que des DISQUES QUI BOUGENT (main ≈ illisible, objets au repos
perdus dès qu'il voit plusieurs frames). Le monde du bébé doit être varié dans ce qui comptera ensuite :
  - 1 à 3 objets de FORMES variées : disque, carré, triangle, T (le T de Push-T : le bébé le connaîtra
    sous tous les angles avant qu'on lui demande de le pousser) ; tailles et couleurs variées ;
  - ROTATION (vitesse angulaire, changée par les chocs) -> l'orientation devient une propriété visible ;
  - objets parfois AU REPOS (ne bougent que si on les heurte), frottement variable (ralentissent, s'arrêtent) ;
  - sa MAIN (carré blanc, comme en phase 2) présente dès le départ, qui BABILLE et pousse les objets ;
  - même son qu'avant (impacts modaux par matériau/masse, bourdonnement continu, stéréo selon x).
Collisions : cercles englobants (approximation suffisante pour apprendre à VOIR ; la physique fine du T
viendra avec Push-T). Aucune étiquette n'est donnée au modèle : SHAPE/ANG/POS ne servent qu'aux examens.

  python av_world5.py            # figure de contrôle av_world5.png
"""
import math
import numpy as np
from av_jepa import PAL, NB, SPF, SR, band_matrix, render_audio, hum_signal

SHAPES = ("disque", "carré", "triangle", "T")
R_HAND = 0.07

def shape_alpha(kind, s, th, cx, cy, xx, yy, H):
    """couverture anti-aliasée (H, H) d'une forme de « rayon » s, orientée th, centrée (cx, cy)."""
    dx, dy = xx - cx, yy - cy; c, si = math.cos(th), math.sin(th)
    u, v = c * dx + si * dy, -si * dx + c * dy                       # repère de l'objet
    if kind == 0: d = s - np.sqrt(dx * dx + dy * dy)
    elif kind == 1: d = 0.8 * s - np.maximum(abs(u), abs(v))
    elif kind == 2:                                                    # triangle équilatéral, inrayon 0.55 s
        d = 0.55 * s - np.max(np.stack([u * math.cos(a) + v * math.sin(a) for a in (math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6)]), 0)
    else:                                                              # T : barre du haut + jambe
        bar = np.minimum(s - abs(u), 0.25 * s - abs(v - 0.62 * s))
        leg = np.minimum(0.25 * s - abs(u), 0.62 * s - abs(v + 0.25 * s))
        d = np.maximum(bar, leg)
    return np.clip(d * H + 0.5, 0, 1)

def gen_world_v5(n, T=16, H=32, seed=0, a_sub=2, hum=0.0, hum_mode="hum", p_hand=0.7, force_T=False, smin=0.02, smax=0.11, vmax=0.1):
    """-> X (n,T,H,H,3), A (n,T,a_sub*2,NB), IMP (n,T), POS (n,T,3,2) [NaN si absent], ANG (n,T,3), SHAPE (n,3) [-1 absent],
    NOBJ (n,), HAND (n,T,2) [NaN si pas de main]. force_T : l'objet 0 est un T (sonde d'examen)."""
    rng = np.random.default_rng(seed); Ls = SPF // a_sub; K = 3
    yy, xx = (np.mgrid[0:H, 0:H].astype(np.float32) + 0.5) / H
    X = np.zeros((n, T, H, H, 3), np.float32); A = np.zeros((n, T, a_sub * 2, NB), np.float32)
    POS = np.full((n, T, K, 2), np.nan, np.float32); ANG = np.full((n, T, K), np.nan, np.float32)
    SHAPE = -np.ones((n, K), np.int64); NOBJ = np.zeros(n, np.int64); HAND = np.full((n, T, 2), np.nan, np.float32)
    IMP = np.zeros((n, T), bool)
    win = np.hanning(Ls).astype(np.float32); W = band_matrix(Ls); tt_all = np.arange(T * SPF) / SR
    for i in range(n):
        N = 1 if force_T and rng.random() < 0.3 else int(rng.integers(1, K + 1)); NOBJ[i] = N
        kind = rng.integers(0, 4, N); kind[0] = 3 if force_T else kind[0]; SHAPE[i, :N] = kind
        s = np.where(kind == 3, rng.uniform(0.12, 0.16, N), rng.uniform(0.08, 0.14, N)); rc = np.where(kind == 3, 0.95, 0.9) * s     # rayon de collision (cercle englobant)
        cols = np.clip(PAL[rng.choice(len(PAL), N, replace=False)] * rng.uniform(0.7, 1.0, (N, 1)) + rng.normal(0, 0.08, (N, 3)), 0.1, 0.95).astype(np.float32)
        m = np.exp(rng.uniform(np.log(1 / 3), np.log(3), N)); mat = rng.integers(0, 3, N)
        P = np.zeros((N, 2), np.float32)
        for k in range(N):
            for _ in range(200):
                P[k] = rng.uniform(rc[k], 1 - rc[k], 2)
                if all(np.linalg.norm(P[k] - P[j]) > rc[k] + rc[j] + 0.03 for j in range(k)): break
        moving = rng.random(N) < 0.6                                 # sinon AU REPOS (jusqu'à ce qu'on le heurte)
        th = rng.uniform(0, 2 * math.pi, N); sp = rng.uniform(smin, smax, N) * moving
        V = np.stack([sp * np.cos(th), sp * np.sin(th)], -1).astype(np.float32)
        ang = rng.uniform(0, 2 * math.pi, N); om = rng.normal(0, 0.15, N) * moving
        fric = rng.choice([0.0, 0.0015, 0.004])                      # glisse / ralentit / s'arrête vite
        hand = rng.random() < p_hand; Hp = None; left, mode, spd, tgt, dirv = 0, "immobile", vmax, None, None
        if hand:
            for _ in range(200):
                Hp = rng.uniform(R_HAND, 1 - R_HAND, 2).astype(np.float32)
                if np.all(np.linalg.norm(P - Hp, axis=1) > rc + R_HAND + 0.02): break
        ev = []
        for t in range(T):
            if t > 0:
                if hand:                                             # BABILLAGE (mêmes modes que av_act --babble 1)
                    if left <= 0:
                        left = rng.integers(2, 7); u_ = rng.random(); spd = rng.uniform(0.015, vmax); mode = "tgt"
                        if u_ < 0.45:
                            k = rng.integers(N); d = P[k] - Hp; tgt = P[k] + rng.uniform(0.0, 0.2) * d / (np.linalg.norm(d) + 1e-6)
                        elif u_ < 0.6: tgt = rng.uniform(R_HAND, 1 - R_HAND, 2)
                        elif u_ < 0.7: mode = "immobile"
                        elif u_ < 0.85: a_ = rng.uniform(0, 2 * math.pi); dirv = np.array([math.cos(a_), math.sin(a_)]); mode = "direction"
                        else: mode = "gigote"
                    left -= 1
                    if mode == "immobile": a = rng.normal(0, 0.004, 2)
                    elif mode == "direction": a = spd * dirv + rng.normal(0, 0.015, 2)
                    elif mode == "gigote": a = rng.normal(0, 0.05, 2)
                    else: d = tgt - Hp; a = spd * d / max(np.linalg.norm(d), spd) + rng.normal(0, 0.015, 2)
                    Vh = np.clip(a, -vmax, vmax).astype(np.float32); Hp = np.clip(Hp + Vh, R_HAND, 1 - R_HAND)
                P0 = P.copy(); P = P + V; ang = ang + om
                for k in range(N):                                   # murs
                    for dd in range(2):
                        if P[k, dd] < rc[k] or P[k, dd] > 1 - rc[k]:
                            wall = rc[k] if P[k, dd] < rc[k] else 1 - rc[k]
                            fr_ = float(np.clip((wall - P0[k, dd]) / (V[k, dd] + 1e-9), 0, 0.999))
                            J = 2 * m[k] * abs(V[k, dd]); V[k, dd] = -V[k, dd]; om[k] += rng.normal(0, 0.05)
                            P[k, dd] = 2 * rc[k] - P[k, dd] if P[k, dd] < rc[k] else 2 * (1 - rc[k]) - P[k, dd]
                            if J > 1e-4: ev.append((t - 1 + fr_, k, J, P[k, 0]))
                if hand:                                             # main -> objet (choc à 2 corps, main de masse 1)
                    for k in range(N):
                        dv = P[k] - Hp; dist = float(np.linalg.norm(dv))
                        if 1e-6 < dist < rc[k] + R_HAND:
                            nv = dv / dist; s_ = float((V[k] - Vh) @ nv)
                            if s_ < 0:
                                J = -2 * s_ * m[k] / (1 + m[k]); V[k] += J / m[k] * nv; om[k] += rng.normal(0, 0.1)
                                ev.append((t - 0.5, k, J, float((P[k, 0] + Hp[0]) / 2)))
                            P[k] = np.clip(Hp + nv * (rc[k] + R_HAND), rc[k], 1 - rc[k])
                for k in range(N):                                   # objet <-> objet
                    for j in range(k + 1, N):
                        dv = P[k] - P[j]; dist = float(np.linalg.norm(dv))
                        if 1e-6 < dist < rc[k] + rc[j]:
                            nv = dv / dist; s_ = float((V[k] - V[j]) @ nv)
                            if s_ < 0:
                                J = -2 * s_ * m[k] * m[j] / (m[k] + m[j]); V[k] += J / m[k] * nv; V[j] -= J / m[j] * nv
                                om[k] += rng.normal(0, 0.1); om[j] += rng.normal(0, 0.1)
                                xc = float((P[k, 0] + P[j, 0]) / 2); ev += [(t - 0.5, k, J, xc), (t - 0.5, j, J, xc)]
                            push = (rc[k] + rc[j] - dist) / 2
                            P[k] = np.clip(P[k] + push * nv, rc[k], 1 - rc[k]); P[j] = np.clip(P[j] - push * nv, rc[j], 1 - rc[j])
                if fric > 0:
                    v_ = np.linalg.norm(V, axis=1, keepdims=True); V *= np.clip(1 - fric / (v_ + 1e-9), 0, 1)
                    om *= 0.97 if fric < 0.003 else 0.9
            img = np.zeros((H, H, 3), np.float32)
            for k in range(N):
                al = shape_alpha(kind[k], s[k], ang[k], P[k, 0], P[k, 1], xx, yy, H)[..., None]
                img = img * (1 - al) + cols[k] * al
            if hand:
                hx = np.clip((R_HAND * 0.85 - np.maximum(abs(xx - Hp[0]), abs(yy - Hp[1]))) * H + 0.5, 0, 1)[..., None]
                img = img * (1 - hx) + hx; HAND[i, t] = Hp
            X[i, t] = img; POS[i, t, :N] = P; ANG[i, t, :N] = ang
        extra = hum_signal(POS[i, :, :N], mat, m, T, hum, 1, hum_mode) if hum > 0 else None
        A[i] = render_audio(ev, mat, m, T, a_sub, rng, W, win, tt_all, 1, IMP[i], extra)
    return dict(X=X, A=A, IMP=IMP, POS=POS, ANG=ANG, SHAPE=SHAPE, NOBJ=NOBJ, HAND=HAND)

if __name__ == "__main__":
    import time, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t0 = time.time(); w = gen_world_v5(200, seed=0, hum=0.15); dt = time.time() - t0
    print(f"200 séquences en {dt:.1f}s ({1000 * dt / 200:.1f} ms/séq) | objets/séq {w['NOBJ'].mean():.2f} | main {np.isfinite(w['HAND'][:, 0, 0]).mean():.0%} "
          f"| formes {np.bincount(w['SHAPE'][w['SHAPE'] >= 0], minlength=4)} | séq. avec choc {w['IMP'][:, 1:].any(1).mean():.0%}")
    fig, ax = plt.subplots(6, 8, figsize=(12, 9))
    for r_ in range(6):
        for c_ in range(8):
            ax[r_, c_].imshow(w["X"][r_, 2 * c_]); ax[r_, c_].axis("off")
            if c_ == 0: ax[r_, c_].set_title(",".join(SHAPES[k] for k in w["SHAPE"][r_] if k >= 0), fontsize=7)
    plt.tight_layout(); plt.savefig("av_world5.png", dpi=90); print("figure -> av_world5.png")
