"""
MONDE DE PHASE 0 « MES MAINS » — le bébé découvre son corps AVANT les objets (fondements : recherche_bebe_mains.md).
QUATRE SENS (entrent dans l'ENCODEUR, appris ensemble, chacun parfois masqué et deviné par les autres) :
  - VUE      : image H×H ; la main = carré blanc, GRANDE au début (proche des yeux) puis qui rapetisse vers la table ;
  - OUÏE     : stéréo ; la main FROISSE quand elle bouge (silence à l'arrêt), le mobile TINTE, chocs modaux, frottements ;
  - TOUCHER  : petite PEAU, 4 côtés de la main (droite, gauche, bas, haut) × [choc, appui ∝ lourdeur sentie] ;
  - SENS DU BRAS (proprioception) : position + déplacement RÉELS de la main, sentis (bruités), même hors du champ de vision.
UNE VOLONTÉ (n'entre QUE dans le PRÉDICTEUR, jamais dans l'encodeur -> pas de triche) : la COMMANDE = vitesse voulue
(copie d'efférence). Commande ≠ mouvement réel quand le bras est bloqué (objet lourd qui résiste, bout du bras) :
c'est là que se découvrent le contact, la lourdeur et la portée.

Étapes (âge équivalent) :
  0a BRAS        (0–2 mois) : main très grande, rien d'autre ; elle gigote.
  0b MAINS       (2–4 mois) : main grande, qui entre et sort du champ (on l'entend et on la sent encore).
  0c CONTINGENCE (2–3 mois) : un MOBILE hors de portée relié au geste par un ruban invisible (Rovee-Collier) ; parfois
                              AUTONOME (le monde bouge seul), parfois DÉCONNECTÉ en cours de route (extinction -> surprise).
  0d CONTACT     (3–4 mois) : main qui rapetisse, 1–2 objets à portée, surtout au repos (c'est MOI qui les fais bouger),
                              parfois un objet qui roule vers la main (toucher PASSIF).
  0e DISTANCE    (5 mois +) : monde v5 complet (1–3 objets variés) + bras de longueur limitée : objets parfois hors de portée.
La vue « du bébé » (flou uniforme, gris, contraste — cf. recherche : flou NEURAL, pas de myopie) est appliquée à
l'entraînement selon VIEW[étape], pas ici. Étiquettes (HAND, POS, LINK, TSRC, ...) = examens seulement.

  python av_world0.py      # figures de contrôle av_world0.png (le monde) + av_world0_sens.png (4 sens + volonté)
"""
import math
import numpy as np
from av_jepa import PAL, NB, SPF, SR, F_MAT, band_matrix, render_audio, hum_signal
from av_world5 import shape_alpha

SHOULDER = np.array([0.5, 1.15], np.float32)          # épaule sous le bas de l'image (table vue de dessus) -> portée du bras
STAGES = {  # taille de la main (R) | boîte du centre de la main | nb objets | mobile | longueur du bras | part d'objets au repos
    "0a": dict(hand=(0.16, 0.20), box=(0.0, 1.0), nobj=(0, 0), mobile=False, reach=None, rest=0.0),
    "0b": dict(hand=(0.12, 0.18), box=(-0.35, 1.35), nobj=(0, 0), mobile=False, reach=None, rest=0.0),
    "0c": dict(hand=(0.10, 0.13), box=(0.0, 1.0), nobj=(0, 0), mobile=True, reach=(0.62, 0.68), rest=0.0),
    "0d": dict(hand=(0.08, 0.11), box=(0.0, 1.0), nobj=(1, 2), mobile=False, reach=(0.95, 1.0), rest=0.7),
    "0e": dict(hand=(0.07, 0.07), box=(0.0, 1.0), nobj=(1, 3), mobile=False, reach=(0.7, 0.9), rest=0.4),
}
VIEW = {"0a": dict(sigma=3.0, gray=1.0, contrast=0.5), "0b": dict(sigma=2.5, gray=0.7, contrast=0.7),
        "0c": dict(sigma=2.0, gray=0.4, contrast=0.85), "0d": dict(sigma=1.0, gray=0.0, contrast=1.0),
        "0e": dict(sigma=0.0, gray=0.0, contrast=1.0)}
SIDES = ("droite", "gauche", "bas", "haut")

def cont_sound(amp, x, T, rng, kind, f0=1000.0):
    """son CONTINU d'un objet : amplitude amp (T,) et pano x (T,) par frame -> (2, T*SPF).
    Fenêtre audio t = mouvement pendant (t-1, t] (même convention que render_audio)."""
    n_s = T * SPF; fr = np.arange(n_s) / SPF - 0.5
    a = np.interp(fr, np.arange(T), amp); xc = np.clip(np.interp(fr, np.arange(T), x), 0, 1)
    if kind == "froisse": z = rng.normal(0, 1, n_s); src = 0.5 * (z - np.roll(z, 1))          # bruit aigu = froissement
    else:
        tt = np.arange(n_s) / SR                                                              # clochette : partiels inharmoniques
        src = sum(np.sin(2 * math.pi * f0 * r * tt + r) / (1 + j) for j, r in enumerate((1.0, 2.32, 4.25)) if f0 * r < 0.95 * SR / 2)
    return np.stack([a * np.sqrt(1 - xc) * src, a * np.sqrt(xc) * src]).astype(np.float32)

def att_out(p):
    """volume d'un son produit HORS DU CHAMP : décroît avec la distance au bord de l'image (1 dans le champ)."""
    d = float(np.linalg.norm(np.maximum(0, np.maximum(-np.asarray(p), np.asarray(p) - 1))))
    return 1.0 / (1.0 + 6.0 * d)

def fric_sound(pos, mat, m, T, level):
    """FROTTEMENT des objets (comme hum_signal mode « fric ») mais AUDIBLE HORS CHAMP : pano saturé au bord, volume atténué
    avec la distance au champ (on entend l'objet sorti s'éloigner, puis cogner le mur extérieur). pos (T, N, 2) -> (2, T*SPF)."""
    n_s = T * SPF; tt = np.arange(n_s) / SR; fr = np.arange(n_s) / SPF; sig = np.zeros((2, n_s), np.float32)
    vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=-1)
    for k in range(pos.shape[1]):
        x = np.interp(fr, np.arange(T), pos[:, k, 0]); sp = np.interp(fr, np.arange(T), vel[:, k])
        g = np.interp(fr, np.arange(T), np.array([att_out(q) for q in pos[:, k]]))
        f0 = 0.5 * F_MAT[mat[k]] * m[k] ** (-1 / 3); tone = sum(np.sin(2 * math.pi * f0 * h * tt + 1.3 * h * k) / h for h in (1, 2, 3))
        amp = level * (sp / 0.08) * math.sqrt(m[k]) * g; xc = np.clip(x, 0, 1)
        sig[0] += amp * np.sqrt(1 - xc) * tone; sig[1] += amp * np.sqrt(xc) * tone
    return sig

def gen_world0(n, stage="0a", T=16, H=32, seed=0, a_sub=2, hum=0.15, vmax=0.1, mobile_delay=0, smin=0.02, smax=0.08,
               force_mtype=None, force_tcut=None, p_obj=0.45, mob_size=0.10, calm=0.0, p_out=0.0, table=0.4, p_auto=None, kick=0.0, p_parent=0.0):
    """-> sens : X (n,T,H,H,3), A (n,T,a_sub*2,NB), TOUCH (n,T,8), PROP (n,T,4) ; volonté : CMD (n,T,2) ;
    examens : HAND (n,T,2), RH (n,), INVIEW (n,T), POS (n,T,3,2) [NaN absent], ANG, SHAPE, NOBJ, IMP, TSRC (n,T) [0 rien,
    1 toucher ACTIF, 2 PASSIF], MOB (n,T,2), LINK (n,T) [mobile relié au geste], MTYPE (n,) [0 relié, 1 autonome,
    2 relié puis coupé, -1 aucun]. mobile_delay > 0 : le mobile répond au geste avec retard (test « direct / différé »).
    force_mtype / force_tcut : paires APPARIÉES pour les tests de surprise (même graine -> même séquence jusqu'à la coupure).
    p_out : part des séquences où la TABLE DÉBORDE du champ de vision (murs à `table` au-delà du bord, idée user) : un objet poussé
    peut SORTIR du champ (on l'entend encore glisser et cogner le mur extérieur, atténué) puis parfois revenir. VIS (n,T,3) = visible.
    p_auto / kick : « JE REGARDE DES CHOSES BOUGER » (étape passive, idée validée avec l'user) — part d'objets qui bougent seuls dès
    le départ (remplace le réglage de l'étape) et proba, par image et par objet, d'une poussée EXTÉRIEURE (une main d'adulte
    invisible déplace le jouet) : le mouvement commun qui permet de DÉCOUPER les objets (Kellman & Spelke), sans que le bébé agisse.
    p_parent : part des séquences où la MAIN D'UN PARENT (idée user) entre dans le champ, va derrière un objet et le POUSSE, puis
    repart : démonstration du principe de contact. Le bébé la voit et entend le choc, mais ne la SENT pas (ni toucher, ni bras) et
    elle ne suit pas ses commandes -> ce n'est pas sa main. Teinte peau, un peu plus grande. PAR (n,T,2) = position (examens).
    calm (0..1) : MONDE LISIBLE / COMPLEXITÉ PROGRESSIVE (idées user) — après un contact actif, le bébé S'ARRÊTE POUR REGARDER (proba
    0.8·calm, 3–6 images) ; il fait aussi des pauses sans contact (0.35·calm des changements de geste) ; les objets sont AU REPOS par
    défaut (bougent seuls avec proba 3 % à calm 1) et s'arrêtent plus vite (frottement 0.008 avec proba calm)."""
    cf = STAGES[stage]; rng = np.random.default_rng(seed); K = 3
    yy, xx = (np.mgrid[0:H, 0:H].astype(np.float32) + 0.5) / H
    X = np.zeros((n, T, H, H, 3), np.float32); A = np.zeros((n, T, a_sub * 2, NB), np.float32)
    TOUCH = np.zeros((n, T, 8), np.float32); PROP = np.zeros((n, T, 4), np.float32); CMD = np.zeros((n, T, 2), np.float32)
    HAND = np.zeros((n, T, 2), np.float32); RH = np.zeros(n, np.float32); INVIEW = np.zeros((n, T), bool)
    POS = np.full((n, T, K, 2), np.nan, np.float32); ANG = np.full((n, T, K), np.nan, np.float32)
    SHAPE = -np.ones((n, K), np.int64); NOBJ = np.zeros(n, np.int64); IMP = np.zeros((n, T), bool); TSRC = np.zeros((n, T), np.int8)
    MOB = np.full((n, T, 2), np.nan, np.float32); LINK = np.zeros((n, T), bool); MTYPE = -np.ones(n, np.int64); VIS = np.zeros((n, T, K), bool)
    PAR = np.full((n, T, 2), np.nan, np.float32); PCONT = np.zeros((n, T), bool)
    win = np.hanning(SPF // a_sub).astype(np.float32); W = band_matrix(SPF // a_sub); tt_all = np.arange(T * SPF) / SR
    for i in range(n):
        Rh = rng.uniform(*cf["hand"]); RH[i] = Rh; Lr = rng.uniform(*cf["reach"]) if cf["reach"] else None
        lo, hi = cf["box"]; lo_c = lo + Rh if lo >= 0 else lo; hi_c = hi - Rh if hi <= 1 else hi

        def place(h):                                                # boîte (champ ou au-delà) puis longueur du bras
            h = np.clip(h, lo_c, hi_c)
            if Lr is not None:
                d = h - SHOULDER; nd = float(np.linalg.norm(d))
                if nd > Lr: h = SHOULDER + d * Lr / nd
            return h.astype(np.float32)

        N = int(rng.integers(cf["nobj"][0], cf["nobj"][1] + 1)); NOBJ[i] = N
        kind = rng.integers(0, 4, N); SHAPE[i, :N] = kind
        s = np.where(kind == 3, rng.uniform(0.12, 0.16, N), rng.uniform(0.08, 0.14, N)); rc = np.where(kind == 3, 0.95, 0.9) * s
        cols = np.clip(PAL[rng.choice(len(PAL), N, replace=False)] * rng.uniform(0.7, 1.0, (N, 1)) + rng.normal(0, 0.08, (N, 3)), 0.1, 0.95).astype(np.float32)
        m = np.exp(rng.uniform(np.log(1 / 3), np.log(3), N)); mat = rng.integers(0, 3, N)
        P = np.zeros((N, 2), np.float32)
        for k in range(N):                                           # 0d : objets À PORTÉE ; 0e : n'importe où (parfois hors de portée)
            for _ in range(200):
                P[k] = rng.uniform(rc[k], 1 - rc[k], 2)
                ok = all(np.linalg.norm(P[k] - P[j]) > rc[k] + rc[j] + 0.03 for j in range(k))
                if ok and (stage != "0d" or np.linalg.norm(P[k] - SHOULDER) < Lr - rc[k] - 0.03): break
        p_mov = 1 - cf["rest"] if calm <= 0 else (1 - calm) * (1 - cf["rest"]) + calm * 0.03   # monde LISIBLE : objets AU REPOS par défaut
        par = N > 0 and p_parent > 0 and rng.random() < p_parent              # un parent viendra montrer comment on pousse
        p_mov = (p_mov if p_auto is None else p_auto) * (0.0 if par else 1.0)  # avec le parent : objets au repos, SEULE sa main les fera bouger
        moving = rng.random(N) < p_mov
        th = rng.uniform(0, 2 * math.pi, N); sp = rng.uniform(smin, smax, N) * moving
        V = np.stack([sp * np.cos(th), sp * np.sin(th)], -1).astype(np.float32)
        ang = rng.uniform(0, 2 * math.pi, N); om = rng.normal(0, 0.15, N) * moving
        fric = (0.008 if calm > 0 and rng.random() < calm else rng.choice([0.0015, 0.004])) if N else 0.0   # calm : l'objet s'arrête dans la séquence
        mg = table if (p_out > 0 and N and rng.random() < p_out) else 0.0                  # marge de la table au-delà du champ
        for _ in range(200):
            out = lo < 0 and rng.random() < 0.3                     # 0b : la main commence parfois HORS du champ
            Hp = place(rng.uniform(lo_c, hi_c, 2) if out else rng.uniform(max(lo_c, 0.05), min(hi_c, 0.95), 2))
            if N == 0 or np.all(np.linalg.norm(P - Hp, axis=1) > rc + Rh + 0.02): break
        if cf["mobile"]:                                             # MOBILE (hochet suspendu) hors de portée
            mt = int(rng.choice(3, p=[0.55, 0.25, 0.20])); tcut = int(rng.integers(5, T - 4))   # tirages FIXES (paires appariées)
            mt = mt if force_mtype is None else force_mtype; tcut = (tcut if force_tcut is None else force_tcut) if mt == 2 else T; MTYPE[i] = mt
            M0 = np.array([rng.uniform(0.25, 0.75), rng.uniform(0.15, 0.3)], np.float32); off = np.zeros(2, np.float32); mv = np.zeros(2, np.float32)
            mkind = int(rng.integers(0, 3)); mcol = PAL[rng.integers(len(PAL))]; mf0 = rng.uniform(850, 1150)
        vact = np.zeros((T, 2), np.float32); mspd = np.zeros(T, np.float32); ev = []
        if par:                                                      # MAIN DU PARENT : arrive d'en face (haut), pousse un objet, repart
            Rp = 0.09; pcol = (np.array([0.95, 0.75, 0.6]) * rng.uniform(0.85, 1.0)).astype(np.float32)
            Pp = np.array([rng.uniform(0.2, 0.8), -0.2], np.float32); Ps = Pp.copy(); pk = int(rng.integers(N))
            a_ = rng.uniform(0, 2 * math.pi); pu = np.array([math.cos(a_), math.sin(a_)], np.float32)     # direction de la poussée
            pwait, ppush, pspd = int(rng.integers(0, 4)), int(rng.integers(2, 5)), rng.uniform(0.03, 0.07); pmode = "attend"
        left, mode, spd, tgt, dirv = 0, "immobile", vmax, None, None
        pg = 0.35 if stage == "0c" else 0.15                         # 0c : plus de gigotage (les coups de pied du mobile)
        for t in range(T):
            if t > 0:
                if left <= 0:                                        # BABILLAGE (modes de v5)
                    left = rng.integers(2, 7); u_ = rng.random(); spd = rng.uniform(0.015, vmax); mode = "tgt"
                    if N > 0 and calm > 0 and rng.random() < 0.35 * calm: mode = "immobile"   # il REGARDE, sans rien faire (monde lisible)
                    elif N > 0 and u_ < p_obj:                          # p_obj : part des gestes VERS un objet (run 2 : plus de contacts)
                        k = rng.integers(N); d = P[k] - Hp; tgt = P[k] + rng.uniform(0.0, 0.2) * d / (np.linalg.norm(d) + 1e-6)
                    else:
                        if N > 0: u_ = 0.45 + (u_ - p_obj) / (1 - p_obj) * 0.55   # le reste garde les proportions d'origine
                        if u_ < 0.6: tgt = rng.uniform(lo_c, hi_c, 2)  # 0b : la cible peut être HORS du champ
                        elif u_ < 0.7: mode = "immobile"
                        elif u_ < 1 - pg: a_ = rng.uniform(0, 2 * math.pi); dirv = np.array([math.cos(a_), math.sin(a_)]); mode = "direction"
                        else: mode = "gigote"
                left -= 1
                if mode == "immobile": a = rng.normal(0, 0.004, 2)
                elif mode == "direction": a = spd * dirv + rng.normal(0, 0.015, 2)
                elif mode == "gigote": a = rng.normal(0, 0.05, 2)
                else: d = tgt - Hp; a = spd * d / max(np.linalg.norm(d), spd) + rng.normal(0, 0.015, 2)
                CMD[i, t] = np.clip(a, -vmax, vmax)
                H0 = Hp.copy(); Hp = place(Hp + CMD[i, t])
                P0 = P.copy(); P = P + V; ang = ang + om
                if kick > 0:                                         # POUSSÉES EXTÉRIEURES (quelqu'un d'autre fait bouger le jouet)
                    for k in range(N):
                        if rng.random() < kick:
                            a_ = rng.uniform(0, 2 * math.pi); dvk = rng.uniform(0.02, 0.07) * np.array([math.cos(a_), math.sin(a_)], np.float32)
                            V[k] = V[k] + dvk; om[k] += rng.normal(0, 0.1); ev.append((t - 0.5, k, float(np.linalg.norm(dvk)) * m[k], float(P[k, 0])))
                for k in range(N):                                   # murs (au bord du champ, ou plus loin si la table déborde)
                    lo_w, hi_w = rc[k] - mg, 1 - rc[k] + mg
                    for dd in range(2):
                        if P[k, dd] < lo_w or P[k, dd] > hi_w:
                            wall = lo_w if P[k, dd] < lo_w else hi_w
                            fr_ = float(np.clip((wall - P0[k, dd]) / (V[k, dd] + 1e-9), 0, 0.999))
                            J = 2 * m[k] * abs(V[k, dd]); V[k, dd] = -V[k, dd]; om[k] += rng.normal(0, 0.05)
                            P[k, dd] = 2 * lo_w - P[k, dd] if P[k, dd] < lo_w else 2 * hi_w - P[k, dd]
                            if J > 1e-4: ev.append((t - 1 + fr_, k, J * att_out(P[k]), P[k, 0]))   # choc hors champ : plus faible
                if par:                                              # le parent agit (pas de toucher ni de bras pour le bébé)
                    if pmode == "attend": pwait -= 1; pmode = "approche" if pwait <= 0 else pmode; Vp = np.zeros(2, np.float32)
                    if pmode == "approche":
                        bpt = P[pk] - pu * (rc[pk] + Rp + 0.02); d_ = bpt - Pp; nd_ = float(np.linalg.norm(d_))
                        Vp = (d_ * min(1.0, 0.08 / max(nd_, 1e-6))).astype(np.float32)
                        if nd_ < 0.03: pmode = "pousse"
                    elif pmode == "pousse": Vp = (pu * pspd).astype(np.float32); ppush -= 1; pmode = "repart" if ppush <= 0 else pmode
                    elif pmode == "repart": d_ = Ps - Pp; Vp = (d_ * min(1.0, 0.08 / max(float(np.linalg.norm(d_)), 1e-6))).astype(np.float32)
                    Pp = Pp + Vp
                    for k in range(N):                               # main du parent -> objet (elle est menée : pas de recul)
                        dv = P[k] - Pp; dist = float(np.linalg.norm(dv)); lim = rc[k] + Rp
                        if 1e-6 < dist < lim:
                            nv = dv / dist; s_ = float((V[k] - Vp) @ nv); PCONT[i, t] = True
                            if s_ < 0:
                                J = -2 * s_ * m[k] * 3 / (3 + m[k]); V[k] += J / m[k] * nv; om[k] += rng.normal(0, 0.1)
                                ev.append((t - 0.5, k, J, float((P[k, 0] + Pp[0]) / 2)))
                            P[k] = np.clip(Pp + nv * lim, rc[k] - mg, 1 - rc[k] + mg)
                vh = Hp - H0
                for k in range(N):                                   # main <-> objet : choc + PEAU + le lourd repousse la main
                    dv = P[k] - Hp; dist = float(np.linalg.norm(dv)); lim = rc[k] + Rh
                    if 1e-6 < dist < lim:
                        nv = dv / dist; vh_n = float(vh @ nv); vo_n = float(V[k] @ nv); s_ = vo_n - vh_n
                        side = (0 if nv[0] > 0 else 1) if abs(nv[0]) >= abs(nv[1]) else (2 if nv[1] > 0 else 3)
                        J = 0.0
                        if s_ < 0:
                            J = -2 * s_ * m[k] / (1 + m[k]); V[k] += J / m[k] * nv; om[k] += rng.normal(0, 0.1)
                            ev.append((t - 0.5, k, J, float((P[k, 0] + Hp[0]) / 2)))
                        TSRC[i, t] = max(TSRC[i, t], 1 if vh_n >= -vo_n else 2)  # qui est allé vers qui : moi (actif) / le monde (passif)
                        ov = lim - dist; w_ = m[k] / (1 + m[k])
                        Hp = place(Hp - nv * ov * w_); P[k] = np.clip(P[k] + nv * ov * (1 - w_), rc[k] - mg, 1 - rc[k] + mg)
                        TOUCH[i, t, 2 * side] += J; TOUCH[i, t, 2 * side + 1] = max(TOUCH[i, t, 2 * side + 1], w_)
                for k in range(N):                                   # objet <-> objet
                    for j in range(k + 1, N):
                        dv = P[k] - P[j]; dist = float(np.linalg.norm(dv))
                        if 1e-6 < dist < rc[k] + rc[j]:
                            nv = dv / dist; s_ = float((V[k] - V[j]) @ nv)
                            if s_ < 0:
                                J = -2 * s_ * m[k] * m[j] / (m[k] + m[j]); V[k] += J / m[k] * nv; V[j] -= J / m[j] * nv
                                om[k] += rng.normal(0, 0.1); om[j] += rng.normal(0, 0.1)
                                xc = float((P[k, 0] + P[j, 0]) / 2); ga = att_out((P[k] + P[j]) / 2); ev += [(t - 0.5, k, J * ga, xc), (t - 0.5, j, J * ga, xc)]
                            push = (rc[k] + rc[j] - dist) / 2
                            P[k] = np.clip(P[k] + push * nv, rc[k] - mg, 1 - rc[k] + mg); P[j] = np.clip(P[j] - push * nv, rc[j] - mg, 1 - rc[j] + mg)
                if fric > 0:
                    v_ = np.linalg.norm(V, axis=1, keepdims=True); V *= np.clip(1 - fric / (v_ + 1e-9), 0, 1)
                    om *= 0.97 if fric < 0.003 else 0.9
                vact[t] = Hp - H0
                if calm > 0 and TSRC[i, t] == 1 and mode != "immobile" and rng.random() < 0.8 * calm:
                    mode, left = "immobile", int(rng.integers(3, 7))         # PAUSE POUR REGARDER ce qui vient de se produire
                if cf["mobile"]:                                     # le ruban : le mobile suit le geste RÉEL (avec retard si test)
                    LINK[i, t] = mt == 0 or (mt == 2 and t < tcut)
                    if mt == 1: drive = rng.normal(0, 0.05, 2) if rng.random() < 0.4 else np.zeros(2)
                    elif LINK[i, t]: drive = 1.0 * vact[t - mobile_delay] if t - mobile_delay >= 1 else np.zeros(2)
                    else: drive = np.zeros(2)
                    mv = 0.75 * mv - 0.12 * off + drive; off = np.clip(off + mv, -0.15, 0.15); mspd[t] = float(np.linalg.norm(mv))
            PROP[i, t] = np.concatenate([Hp, vact[t]]) + rng.normal(0, 0.005, 4)
            img = np.zeros((H, H, 3), np.float32)
            for k in range(N):
                al = shape_alpha(kind[k], s[k], ang[k], P[k, 0], P[k, 1], xx, yy, H)[..., None]
                img = img * (1 - al) + cols[k] * al
            if cf["mobile"]:
                MOB[i, t] = M0 + off; al = shape_alpha(mkind, mob_size, 3.0 * off[0], *MOB[i, t], xx, yy, H)[..., None]
                img = img * (1 - al) + mcol * al
            if par:                                                  # la main du parent (derrière la sienne, plus proche des yeux)
                PAR[i, t] = Pp; ph = np.clip((Rp * 0.85 - np.maximum(abs(xx - Pp[0]), abs(yy - Pp[1]))) * H + 0.5, 0, 1)[..., None]
                img = img * (1 - ph) + pcol * ph
            hx = np.clip((Rh * 0.85 - np.maximum(abs(xx - Hp[0]), abs(yy - Hp[1]))) * H + 0.5, 0, 1)[..., None]
            X[i, t] = img * (1 - hx) + hx; HAND[i, t] = Hp; INVIEW[i, t] = bool(np.all((Hp > 0) & (Hp < 1)))
            POS[i, t, :N] = P; ANG[i, t, :N] = ang; VIS[i, t, :N] = np.all((P > -s[:, None]) & (P < 1 + s[:, None]), axis=-1)
        extra = cont_sound(2 * hum * np.linalg.norm(vact, axis=1) / 0.08, HAND[i, :, 0], T, rng, "froisse")
        if N > 0 and hum > 0: extra = extra + fric_sound(POS[i, :, :N], mat, m, T, hum)          # identique à hum « fric » dans le champ, atténué hors champ
        if cf["mobile"]: extra = extra + cont_sound(2 * hum * mspd / 0.03, MOB[i, :, 0], T, rng, "tinte", mf0)
        A[i] = render_audio(ev, mat, m, T, a_sub, rng, W, win, tt_all, 1, IMP[i], extra)
    return dict(X=X, A=A, TOUCH=TOUCH, PROP=PROP, CMD=CMD, HAND=HAND, RH=RH, INVIEW=INVIEW, POS=POS, ANG=ANG, SHAPE=SHAPE,
                NOBJ=NOBJ, IMP=IMP, TSRC=TSRC, MOB=MOB, LINK=LINK, MTYPE=MTYPE, VIS=VIS, PAR=PAR, PCONT=PCONT)

def baby_view(X, stage):
    """ce que le bébé VOIT à cette étape (numpy, pour les figures) : flou gaussien UNIFORME + gris + contraste réduit."""
    v = VIEW[stage]; X = X.astype(np.float32)
    if v["sigma"] > 0:
        r = int(math.ceil(2.5 * v["sigma"])); x = np.arange(-r, r + 1); k = np.exp(-x ** 2 / (2 * v["sigma"] ** 2)); k /= k.sum()
        for ax in (-3, -2):
            X = np.apply_along_axis(lambda z: np.convolve(np.pad(z, r, mode="edge"), k, "valid"), ax, X)
    X = (1 - v["gray"]) * X + v["gray"] * X.mean(-1, keepdims=True)
    mu = X.mean((-3, -2, -1), keepdims=True); return np.clip(mu + (X - mu) * v["contrast"], 0, 1)

if __name__ == "__main__":
    import time, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = {"0a": "0a BRAS", "0b": "0b MAINS", "0c": "0c CONTINGENCE", "0d": "0d CONTACT", "0e": "0e DISTANCE"}
    W_ = {}
    for st in STAGES:
        t0 = time.time(); w = gen_world0(200, st, seed=1); dt = time.time() - t0; W_[st] = w
        tc = w["TSRC"] > 0; dcmd = np.linalg.norm(w["CMD"] - w["PROP"][..., 2:], axis=-1)[:, 1:]
        msg = (f"{st} : {1000 * dt / 200:.1f} ms/séq | main {w['RH'].mean():.2f} (visible {w['INVIEW'].mean():.0%}) | objets/séq {w['NOBJ'].mean():.1f} "
               f"| frames avec contact {tc.mean():.1%} (actif {(w['TSRC'] == 1).sum()}, passif {(w['TSRC'] == 2).sum()}) "
               f"| |commande − mouvement| contact {dcmd[tc[:, 1:]].mean() if tc.any() else 0:.3f} / libre {dcmd[~tc[:, 1:]].mean():.3f}")
        if st == "0c": msg += f" | mobile relié/autonome/coupé {np.bincount(w['MTYPE'], minlength=3)}"
        if st == "0e":
            far = np.linalg.norm(w["POS"][:, 0] - SHOULDER, axis=-1)
            msg += f" | objets loin de l'épaule (> 0.9) {np.nanmean(far > 0.9):.0%}"
        print(msg, flush=True)
    # FIGURE 1 : le monde à chaque étape — réalité / ce que voit le bébé
    fig, ax = plt.subplots(10, 8, figsize=(11, 14.5))
    for r_, st in enumerate(STAGES):
        w = W_[st]; i = 3 if st != "0d" else int(np.argmax((w["TSRC"] > 0).sum(1))); bv = baby_view(w["X"][i], st)
        for c_ in range(8):
            for rr, im in ((2 * r_, w["X"][i, 2 * c_]), (2 * r_ + 1, bv[2 * c_])):
                ax[rr, c_].imshow(im, vmin=0, vmax=1); ax[rr, c_].set_xticks([]); ax[rr, c_].set_yticks([])
        ax[2 * r_, 0].set_ylabel(f"{names[st]}\nréalité", fontsize=8); ax[2 * r_ + 1, 0].set_ylabel("vu par\nle bébé", fontsize=8)
    for c_ in range(8): ax[0, c_].set_title(f"t={2 * c_}", fontsize=8)
    plt.tight_layout(); plt.savefig("av_world0.png", dpi=85); plt.close()
    # FIGURE 2 : les 4 sens + la volonté, sur une séquence 0c (mobile relié puis coupé) et une 0d (contact)
    w0c, w0d = W_["0c"], W_["0d"]
    i_c = int(np.argmax(w0c["MTYPE"] == 2)); i_d = int(np.argmax((w0d["TSRC"] > 0).sum(1) + 0.01 * w0d["TOUCH"].sum((1, 2))))
    fig, ax = plt.subplots(5, 2, figsize=(13, 11), gridspec_kw=dict(height_ratios=[1.3, 1, 0.8, 1, 1]))
    for c_, (st, w, i) in enumerate((("0c", w0c, i_c), ("0d", w0d, i_d))):
        T = w["X"].shape[1]; fr = [w["X"][i, t] for t in range(0, T, 2)]
        strip = np.concatenate([np.pad(f, ((1, 1), (1, 1), (0, 0)), constant_values=0.4) for f in fr], 1)
        ax[0, c_].imshow(strip); ax[0, c_].axis("off"); ax[0, c_].set_title(f"{names[st]} — VUE (t = 0, 2, …, 14)", fontsize=9)
        a_sub = w["A"].shape[2] // 2; spec = w["A"][i].reshape(T, a_sub, 2, -1).mean(2).reshape(T * a_sub, -1).T
        ax[1, c_].imshow(spec, aspect="auto", origin="lower", extent=(-0.5, T - 0.5, 0, SR / 2), cmap="magma")
        ax[1, c_].set_ylabel("OUÏE\nfréquence (Hz)", fontsize=8)
        ax[2, c_].imshow(w["TOUCH"][i].T, aspect="auto", cmap="Greens", extent=(-0.5, T - 0.5, 7.5, -0.5))
        ax[2, c_].set_yticks(range(8)); ax[2, c_].set_yticklabels([f"{s} {q}" for s in SIDES for q in ("choc", "appui")], fontsize=6)
        ax[2, c_].set_ylabel("TOUCHER", fontsize=8)
        ax[3, c_].plot(np.linalg.norm(w["CMD"][i], axis=-1), "k--", label="COMMANDE (voulu)")
        ax[3, c_].plot(np.linalg.norm(w["PROP"][i, :, 2:], axis=-1), "b", label="SENS DU BRAS (réel)")
        for t in np.where(w["TSRC"][i] > 0)[0]: ax[3, c_].axvspan(t - 0.5, t + 0.5, color="green", alpha=0.2)
        ax[3, c_].set_ylabel("vitesse de la main", fontsize=8); ax[3, c_].legend(fontsize=7)
        if st == "0c":
            ax[4, c_].plot(np.linalg.norm(np.diff(w["MOB"][i], axis=0, prepend=w["MOB"][i][:1]), axis=-1), "m", label="mouvement du mobile")
            ax[4, c_].plot(np.linalg.norm(w["PROP"][i, :, 2:], axis=-1), "b", alpha=0.5, label="mouvement de la main")
            for t in np.where(~w["LINK"][i][1:])[0] + 1: ax[4, c_].axvspan(t - 0.5, t + 0.5, color="red", alpha=0.12)
            ax[4, c_].set_title("rouge = ruban COUPÉ (le geste ne fait plus bouger le mobile)", fontsize=8)
        else:
            ax[4, c_].plot(w["HAND"][i, :, 0], w["HAND"][i, :, 1], "b.-", label="main")
            for k in range(w["NOBJ"][i]): ax[4, c_].plot(w["POS"][i, :, k, 0], w["POS"][i, :, k, 1], ".-", label=f"objet {k}")
            ax[4, c_].set_xlim(0, 1); ax[4, c_].set_ylim(1, 0); ax[4, c_].set_aspect("equal"); ax[4, c_].set_title("trajectoires (vue de dessus)", fontsize=8)
        ax[4, c_].legend(fontsize=7)
        for r_ in (1, 2, 3): ax[r_, c_].set_xlim(-0.5, T - 0.5)
    plt.tight_layout(); plt.savefig("av_world0_sens.png", dpi=85); print("figures -> av_world0.png, av_world0_sens.png")
