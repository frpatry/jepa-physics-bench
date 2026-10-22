"""
PLANIFICATION avec le bébé de PHASE 0 (le test qui compte) : « pousse l'objet sur la cible », jamais appris.
Le bébé a seulement babillé (av_phase0.py). Ici, il IMAGINE les conséquences de séquences de gestes avec son modèle du
monde (encodeur des 4 sens gelé + prédicteur conditionné par la COMMANDE, en une passe : contexte = tout ce qu'il a vécu
jusqu'à t, requêtes = vue de la frame t+h), choisit la meilleure (CEM), joue le 1er geste, regarde, recommence (MPC).
Coût = distance objet–cible lue par un LECTEUR de positions (instrument supervisé, comme en phase 2 ; entraîné sur les
latents cibles normalisés = l'espace que le prédicteur imite).

Sorties :
  1 LECTEUR : précision sur les vrais latents (px), main et objet ;
  2 IMAGINATION DÉCODÉE : contexte frames 0..7, futur 8..15 imaginé avec SES gestes vs ceux d'une autre séquence vs copie ;
  3 DIAG (c) : l'imagination classe-t-elle des gestes candidats comme la VRAIE physique ? (corrélation de rang) ;
  4 PLANIFICATION : hasard / oracle (connaît l'état) / MPC du bébé — distance finale, réussite (< 0.06), progrès.
Repères phase 2 (autre monde, 2 disques) : hasard 0.201 / 2 %, oracle 0.145 / 18 %, MPC 0.175–0.18 / 0–5 %, état exact 0.121 / 27 %.

  python av_plan0.py --ckpt /content/drive/MyDrive/jepa_runs/phase0.pt
"""
import argparse, copy, math, time
import numpy as np, torch, torch.nn.functional as F
from av_jepa import PAL, SPF, SR, band_matrix, render_audio, hum_signal
from av_world0 import gen_world0, cont_sound, SHOULDER
from av_world5 import shape_alpha
from av_shape import contact as shape_contact
from av_dev_long import _PosReader
from av_phase0 import Baby0, baby_from_cfg, to_tok0, to_np, to_torch, layout, VIEW, T, H

VMAX = 0.1

class LocalReader(torch.nn.Module):
    """LECTEUR LOCAL (instrument adapté à un œil local) : chaque case lit dans SON latent « la chose est-elle chez moi ? » (score) et
    « où exactement dans ma case ? » (décalage) ; la position = moyenne des centres de cases + décalages, pondérée par les scores
    (softmax). Rien n'est mélangé entre cases. Même interface que _PosReader : sorties NORMALISÉES ((pos − mu) / sd)."""
    def __init__(s, d, npf, nout=4, h=128, mu=None, sd=None):
        super().__init__(); nP = int(round(npf ** 0.5)); s.nP, s.k = nP, nout // 2
        s.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU())
        s.score, s.off = torch.nn.Linear(h, s.k), torch.nn.Linear(h, 2 * s.k)
        r, c = torch.meshgrid(torch.arange(nP), torch.arange(nP), indexing="ij")
        s.register_buffer("ctr", torch.stack([(c.flatten() + 0.5) / nP, (r.flatten() + 0.5) / nP], -1).float())   # (npf, 2) x, y
        s.register_buffer("mu", torch.zeros(nout) if mu is None else mu.float().clone()); s.register_buffer("sd", torch.ones(nout) if sd is None else sd.float().clone())
    def forward(s, x):
        f = s.f(x.float()); w = s.score(f).softmax(1)                                   # (B, npf, k) : où est chaque chose
        off = torch.tanh(s.off(f)).view(*f.shape[:2], s.k, 2) * (0.5 / s.nP)            # décalage dans la case
        pos = (w.unsqueeze(-1) * (s.ctr[None, :, None] + off)).sum(1).reshape(len(x), -1)   # (B, 2k)
        return (pos - s.mu) / s.sd

class Push0:
    """un épisode de la tâche dans la PHYSIQUE du monde 0e (copie de gen_world0, 1 objet au repos, main R 0.07, bras limité)."""
    def __init__(s, ep, a_sub=2, hum=0.15, dmin=0.12, dmax=0.22, fric=0.004, task="objet", blocks=False, damp=0.0, echo=0.0, L=None, shape_phys=False):
        s.task, s.blocks, s.bowl, s.shape_phys = task, blocks, task == "bol", shape_phys or task == "pusht"
        s.damp, s.echo, s.vib = damp, echo, np.zeros(2, np.float32)   # MÊME monde que le vécu (remarque user : pas de patinoire si le bébé a appris sur une table)                            # « main » : amener SA MAIN sur un point, pièce VIDE (idée user)
        rng = np.random.default_rng(10_000 + ep); s.rng = np.random.default_rng(20_000 + ep); s.a_sub, s.hum = a_sub, hum
        s.Rh, s.Lr = 0.07, 0.9; s.kind = 3 if task == "pusht" else int(rng.integers(0, 4))   # PUSH-T : toujours un T
        s.sz = rng.uniform(0.12, 0.16) if s.kind == 3 else rng.uniform(0.08, 0.14); s.rc = (0.95 if s.kind == 3 else 0.9) * s.sz
        s.col = np.clip(PAL[rng.integers(len(PAL))] * rng.uniform(0.7, 1.0) + rng.normal(0, 0.08, 3), 0.1, 0.95).astype(np.float32)
        s.m = float(np.exp(rng.uniform(np.log(1 / 3), np.log(3)))); s.mat = int(rng.integers(0, 3)); s.fric = fric   # frottement « s'arrête vite » (vécu en babillant) : sinon l'objet glisse comme sur la glace
        if blocks: s.m = float(np.exp(np.log(s.m) * np.log(1.25) / np.log(3)))   # BLOC : masse ~constante (mêmes tirages aléatoires)
        for _ in range(500):                                            # objet À PORTÉE, avec de la place autour pour pousser
            s.P = rng.uniform(s.rc + 0.05, 1 - s.rc - 0.05, 2).astype(np.float32)
            if np.linalg.norm(s.P - SHOULDER) < s.Lr - s.rc - 0.12: break
        s.V = np.zeros(2, np.float32); s.ang = float(rng.uniform(0, 2 * math.pi)); s.om = 0.0
        for _ in range(500):                                            # la main démarre PRÈS de l'objet
            s.Hp = s.place(rng.uniform(s.Rh, 1 - s.Rh, 2)); dd = np.linalg.norm(s.P - s.Hp)
            if s.rc + s.Rh + 0.03 < dd < 0.3: break
        for _ in range(500):                                            # cible à 0.12–0.22 de l'objet, dans le champ et à portée
            a_ = rng.uniform(0, 2 * math.pi); s.g = (s.P + rng.uniform(dmin, dmax) * np.array([math.cos(a_), math.sin(a_)])).astype(np.float32)
            if np.all((s.g > s.rc + 0.02) & (s.g < 1 - s.rc - 0.02)) and np.linalg.norm(s.g - SHOULDER) < s.Lr - 0.05: break
        s.touched = False
        if task.startswith("main"):                                     # pièce vide : l'objet est rangé loin (invisible, inerte)
            s.P = np.array([5.0, 5.0], np.float32); lo, hi = (0.35, 0.6) if task == "main_loin" else (0.15, 0.35)
            for _ in range(2000):
                s.Hp = s.place(rng.uniform(0.15, 0.85, 2)); a_ = rng.uniform(0, 2 * math.pi)
                s.g = (s.Hp + rng.uniform(lo, hi) * np.array([math.cos(a_), math.sin(a_)])).astype(np.float32)
                if np.all((s.g > 0.12) & (s.g < 0.88)) and np.linalg.norm(s.g - SHOULDER) < s.Lr - 0.03: break
        s.P0 = s.P.copy(); s.u = np.zeros(2, np.float32)
        if task in ("bouger", "direction"):                             # FAIRE BOUGER l'objet (n'importe comment / dans une direction donnée)
            for _ in range(2000):
                s.Hp = s.place(rng.uniform(s.Rh, 1 - s.Rh, 2)); dd = float(np.linalg.norm(s.P - s.Hp))
                if s.rc + s.Rh + 0.03 < dd < 0.3: break
            a_ = rng.integers(0, 4) * math.pi / 2; s.u = np.array([math.cos(a_), math.sin(a_)], np.float32)   # gauche/droite/haut/bas
            s.g = (s.P + 0.5 * s.u).astype(np.float32)                       # (repère de l'oracle : pousser franchement dans la direction)
        if task == "toucher":                                           # TOUCHER un objet immobile : la main part à 0.25–0.45 de lui
            for _ in range(2000):
                s.Hp = s.place(rng.uniform(s.Rh, 1 - s.Rh, 2)); dd = float(np.linalg.norm(s.P - s.Hp))
                if 0.25 < dd < 0.45: break
            s.g = s.P.copy()
        s.g_ang = s.ang                                                 # PUSH-T : orientation voulue = tournée de 30–90° (à gauche ou à droite)
        if task == "pusht": s.g_ang = float(s.ang + rng.choice([-1.0, 1.0]) * rng.uniform(math.pi / 6, math.pi / 2))
        L = s.L = L or T                                                # durée de l'épisode (> T : ÉPISODE LONG, le bébé voit une fenêtre de T images qui glisse)
        s.t = 0; s.X = np.zeros((L, H, H, 3), np.float32); s.POS = np.zeros((L, 2), np.float32); s.HAND = np.zeros((L, 2), np.float32)
        s.CMD = np.zeros((L, 2), np.float32); s.TOUCH = np.zeros((L, 8), np.float32); s.PROP = np.zeros((L, 4), np.float32)
        s.vact = np.zeros((L, 2), np.float32); s.ev = []; s.yy, s.xx = (np.mgrid[0:H, 0:H].astype(np.float32) + 0.5) / H
        s.PROP[0] = np.concatenate([s.Hp, s.vact[0]]) + s.rng.normal(0, 0.005, 4); s.record(0)
    def place(s, h):
        h = np.clip(h, s.Rh, 1 - s.Rh); d = h - SHOULDER; nd = float(np.linalg.norm(d))
        return (SHOULDER + d * s.Lr / nd if nd > s.Lr else h).astype(np.float32)
    def dist(s):
        if s.task == "toucher": return max(0.0, float(np.linalg.norm(s.Hp - s.P)) - (s.rc + s.Rh))   # écart avant le contact
        if s.task == "bouger": return max(0.0, 0.05 - float(np.linalg.norm(s.P - s.P0)))                 # il reste à le déplacer de…
        if s.task == "direction": return max(0.0, 0.05 - float((s.P - s.P0) @ s.u))                       # …dans la bonne direction
        if s.task == "pusht": return float(np.linalg.norm(s.P - s.g)) + 0.15 * s.ang_err()   # position + 0.15 × angle (rad)
        return float(np.linalg.norm((s.Hp if s.task.startswith("main") else s.P) - s.g))
    def ang_err(s):
        return abs((s.ang - s.g_ang + math.pi) % (2 * math.pi) - math.pi)
    def frame(s, P, Hp, ang=None):
        """image du monde avec l'objet en P et la main en Hp (le BOL, s'il y en a un, est un anneau gris fixe autour de la cible)."""
        img = np.zeros((H, H, 3), np.float32)
        if s.bowl:
            r_ = np.sqrt((s.xx - s.g[0]) ** 2 + (s.yy - s.g[1]) ** 2); rb = s.rc + 0.035
            ring = np.clip(1.0 - np.abs(r_ - rb) * H / 0.9, 0, 1)[..., None]; img = img * (1 - ring) + 0.45 * ring
        al = shape_alpha(s.kind, s.sz, s.ang if ang is None else ang, P[0], P[1], s.xx, s.yy, H)[..., None]; img = img * (1 - al) + s.col * al
        hx = np.clip((s.Rh * 0.85 - np.maximum(abs(s.xx - Hp[0]), abs(s.yy - Hp[1]))) * H + 0.5, 0, 1)[..., None]
        return img * (1 - hx) + hx
    def record(s, t, render=True):
        s.POS[t], s.HAND[t] = s.P, s.Hp
        if render: s.X[t] = s.frame(s.P, s.Hp)
    def step(s, cmd, render=True):
        t = s.t = s.t + 1; rc, m = s.rc, s.m; s.CMD[t] = np.clip(cmd, -VMAX, VMAX)
        H0 = s.Hp.copy(); s.Hp = s.place(s.Hp + s.CMD[t])
        if s.task.startswith("main"):                                   # pièce vide : seule la main bouge
            s.vact[t] = s.Hp - H0; s.PROP[t] = np.concatenate([s.Hp, s.vact[t]]) + s.rng.normal(0, 0.005, 4); s.record(t, render); return
        P0 = s.P.copy(); s.P = s.P + s.V; s.ang += s.om
        for dd in range(2):                                             # murs
            if s.P[dd] < rc or s.P[dd] > 1 - rc:
                wall = rc if s.P[dd] < rc else 1 - rc; fr_ = float(np.clip((wall - P0[dd]) / (s.V[dd] + 1e-9), 0, 0.999))
                J = 2 * m * abs(s.V[dd]); s.V[dd] = -s.V[dd]; s.om += s.rng.normal(0, 0.05)
                s.P[dd] = 2 * rc - s.P[dd] if s.P[dd] < rc else 2 * (1 - rc) - s.P[dd]
                if J > 1e-4: s.ev.append((t - 1 + fr_, 0, J, s.P[0]))
        vh = s.Hp - H0; dv = s.P - s.Hp; dist = float(np.linalg.norm(dv)); lim = rc + s.Rh
        c_ = shape_contact(s.kind, s.sz, s.ang, s.P, s.V, s.om, m, s.Hp, vh, s.Rh) if s.shape_phys else None
        if (c_ is not None) if s.shape_phys else (1e-6 < dist < lim):  # main <-> objet (même règle que le monde vécu)
            nv = c_["nv"] if s.shape_phys else dv / dist; vh_n = float(vh @ nv); vo_n = float(s.V @ nv); s_ = vo_n - vh_n
            side = (0 if nv[0] > 0 else 1) if abs(nv[0]) >= abs(nv[1]) else (2 if nv[1] > 0 else 3); J = 0.0
            if s.shape_phys and c_["J"] > 0:                            # PHYSIQUE DE FORME : impulsion au point de contact (rotation)
                J = c_["J"]; s.V, s.om = c_["V"].astype(np.float32), c_["om"]; s.ev.append((t - 0.5, 0, J, float((s.P[0] + s.Hp[0]) / 2)))
                if s.echo > 0: s.vib = s.vib - J * nv
            elif not s.shape_phys and s_ < 0:
                J = -2 * s_ * m / (1 + m); s.V = s.V + J / m * nv; s.om += s.rng.normal(0, 0.1); s.ev.append((t - 0.5, 0, J, float((s.P[0] + s.Hp[0]) / 2)))
                if s.echo > 0: s.vib = s.vib - J * nv                  # le choc repousse le bras (rémanence, comme le monde vécu)
            ov = c_["ov"] if s.shape_phys else lim - dist; w_ = m / (1 + m)
            s.Hp = s.place(s.Hp - nv * ov * w_); s.P = np.clip(s.P + nv * ov * (1 - w_), rc, 1 - rc).astype(np.float32)
            s.TOUCH[t, 2 * side] += J; s.TOUCH[t, 2 * side + 1] = max(s.TOUCH[t, 2 * side + 1], w_); s.touched = True
        v_ = float(np.linalg.norm(s.V)); s.V = (s.V * np.clip(1 - s.fric / (v_ + 1e-9), 0, 1)).astype(np.float32)
        s.om *= 0.97 if s.fric < 0.003 else 0.9
        if s.blocks: s.V = (s.V * 0.25).astype(np.float32); s.om *= 0.5        # BLOC : il ne bouge que poussé
        if s.damp > 0: s.V = (s.V * (1 - s.damp)).astype(np.float32); s.om *= 1 - s.damp   # frottement de table (--damp du vécu)
        if s.echo > 0:                                                  # rémanence du toucher (--echo du vécu)
            tr_ = s.TOUCH[t - 1]
            s.TOUCH[t, 0::2] += np.where(s.echo * tr_[0::2] > 0.003, s.echo * tr_[0::2], 0)
            s.TOUCH[t, 1::2] = np.maximum(s.TOUCH[t, 1::2], np.where(s.echo * tr_[1::2] > 0.05, s.echo * tr_[1::2], 0))
        s.vact[t] = s.Hp - H0; s.PROP[t] = np.concatenate([s.Hp, s.vact[t] + s.vib]) + s.rng.normal(0, 0.005, 4); s.record(t, render)
        s.vib = (-s.echo * s.vib if float(np.abs(s.vib).max()) * s.echo > 0.002 else np.zeros(2)).astype(np.float32)
    def batch(s, s0=0):
        """vécu jusqu'à t -> lot (1, T, ...) des images s0 .. s0+T-1 (FENÊTRE ; s0 = 0 = comme avant) ; le futur est rempli « immobile » (hors contexte)."""
        W = slice(s0, s0 + T); t = s.t - s0
        Pp, Hh = s.POS[W].copy(), s.HAND[W].copy(); Pp[t + 1:], Hh[t + 1:] = s.POS[s.t], s.HAND[s.t]; vact = s.vact[W].copy(); vact[t + 1:] = 0
        extra = cont_sound(2 * s.hum * np.linalg.norm(vact, axis=1) / 0.08, Hh[:, 0], T, s.rng, "froisse")
        extra = extra + hum_signal(Pp[:, None], np.array([s.mat]), np.array([s.m]), T, s.hum, 1, "fric")
        ev = [(e[0] - s0,) + tuple(e[1:]) for e in s.ev if 0 <= e[0] - s0 < T]                  # chocs dans la fenêtre (temps décalés)
        Ls = SPF // s.a_sub; A = render_audio(ev, np.array([s.mat]), np.array([s.m]), T, s.a_sub, s.rng, band_matrix(Ls),
                                               np.hanning(Ls).astype(np.float32), np.arange(T * SPF) / SR, 1, np.zeros(T, bool), extra)
        w = dict(X=s.X[W][None], A=A[None], TOUCH=s.TOUCH[W][None], PROP=s.PROP[W][None], CMD=s.CMD[W][None])
        return to_torch(to_np([w]))

def oracle(env):
    if env.task == "bouger":                                            # foncer dans l'objet
        d = env.P - env.Hp; n_ = float(np.linalg.norm(d)); return np.clip(d / max(n_, 1e-6) * VMAX, -VMAX, VMAX).astype(np.float32)
    if env.task == "toucher":                                           # aller droit vers l'objet jusqu'au contact
        d = env.P - env.Hp; n_ = float(np.linalg.norm(d)); spd = min(VMAX, env.dist() + 0.02)
        return np.clip(d / max(n_, 1e-6) * spd, -VMAX, VMAX).astype(np.float32)
    if env.task.startswith("main"):                                     # aller droit au point (vitesse ∝ distance restante)
        d = env.g - env.Hp; n_ = float(np.linalg.norm(d)); return np.clip(d * min(1.0, VMAX / max(n_, 1e-6)), -VMAX, VMAX).astype(np.float32)
    """connaît l'état ET la physique (masse, frottement) : contourne l'objet, se place derrière, donne UNE poussée dosée pour
    que l'objet s'arrête sur la cible (glissade v²/2f), puis attend qu'il s'arrête avant de corriger."""
    P, g, Hp = env.P, env.g, env.Hp; dg = g - P; dist = float(np.linalg.norm(dg)); u = dg / (dist + 1e-6); perp = np.array([-u[1], u[0]])
    if np.linalg.norm(env.V) > 0.003 or dist < 0.02: return np.zeros(2, np.float32)
    behind = P - u * (env.rc + env.Rh + 0.01); rel = Hp - P
    if np.linalg.norm(Hp - behind) < 0.02 and env.blocks:               # BLOC : pousser en continu, ralentir près de la cible
        return np.clip(u * min(0.08, max(dist - 0.005, 0.008)), -VMAX, VMAX).astype(np.float32)
    if np.linalg.norm(Hp - behind) < 0.02:                              # en place : vitesse de main pour la bonne glissade
        vh = math.sqrt(2 * env.fric * max(dist - 0.005, 0)) * (1 + env.m) / 2; return np.clip(u * max(vh, 0.012), -VMAX, VMAX).astype(np.float32)
    if rel @ u > -(env.rc + env.Rh) * 0.5: side = perp if rel @ perp > 0 else -perp; tgt = P + side * (env.rc + env.Rh + 0.06) - u * 0.05
    else: tgt = behind
    d = tgt - Hp; spd = min(VMAX, float(np.linalg.norm(d))); return np.clip(spd * d / max(np.linalg.norm(d), 1e-6), -VMAX, VMAX).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--episodes", type=int, default=60); p.add_argument("--diag", type=int, default=30)
    p.add_argument("--plan_c", type=int, default=3); p.add_argument("--plan_h", type=int, default=6); p.add_argument("--plan_seg", type=int, default=2)
    p.add_argument("--plan_steps", type=int, default=12, help="gestes par épisode (12 = comme à T=16)");
    p.add_argument("--propose", type=int, default=0, help="GESTES MANQUANTS (bébé entraîné avec --goal_w) : 1 = la tête propose les gestes vers l'image-but "
                   "(maintenant + but -> K gestes) et la recherche CEM PART de cette proposition (l'imagination vérifie) ; 2 = la proposition seule, exécutée "
                   "sans imagination (intuition « système 1 ») ; objet / pusht")
    p.add_argument("--propose_k", type=int, default=0, help="nb de gestes demandés à la tête (0 = son maximum, --goal_k de l'entraînement)")
    p.add_argument("--chain", type=int, default=0, help="1 = IMAGINER PAS À PAS en planifiant (idée user, sans réentraînement) : l'image t+1 depuis le vrai passé, "
                   "puis t+2 en repartant de t+1 IMAGINÉE (tous les sens, remise à l'échelle des latents de contexte), etc. — au lieu des Hh images d'un coup")
    p.add_argument("--stop_success", type=int, default=0, help="1 = l'épisode S'ARRÊTE dès la réussite (règle du vrai Push-T : réussi s'il atteint le but à un moment) ; "
                   "0 = jugé à la fin seulement (ancien) ; journal « gestes jusqu'à la réussite »")
    p.add_argument("--slide", type=int, default=0, help="1 = ÉPISODES LONGS (idée user : un bébé plus jeune a besoin de plus de gestes) : --plan_steps peut dépasser T ; "
                   "à chaque geste le bébé voit ses T dernières images (fenêtre qui glisse) et imagine --plan_h images plus loin, puis replanifie"); p.add_argument("--pop", type=int, default=64); p.add_argument("--iters", type=int, default=4)
    p.add_argument("--n_read", type=int, default=6000); p.add_argument("--read_steps", type=int, default=4000)
    p.add_argument("--dump_n", type=int, default=0, help="enregistre les N premiers épisodes MPC (gestes + positions IMAGINÉES) pour les animations")
    p.add_argument("--task", type=str, default="objet", choices=["pusht", "objet", "main", "main_loin", "toucher", "bouger", "direction", "bol"], help="bol : pousser le BLOC dans le BOL (anneau gris visible autour de la cible) ; main / main_loin : SA MAIN sur un point (pièce vide, 0.15–0.35 / 0.35–0.6) ; toucher : toucher un objet immobile")
    p.add_argument("--obj_diag", type=int, default=0, help="1 = diag de l'objet dans l'imagination (2b) puis arrêt")
    p.add_argument("--obj_diag_fig", type=str, default="/content/obj_diag.png")
    p.add_argument("--plan_hs", type=str, default="", help="balayage d'horizons pour le MPC, ex. 1,2,6 (même lecteur, un seul chargement)")
    p.add_argument("--subgoal", type=int, default=0, help="1 = SOUS-BUTS (objet : se placer derrière puis pousser ; bouger : aller à l'objet puis pousser)")
    p.add_argument("--subgoals", type=str, default="", help="balayage, ex. 0,1")
    p.add_argument("--behind", type=float, default=0.21, help="sous-but « derrière » : distance au centre de l'objet perçu (> rayon objet + main ≈ 0.2, sinon inatteignable)"); p.add_argument("--sg_tol", type=float, default=0.06)
    p.add_argument("--reach", type=float, default=0.2, help="bouger : sous-but « aller à l'objet » tant que main–objet perçus > reach")
    p.add_argument("--cost_all", type=int, default=0, help="1 = coût moyen sur TOUS les pas du plan (sinon seulement la fin : le planificateur peut « remettre à plus tard »)")
    p.add_argument("--cost_alls", type=str, default="", help="balayage, ex. 0,1")
    p.add_argument("--blocks", type=int, default=0, help="1 = tâches dans le MONDE DE BLOCS (option B : l'objet ne bouge que poussé) ; lecteur entraîné avec blocs")
    p.add_argument("--film", type=int, default=0, help="FILM : une image par geste planifié pour les N premiers épisodes MPC (plan imaginé vs plan exécuté pour de vrai)")
    p.add_argument("--film_dir", type=str, default="/content")
    p.add_argument("--reader", type=str, default="attn", choices=["attn", "local"], help="INSTRUMENT de lecture des positions : attn = lecteur attentionnel "
                   "(conçu pour des cases globales) ; local = chaque case dit « la chose est chez moi + décalage » (adapté à un œil local)")
    p.add_argument("--read_level", type=str, default="top", choices=["top", "low", "both"], help="PYRAMIDE : à quel étage on LIT les positions — top = sortie du haut "
                   "(comme avant) ; low = étage fin ; both = les DEUX étages ensemble (général + précis, chaque case = [haut | fin])")
    p.add_argument("--read_on", type=str, default="real", choices=["real", "imag"], help="sur quoi le lecteur de l'IMAGINATION est entraîné : real = vrais latents "
                   "(comme avant) ; imag = latents IMAGINÉS (babillage, vraies positions) — l'imagination peut « écrire » autrement que l'œil ; la perception du présent garde le lecteur du réel")
    p.add_argument("--goal_img", type=int, default=0, help="1 = le BUT est une IMAGE (le monde voulu : objet dans le bol), comparée dans les latents du bébé "
                   "LÀ OÙ LA SCÈNE DOIT CHANGER (cases dont les pixels diffèrent entre maintenant et le but) ; aucun lecteur supervisé dans le coût")
    p.add_argument("--hand_sense", type=str, default="vue", choices=["vue", "bras"], help="AUDIT DES SENS : où le bébé sait où est SA MAIN — vue = lue dans l'image "
                   "(imaginée) ; bras = lue dans le BRAS (proprioception, imaginée par le prédicteur) : planification et calibration utilisent alors la main du bras")
    p.add_argument("--senses", type=int, default=0, help="1 = AUDIT DES SENS (section 2s) : la main lue dans le BRAS imaginé et le CONTACT lu dans le TOUCHER imaginé, "
                   "vs copie et vs les gestes d'un AUTRE (l'imagination de chaque sens dépend-elle des gestes ?)")
    p.add_argument("--instr", type=int, default=0, help="MESURE DE L'INSTRUMENT (tâche main) : N épisodes ; pour des gestes de familles différentes (babillage, "
                   "candidats CEM, gestes extrêmes, plans choisis par le planificateur), main lue par le lecteur dans l'imagination vs dans les vraies images "
                   "vs vraie position, et un lecteur NEUF entraîné sur l'imagination (l'info est-elle là ?) ; puis arrêt")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    wk_ = cfg.get("wkw", {}); print(f"monde des tâches = monde vécu : frottement {wk_.get('damp', 0.0)}, rémanence {wk_.get('echo', 0.0)}", flush=True)
    global T; T = int(cfg.get("T", 16)); import av_phase0; av_phase0.T = T                 # longueur des séquences de l'instantané
    h = T // 2; nf = min(8, T - h)                                                   # contexte frames < h, imaginé h .. h+nf-1 (8 images comme à T=16)
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; md, fr = layout(nv, npf); v = VIEW["0e"]; d = cfg["d"]
    m = baby_from_cfg(cfg, dev)
    m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    print(f"bébé de phase 0, pas {ck['state']['it']} | {time.time() - t0:.0f}s", flush=True)
    if a.read_level != "top": assert int(cfg.get("levels", 0)), "--read_level low/both : bébé à pyramide (--levels) seulement"
    LN = lambda u: F.layer_norm(u.float(), (d,))
    def teachN(x, ix):                                                   # latents NORMALISÉS de l'œil-cible à l'étage lu (dR par case)
        if a.read_level == "top": return LN(tgt(x, ix))
        top_, low_ = tgt(x, ix, levels=True)
        return LN(low_) if a.read_level == "low" else torch.cat([LN(top_), LN(low_)], -1)
    def imagine(*ar, **kw):                                              # imagination à l'étage lu
        if a.read_level == "top": return m.pred(*ar, **kw)
        top_, low_ = m.pred(*ar, ret_low=True, **kw)
        return low_ if a.read_level == "low" else torch.cat([top_.float(), low_.float()], -1)
    dR = 2 * d if a.read_level == "both" else d

    @torch.no_grad()
    def vis_lat(b, bs=50):                      # latents CIBLES normalisés (= ce que le prédicteur imite) : (n, T, npf, d)
        out = []
        for i in range(0, len(b["X"]), bs):
            tok, _ = to_tok0({k: x[i:i + bs] for k, x in b.items()}, st, v, P_, dev); ix = torch.arange(nv, device=dev).expand(len(tok), -1)
            out.append(teachN(tok[:, :nv], ix).view(len(tok), T, npf, dR).half().cpu())
        return torch.cat(out)
    # ---------- 1 LECTEUR (instrument) : objet + main, monde 0e à UN objet
    w = gen_world0(a.n_read, "0e", T, H, seed=777, **{**cfg.get("wkw", {}), **({"blocks": 1.0} if a.blocks else {})}); keep = np.where(w["NOBJ"] == 1)[0]
    wb = to_np([{k: w[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}]); b = to_torch(wb); n = len(keep)
    Y = torch.from_numpy(np.concatenate([w["POS"][keep, :, 0], w["HAND"][keep]], -1)).float()       # (n, T, 4)
    Z = vis_lat(b); ntr = int(0.8 * n); Ztr, Ytr = Z[:ntr].flatten(0, 1), Y[:ntr].flatten(0, 1)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6; torch.manual_seed(0)
    mk_reader = lambda nout, mu_, sd_: (LocalReader(dR, npf, nout, mu=mu_, sd=sd_) if a.reader == "local" else _PosReader(dR, npf, nout=nout)).to(dev)
    ro = mk_reader(4, mu, sd)
    opt = torch.optim.AdamW(ro.parameters(), 3e-4, weight_decay=0.05); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.read_steps)
    for _ in range(a.read_steps):
        bi = torch.randint(0, len(Ztr), (256,)); l = F.mse_loss(ro(Ztr[bi].to(dev).float()), ((Ytr[bi] - mu) / sd).to(dev))
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
    ro.eval(); mu, sd = mu.to(dev), sd.to(dev)
    read = lambda z: ro(z.float()) * sd + mu
    read_now = read                                                      # PERCEPTION du présent : toujours le lecteur du réel
    with torch.no_grad():
        pt = torch.cat([read(Z[ntr:].flatten(0, 1)[i:i + 512].to(dev)).cpu() for i in range(0, (n - ntr) * T, 512)])
    e = (pt - Y[ntr:].flatten(0, 1)).view(-1, 2, 2).norm(dim=-1) * 32
    print(f"1 LECTEUR sur vrais latents ({n} séq. à 1 objet) : objet {e[:, 0].mean():.2f} px, main {e[:, 1].mean():.2f} px  ({time.time() - t0:.0f}s)", flush=True)
    # ---------- 1b SENS DU CORPS : lecteurs de la main dans le BRAS (proprioception) et du CONTACT dans le TOUCHER (instruments, latents de l'œil-cible)
    ia, ic = nv + 2 * T, nv + T                                          # premier indice du bras / du toucher
    rb = rt = None
    if a.hand_sense == "bras" or a.senses:
        assert a.read_level == "top", "audit des sens : --read_level top"
        Zs_ = []
        with torch.no_grad():
            for i in range(0, n, 50):
                tok, _ = to_tok0({k: x[i:i + 50] for k, x in b.items()}, st, v, P_, dev); ix = torch.arange(nv + 3 * T, device=dev).expand(len(tok), -1)
                Zs_.append(LN(tgt(tok, ix))[:, nv:].half().cpu())
        Zs_ = torch.cat(Zs_); Zaud, Ztch, Zarm = Zs_[:, :T].float(), Zs_[:, T:2 * T].float(), Zs_[:, 2 * T:].float()   # (n, T, d) ouïe, toucher, bras
        Yh_ = torch.from_numpy(w["HAND"][keep]).float(); Yc_ = torch.from_numpy((np.abs(w["TOUCH"][keep]).sum(-1) > 0).astype(np.float32))
        def fit(Zx, Yx, nout, bce=False, steps=3000):
            torch.manual_seed(0); net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.GELU(), torch.nn.Linear(128, nout)).to(dev)
            mu_, sd_ = (torch.zeros(nout), torch.ones(nout)) if bce else (Yx.mean(0), Yx.std(0) + 1e-6)
            o_ = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01)
            for _ in range(steps):
                bi = torch.randint(0, len(Zx), (512,)); out = net(Zx[bi].to(dev)); yy = ((Yx[bi] - mu_) / sd_).to(dev)
                l = F.binary_cross_entropy_with_logits(out, yy) if bce else F.mse_loss(out, yy); o_.zero_grad(); l.backward(); o_.step()
            net.eval(); mu_, sd_ = mu_.to(dev), sd_.to(dev)
            return (lambda z: torch.sigmoid(net(z.float()))) if bce else (lambda z: net(z.float()) * sd_ + mu_)
        rb = fit(Zarm[:ntr].flatten(0, 1), Yh_[:ntr].flatten(0, 1).reshape(-1, 2), 2)
        rt = fit(Ztch[:ntr].flatten(0, 1), Yc_[:ntr].flatten(0, 1)[:, None], 1, bce=True)
        # OUÏE : ce qu'on ENTEND — un choc (contact), l'objet qui GLISSE (frottement), la vitesse de SA main (froissement)
        Po_ = torch.from_numpy(w["POS"][keep, :, 0]).float(); dP = torch.cat([torch.zeros_like(Po_[:, :1, 0]), (Po_[:, 1:] - Po_[:, :-1]).norm(dim=-1)], 1)
        Ym_ = (dP * 32 > 0.3).float(); Ys_ = torch.cat([torch.zeros_like(Yh_[:, :1, 0]), (Yh_[:, 1:] - Yh_[:, :-1]).norm(dim=-1)], 1) * 32
        ra_c = fit(Zaud[:ntr].flatten(0, 1), Yc_[:ntr].flatten(0, 1)[:, None], 1, bce=True)
        ra_m = fit(Zaud[:ntr].flatten(0, 1), Ym_[:ntr].flatten(0, 1)[:, None], 1, bce=True)
        ra_s = fit(Zaud[:ntr].flatten(0, 1), Ys_[:ntr].flatten(0, 1)[:, None], 1)
        with torch.no_grad():
            eb = (rb(Zarm[ntr:].flatten(0, 1).to(dev)).cpu() - Yh_[ntr:].flatten(0, 1)).norm(dim=-1) * 32
            pc = rt(Ztch[ntr:].flatten(0, 1).to(dev)).cpu()[:, 0]; yc = Yc_[ntr:].flatten(0, 1)
        acc = lambda p_, y_: float(((p_ > 0.5).float() == y_).float().mean())
        def auc(p_, y_):
            p_, y_ = p_.flatten(), y_.flatten(); pos, neg = p_[y_ > 0.5], p_[y_ < 0.5]
            if len(pos) == 0 or len(neg) == 0: return float("nan")
            if len(pos) * len(neg) > 4e7: pos, neg = pos[torch.randperm(len(pos))[:4000]], neg[torch.randperm(len(neg))[:10000]]
            return float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())
        r2 = lambda p_, y_: float(1 - ((p_.flatten() - y_.flatten()) ** 2).mean() / y_.flatten().var().clamp_min(1e-9))
        with torch.no_grad():
            za_ = Zaud[ntr:].flatten(0, 1).to(dev)
            print(f"1b OUÏE (lecteurs, vraies images) : choc/contact AUC {auc(ra_c(za_).cpu(), Yc_[ntr:]):.2f} | objet qui GLISSE AUC {auc(ra_m(za_).cpu(), Ym_[ntr:]):.2f} "
                  f"(objet en mouvement dans {float(Ym_.mean()):.0%} des images) | vitesse de SA main R² {r2(ra_s(za_).cpu(), Ys_[ntr:]):+.2f}", flush=True)
        print(f"1b SENS DU CORPS (lecteurs) : main lue dans le BRAS {eb.mean():.2f} px (dans la VUE {e[:, 1].mean():.2f} px) | contact lu dans le TOUCHER : "
              f"justesse {acc(pc, yc):.0%} (contact dans {float(yc.mean()):.0%} des images)", flush=True)
    dho = (Y[ntr:, :, :2] - Y[ntr:, :, 2:]).flatten(0, 1).norm(dim=-1); eo = e[:, 0]               # B0 : la PERCEPTION de l'objet selon la distance à la main
    bins = [(0, 0.2, "au contact (< 0.2)"), (0.2, 0.35, "proche (0.2–0.35)"), (0.35, 9, "loin (> 0.35)")]
    print("   B0 objet perçu selon la distance main–objet : " + " | ".join(f"{nm} {eo[(dho >= lo) & (dho < hi)].mean():.2f} px (n={int(((dho >= lo) & (dho < hi)).sum())})" for lo, hi, nm in bins), flush=True)
    pd_ = pt.view(-1, 4); bias = ((pd_[:, :2] - Y[ntr:].flatten(0, 1)[:, :2]) * F.normalize(Y[ntr:].flatten(0, 1)[:, 2:] - Y[ntr:].flatten(0, 1)[:, :2], dim=-1)).sum(-1) * 32
    print("   B0 erreur de l'objet perçu PROJETÉE vers la main (+ = tiré vers la main) : " + " | ".join(f"{nm} {bias[(dho >= lo) & (dho < hi)].mean():+.2f} px" for lo, hi, nm in bins), flush=True)
    # ---------- 2 IMAGINATION DÉCODÉE : contexte 0..7, futur 8..15
    te = slice(ntr, n); bt = {k: x[te] for k, x in b.items()}; Yt = Y[te]; cm, tm = fr < h, (fr >= h) & (fr < h + nf) & (md == 0)
    ci0, ti0 = torch.from_numpy(np.where(cm)[0]).to(dev), torch.from_numpy(np.where(tm)[0]).to(dev)
    def imagined(lie, body=None):
        """body (corps/monde seulement) : "vrai" = état du corps prédit avec SES gestes imposé au monde (même si on ment sur la commande) ;
        "faux" = état du corps d'une AUTRE séquence imposé au monde."""
        out = []
        with torch.no_grad():
            for i in range(0, len(bt["X"]), 50):
                tok, cmd = to_tok0({k: x[i:i + 50] for k, x in bt.items()}, st, v, P_, dev); B = len(tok)
                ci, ti = ci0.expand(B, -1), ti0.expand(B, -1)
                ctx = m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci); kw = {}
                if body is not None:
                    sb = m.pred.body(ctx, ci, cmd)[1]; kw = dict(body_real=sb if body == "vrai" else sb.roll(1, 0))
                if lie: cmd = cmd.roll(1, 0)
                zp = imagine(ctx, ti, cmd, ci, **kw).float()
                out.append(read(zp.view(B * nf, npf, dR)).view(B, nf, 4).cpu())
        return torch.cat(out)
    with torch.no_grad():
        Zt = Z[te]; true_r = torch.cat([read(Zt[:, h:h + nf].flatten(0, 1)[i:i + 512].to(dev)).cpu() for i in range(0, len(Zt) * nf, 512)]).view(-1, nf, 4)
        copy_r = read(Zt[:, h - 1].to(dev)).cpu()[:, None].expand(-1, nf, -1)
    im_t, im_l = imagined(False), imagined(True); tgt_y = Yt[:, h:h + nf]
    moved = (Yt[:, h + nf - 1, :2] - Yt[:, h - 1, :2]).norm(dim=-1) * 32 > 1.5                   # l'objet a bougé (poussé) entre 7 et 15
    def err(pr, sl, k): return float(((pr - tgt_y)[sl].view(-1, nf, 2, 2).norm(dim=-1)[..., k] * 32).mean())
    print(f"2 IMAGINATION DÉCODÉE (px, frames 8–15 ; {int(moved.sum())} séq. où l'objet est poussé) :")
    for nm, k in (("objet (poussé)", 0), ("main", 1)):
        sl = moved if k == 0 else slice(None)
        print(f"   {nm:>15s} | lecture du vrai futur {err(true_r, sl, k):5.2f} | COPIE {err(copy_r, sl, k):5.2f} | imaginé avec SES gestes {err(im_t, sl, k):5.2f} "
              f"| avec les gestes d'un AUTRE {err(im_l, sl, k):5.2f}", flush=True)
    if a.senses:                                # 2s AUDIT DES SENS : chaque sens IMAGINÉ dépend-il des gestes ?
        def imag_sense(i0, lie, bb=None):
            bb = bt if bb is None else bb; out = []; tq_ = torch.arange(i0 + h, i0 + h + nf, device=dev)
            with torch.no_grad():
                for i in range(0, len(bb["X"]), 50):
                    tok, cmd = to_tok0({k: x[i:i + 50] for k, x in bb.items()}, st, v, P_, dev); B = len(tok); ci = ci0.expand(B, -1)
                    if lie: cmd = cmd.roll(1, 0)
                    out.append(m.pred(m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci), tq_.expand(B, -1), cmd, ci).float().cpu())
            return torch.cat(out)                                         # (n_test, nf, d)
        Yh_t, Yc_t = Yh_[te][:, h:h + nf], Yc_[te][:, h:h + nf]
        with torch.no_grad():
            arm = {k: rb(imag_sense(ia, k).flatten(0, 1).to(dev)).cpu().view(-1, nf, 2) for k in (False, True)}
            arm_true = rb(Zarm[te][:, h:h + nf].flatten(0, 1).to(dev)).cpu().view(-1, nf, 2); arm_copy = rb(Zarm[te][:, h - 1].to(dev)).cpu()[:, None].expand(-1, nf, -1)
            tch = {k: rt(imag_sense(ic, k).flatten(0, 1).to(dev)).cpu().view(-1, nf) for k in (False, True)}
            tch_copy = rt(Ztch[te][:, h - 1].to(dev)).cpu()[:, None].expand(-1, nf, -1)[..., 0]
        pxh = lambda pr: float((pr - Yh_t).norm(dim=-1).mean() * 32)
        with torch.no_grad():
            aud = {k: imag_sense(nv, k).flatten(0, 1).to(dev) for k in (False, True)}; aud_copy = Zaud[te][:, h - 1:h].expand(-1, nf, -1).flatten(0, 1).to(dev)
            Ym_t, Ys_t = Ym_[te][:, h:h + nf], Ys_[te][:, h:h + nf]
            oA = lambda f_, y_: " | ".join(f"{nm} {f_(z_):.2f}" for nm, z_ in (("COPIE", aud_copy), ("imaginé avec SES gestes", aud[False]), ("avec les gestes d'un AUTRE", aud[True])))
        Dr = (Yh_t[:, -1] - Yh_[te][:, h - 1]); Di = (arm[False][:, -1] - Yh_[te][:, h - 1]); okm = Dr.norm(dim=-1) > 0.5 / 32
        cosb = float(F.cosine_similarity(Di[okm], Dr[okm], dim=-1).mean())
        print(f"2s AUDIT DES SENS (frames {h}–{h + nf - 1}, {len(Yh_t)} séq.) :")
        print(f"   MAIN dans le BRAS imaginé | lecture du vrai futur {pxh(arm_true):5.2f} px | COPIE {pxh(arm_copy):5.2f} | imaginé avec SES gestes {pxh(arm[False]):5.2f} "
              f"| avec les gestes d'un AUTRE {pxh(arm[True]):5.2f} | direction du déplacement (fin) cos {cosb:+.2f}", flush=True)
        print(f"   CONTACT dans le TOUCHER imaginé (AUC, 0.5 = hasard ; contact dans {float(Yc_t.mean()):.0%} des images) | COPIE {auc(tch_copy, Yc_t):.2f} "
              f"| imaginé avec SES gestes {auc(tch[False], Yc_t):.2f} | avec les gestes d'un AUTRE {auc(tch[True], Yc_t):.2f}", flush=True)
        with torch.no_grad():
            print(f"   OUÏE imaginée — CHOC (AUC) : " + oA(lambda z_: auc(ra_c(z_).cpu(), Yc_t), None), flush=True)
            print(f"   OUÏE imaginée — objet qui GLISSE (AUC) : " + oA(lambda z_: auc(ra_m(z_).cpu(), Ym_t), None), flush=True)
            print(f"   OUÏE imaginée — vitesse de SA main (R²) : " + oA(lambda z_: r2(ra_s(z_).cpu(), Ys_t), None), flush=True)
        # SONDES NEUVES entraînées sur l'IMAGINÉ (séquences d'entraînement), testées sur l'imaginé d'AUTRES séquences :
        # l'info est-elle dans l'imagination (sonde neuve haute) ou la sonde du réel lit-elle mal l'imaginé (sonde neuve >> sonde du réel) ?
        bb_tr = {k: x[:ntr] for k, x in b.items()}
        with torch.no_grad():
            Ia_tr, It_tr, Ib_tr = (imag_sense(i0_, False, bb_tr).flatten(0, 1) for i0_ in (nv, ic, ia))
            It_te, Ib_te = imag_sense(ic, False).flatten(0, 1).to(dev), imag_sense(ia, False).flatten(0, 1).to(dev)
        sl_ = lambda Y_: Y_[:ntr][:, h:h + nf].flatten(0, 1)
        na_c, na_m = fit(Ia_tr, sl_(Yc_)[:, None], 1, bce=True), fit(Ia_tr, sl_(Ym_)[:, None], 1, bce=True)
        na_s, nt_c = fit(Ia_tr, sl_(Ys_)[:, None], 1), fit(It_tr, sl_(Yc_)[:, None], 1, bce=True)
        nb_h = fit(Ib_tr, sl_(Yh_).reshape(-1, 2), 2)
        with torch.no_grad():
            print(f"   SONDES NEUVES sur l'IMAGINÉ ({ntr} séq. -> {len(Yh_t)} autres) : TOUCHER contact AUC {auc(nt_c(It_te).cpu(), Yc_t):.2f} "
                  f"(sonde du réel {auc(tch[False], Yc_t):.2f}) | OUÏE choc AUC {auc(na_c(aud[False]).cpu(), Yc_t):.2f} (réel {auc(ra_c(aud[False]).cpu(), Yc_t):.2f}) "
                  f"| OUÏE objet qui glisse AUC {auc(na_m(aud[False]).cpu(), Ym_t):.2f} (réel {auc(ra_m(aud[False]).cpu(), Ym_t):.2f}, copie {auc(ra_m(aud_copy).cpu(), Ym_t):.2f}) "
                  f"| OUÏE vitesse main R² {r2(na_s(aud[False]).cpu(), Ys_t):+.2f} | BRAS main {float((nb_h(Ib_te).cpu().view(-1, nf, 2) - Yh_t).norm(dim=-1).mean() * 32):.2f} px "
                  f"(réel {pxh(arm[False]):.2f})", flush=True)
    if cfg.get("split", 0):                     # 2d LE MONDE EST-IL EXTÉRIEUR AU CORPS ? (corps/monde séparés)
        dpx = lambda u, w, k: float(((u - w)[..., 2 * k:2 * k + 2].norm(dim=-1) * 32).mean())
        i_lie_v, i_fake = imagined(True, "vrai"), imagined(False, "faux")
        print(f"2d MONDE EXTÉRIEUR : mentir sur la COMMANDE déplace la main imaginée de {dpx(im_l, im_t, 1):.2f} px, l'objet de {dpx(im_l, im_t, 0):.2f} px ; "
              f"en GARDANT le vrai état du corps : main {dpx(i_lie_v, im_t, 1):.2f} px, objet {dpx(i_lie_v, im_t, 0):.2f} px (attendu ≈ 0) | "
              f"FAUX état du corps : main {dpx(i_fake, im_t, 1):.2f} px, objet {dpx(i_fake, im_t, 0):.2f} px", flush=True)
    if a.obj_diag:                              # 2b QUE FAIT L'IMAGINATION DE L'OBJET ? (positions lues, px ; départ = frame 7 lue)
        cos = lambda u, w: float((F.cosine_similarity(u, w, dim=-1)).mean())
        st0 = copy_r[:, 0]; Tdisp, Idisp = (Yt[:, h + nf - 1] - Yt[:, h - 1]) * 32, (im_t[:, -1] - st0) * 32
        still = (Yt[:, h:h + nf, :2] - Yt[:, h - 1:h, :2]).norm(dim=-1).amax(1) * 32 < 0.3
        print(f"2b L'OBJET DANS L'IMAGINATION (frames 7 -> 15) :")
        print(f"   objet POUSSÉ ({int(moved.sum())}) : déplacement réel {Tdisp[moved, :2].norm(dim=-1).mean():.2f} px | imaginé {Idisp[moved, :2].norm(dim=-1).mean():.2f} px "
              f"| cos(imaginé, réel) {cos(Idisp[moved, :2], Tdisp[moved, :2]):+.2f} | cos(objet imaginé, MAIN réelle) {cos(Idisp[moved, :2], Tdisp[moved, 2:]):+.2f} "
              f"| cos(objet réel, main réelle) {cos(Tdisp[moved, :2], Tdisp[moved, 2:]):+.2f}")
        di, dr = (im_t[:, -1, :2] - im_t[:, -1, 2:]).norm(dim=-1) * 32, (Yt[:, h + nf - 1, :2] - Yt[:, h + nf - 1, 2:]).norm(dim=-1) * 32
        print(f"   distance objet–main à la frame 15 : imaginée {di[moved].mean():.2f} px vs réelle {dr[moved].mean():.2f} px (poussé) ; "
              f"{di[still].mean():.2f} vs {dr[still].mean():.2f} (immobile)")
        print(f"   objet IMMOBILE ({int(still.sum())}) : dérive imaginée {Idisp[still, :2].norm(dim=-1).mean():.2f} px (réelle 0) "
              f"| erreur imaginée {err(im_t, still, 0):.2f} vs copie {err(copy_r, still, 0):.2f}")
        eh = lambda pr: ((pr - tgt_y)[moved].view(-1, nf, 2, 2).norm(dim=-1)[..., 0] * 32).mean(0)
        print("   objet poussé, erreur par horizon 1..8 : imaginé " + " ".join(f"{x:.1f}" for x in eh(im_t)) + " | copie " + " ".join(f"{x:.1f}" for x in eh(copy_r)))
        # 2c L'OBJET EST-IL DANS L'IMAGINATION ? lecteur NEUF entraîné sur des latents IMAGINÉS (séquences d'entraînement),
        #    testé sur les imaginés tenus à l'écart : s'il lit bien l'objet -> l'info y est (décalage de distribution du lecteur) ;
        #    sinon -> l'imagination a vraiment perdu l'objet.
        def imag_lat(bb):
            out = []
            with torch.no_grad():
                for i in range(0, len(bb["X"]), 50):
                    tok, cmd = to_tok0({k: x[i:i + 50] for k, x in bb.items()}, st, v, P_, dev); B = len(tok); ci, ti = ci0.expand(B, -1), ti0.expand(B, -1)
                    out.append(imagine(m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci), ti, cmd, ci).float().view(B, nf, npf, dR).half().cpu())
            return torch.cat(out)
        Ztr_i = imag_lat({k: x[:ntr] for k, x in b.items()}).flatten(0, 1); Zte_i = imag_lat(bt).flatten(0, 1)
        Ytr_i, Yte_i = Y[:ntr, h:h + nf].flatten(0, 1), Yt[:, h:h + nf].flatten(0, 1)
        torch.manual_seed(0); r2_ = mk_reader(4, mu, sd); o2 = torch.optim.AdamW(r2_.parameters(), 3e-4, weight_decay=0.05)
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, a.read_steps)
        for _ in range(a.read_steps):
            bi = torch.randint(0, len(Ztr_i), (256,)); l = F.mse_loss(r2_(Ztr_i[bi].to(dev).float()), ((Ytr_i[bi] - mu.cpu()) / sd.cpu()).to(dev))
            o2.zero_grad(); l.backward(); o2.step(); s2.step()
        r2_.eval()
        with torch.no_grad(): p2 = torch.cat([(r2_(Zte_i[i:i + 512].to(dev).float()) * sd + mu).cpu() for i in range(0, len(Zte_i), 512)]).view(-1, nf, 4)
        print(f"2c lecteur NEUF entraîné sur l'IMAGINATION : objet poussé {err(p2, moved, 0):.2f} px, objet immobile {err(p2, still, 0):.2f} px, "
              f"main {err(p2, slice(None), 1):.2f} px (copie : {err(copy_r, moved, 0):.2f} / {err(copy_r, still, 0):.2f} / {err(copy_r, slice(None), 1):.2f})", flush=True)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        idx = torch.where(moved)[0][:8].numpy(); fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
        for j, i in enumerate(idx):
            A_ = ax.flat[j]; A_.imshow(bt["X"][i, h + nf - 1].numpy(), extent=(0, 1, 1, 0), alpha=0.45)
            A_.plot(Yt[i, h - 1:h + nf, 0], Yt[i, h - 1:h + nf, 1], "w.-", lw=2, label="objet réel"); A_.plot(im_t[i, :, 0], im_t[i, :, 1], "r.-", label="objet imaginé")
            A_.plot(Yt[i, h - 1:h + nf, 2], Yt[i, h - 1:h + nf, 3], "c.-", label="main réelle"); A_.plot(im_t[i, :, 2], im_t[i, :, 3], "y.--", label="main imaginée")
            A_.plot(*Yt[i, h - 1, :2], "wo", ms=9, mfc="none"); A_.set_xlim(0, 1); A_.set_ylim(1, 0); A_.set_xticks([]); A_.set_yticks([])
        ax.flat[0].legend(fontsize=7, loc="lower left"); fig.suptitle("frames 7 -> 15 : l'objet poussé, réel (blanc) vs imaginé (rouge) ; la main, réelle (cyan) vs imaginée (jaune) ; image = frame 15")
        plt.tight_layout(); plt.savefig(a.obj_diag_fig, dpi=80); print(f"   figure -> {a.obj_diag_fig}  ({time.time() - t0:.0f}s)", flush=True)
        return
    if a.read_on == "imag":                                              # LECTEUR DE L'IMAGINATION : appris sur des latents IMAGINÉS (babillage)
        tq = time.time(); Zi_ = []
        with torch.no_grad():
            for i in range(0, n, 50):
                tok, cmd = to_tok0({k: x[i:i + 50] for k, x in b.items()}, st, v, P_, dev); B = len(tok); ci, ti = ci0.expand(B, -1), ti0.expand(B, -1)
                Zi_.append(imagine(m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci), ti, cmd, ci).float().view(B, nf, npf, dR).half().cpu())
        Zi_ = torch.cat(Zi_); Yi_ = Y[:, h:h + nf]; Zitr, Yitr = Zi_[:ntr].flatten(0, 1), Yi_[:ntr].flatten(0, 1)
        torch.manual_seed(0); ri_ = mk_reader(4, mu, sd); oi = torch.optim.AdamW(ri_.parameters(), 3e-4, weight_decay=0.05); si = torch.optim.lr_scheduler.CosineAnnealingLR(oi, a.read_steps)
        for _ in range(a.read_steps):
            bi = torch.randint(0, len(Zitr), (256,)); l = F.mse_loss(ri_(Zitr[bi].to(dev).float()), ((Yitr[bi] - mu.cpu()) / sd.cpu()).to(dev)); oi.zero_grad(); l.backward(); oi.step(); si.step()
        ri_.eval()
        with torch.no_grad():
            pi_ = torch.cat([(ri_(Zi_[ntr:].flatten(0, 1)[i:i + 512].to(dev).float()) * sd + mu).cpu() for i in range(0, (n - ntr) * nf, 512)])
            pr_ = torch.cat([read(Zi_[ntr:].flatten(0, 1)[i:i + 512].to(dev).float()).cpu() for i in range(0, (n - ntr) * nf, 512)])
        ei = (pi_ - Yi_[ntr:].flatten(0, 1)).view(-1, 2, 2).norm(dim=-1) * 32; er = (pr_ - Yi_[ntr:].flatten(0, 1)).view(-1, 2, 2).norm(dim=-1) * 32
        print(f"LECTEUR DE L'IMAGINATION (étage {a.read_level}, {ntr} séq. d'entraînement, futur {h}..{h + nf - 1}) : main {ei[:, 1].mean():.2f} px, objet {ei[:, 0].mean():.2f} px "
              f"| lecteur du RÉEL sur la même imagination : main {er[:, 1].mean():.2f} px, objet {er[:, 0].mean():.2f} px ({time.time() - tq:.0f}s)", flush=True)
        read = lambda z: ri_(z.float()) * sd + mu                        # l'imagination est lue par SON lecteur ; la perception garde read_now
    # ---------- 3 & 4 : épisodes
    def imag_cost(env, Hh):
        s0 = max(0, env.t + Hh - (T - 1)) if a.slide else 0                # FENÊTRE : ses T dernières images, avec la place pour imaginer Hh images
        t = env.t - s0; b1 = env.batch(s0); tok, cmd0 = to_tok0(b1, st, v, P_, dev)
        ci = torch.from_numpy(np.where(fr <= t)[0]).to(dev)[None]
        tis = torch.stack([torch.from_numpy(np.where((fr == t + hh) & (md == 0))[0]) for hh in range(1, Hh + 1)]).to(dev); ti = tis[-1:]
        vi = torch.from_numpy(np.where((fr <= t) & (md == 0))[0]).to(dev)[None]
        with torch.no_grad():
            ctx = m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci)
            zt = teachN(torch.gather(tok, 1, vi.unsqueeze(-1).expand(-1, -1, tok.size(-1))), vi)[:, -npf:]
            now = read_now(zt)[0]                                            # objet x,y, main x,y PERÇUS maintenant (perception, pas imagination)
            if rb is not None and a.hand_sense == "bras":                  # PROPRIOCEPTION : la main SENTIE maintenant (bras), pas vue
                ai_ = torch.arange(ia, ia + t + 1, device=dev)[None]
                now = now.clone(); now[2:] = rb(LN(tgt(torch.gather(tok, 1, ai_.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ai_))[:, -1])[0]
        tib = torch.arange(ia + t + 1, ia + t + Hh + 1, device=dev)[:, None]      # bras imaginé à t+1 .. t+Hh (une requête par horizon)
        def arm_hand(K, cmd, all_h):              # (K, Hh ou 1, 2) : la main lue dans le BRAS imaginé
            if all_h: return rb(imagine(ctx.expand(K * Hh, -1, -1), tib.repeat(K, 1), cmd.repeat_interleave(Hh, 0), ci.expand(K * Hh, -1))[:, 0]).view(K, Hh, 2)
            return rb(imagine(ctx.expand(K, -1, -1), tib[-1:].expand(K, -1), cmd, ci.expand(K, -1))[:, 0])[:, None]
        gt = torch.from_numpy(env.g).to(dev)
        if a.task in ("toucher", "bouger", "direction"): gt = now[:2]   # où est l'objet MAINTENANT
        phase, sub = 0, None                                             # SOUS-BUTS (le bébé décompose : d'abord la main, ensuite l'objet)
        if a.subgoal and a.task in ("objet", "bouger", "pusht"):
            ob, hd = now[:2], now[2:]
            if a.task == "bouger": phase, sub = (1, ob) if float((hd - ob).norm()) > a.reach else (2, None)
            else:
                dg = gt - ob; u = dg / (dg.norm() + 1e-6); perp = torch.stack([-u[1], u[0]]); behind = ob - u * a.behind; rel = hd - ob
                dbh = float((hd - behind).norm()); env.ph2 = getattr(env, "ph2", False) and dbh < 2.5 * a.sg_tol or dbh <= a.sg_tol   # hystérésis : la perception de l'objet tremble
                if not env.ph2:
                    phase = 1                                            # se placer DERRIÈRE (en contournant si la main est du mauvais côté)
                    if float(rel @ u) > -0.5 * a.behind: s_ = perp if float(rel @ perp) > 0 else -perp; sub = ob + s_ * (a.behind + 0.04) - u * 0.05
                    else: sub = behind
                else: phase = 2
        if a.goal_img:                          # BUT = IMAGE : la suite « le monde saute au but et y reste » encodée par l'œil-cible (comme une cible d'entraînement)
            Xg = env.X[s0:s0 + T].copy(); gf = env.frame(env.g, env.Hp, env.g_ang); Xg[t + 1:] = gf
            bg = dict(b1); bg["X"] = torch.from_numpy((Xg * 255).round().astype(np.uint8))[None]
            with torch.no_grad():
                tkg, _ = to_tok0(bg, st, v, P_, dev); ixg = torch.arange(nv, device=dev)[None]
                zg_all = teachN(tkg[:, :nv], ixg).view(T, npf, dR)
            dpx = np.abs(gf - env.X[env.t]).max(-1).reshape(nP, P_, nP, P_).max((1, 3)).reshape(-1)        # où la scène doit changer
            gmask = torch.from_numpy(dpx > 0.15).to(dev)
            if not bool(gmask.any()): gmask = torch.ones(npf, dtype=torch.bool, device=dev)
        fidx = [torch.from_numpy(np.where(fr == t + hh)[0]).to(dev) for hh in range(1, Hh + 1)]   # TOUS les jetons de l'image t+hh (vue d'abord)
        cs_, cm_ = ctx.float().std(-1).mean(), ctx.float().mean(-1).mean()   # échelle des latents de CONTEXTE (l'imaginé est à l'échelle normalisée)
        def chain(K, cmd):                      # --chain : IMAGINER PAS À PAS (idée user) — image t+1 depuis le vrai passé, puis t+2 en repartant de t+1
            cx, cid, out = ctx.expand(K, -1, -1), ci.expand(K, -1), []   # IMAGINÉE (tous les sens), etc. -> [(K, npf, dR) vue à t+1 .. t+Hh]
            for hh in range(1, Hh + 1):
                o = imagine(cx, fidx[hh - 1][None].expand(K, -1), cmd, cid); out.append(o[:, :npf])
                if hh < Hh: cx = torch.cat([cx, (o.float() * cs_ + cm_).to(cx.dtype)], 1); cid = torch.cat([cid, fidx[hh - 1][None].expand(K, -1)], 1)
            return out
        def gcost(zi, hh):                      # zi (K, npf, dR) imaginé à t+hh -> écart au but, sur les cases qui doivent changer
            return ((zi.float() - zg_all[t + hh][None]) ** 2).sum(-1)[:, gmask].mean(-1)
        def cost(cand, raw=False):              # cand (K, Hh, 2) gestes bruts -> distance objet–cible IMAGINÉE à t+Hh
            K = len(cand); cmd = cmd0.expand(K, -1, -1).clone(); cmd[:, t + 1:] = 0; cmd[:, t + 1:t + 1 + Hh] = cand / 0.05
            with torch.no_grad():
                if a.cost_all and not raw:                               # coût à CHAQUE pas du plan (arriver tôt, pas « plus tard ») : (K, Hh, 4)
                    r_ = read(imagine(ctx.expand(K * Hh, -1, -1), tis.repeat(K, 1), cmd.repeat_interleave(Hh, 0), ci.expand(K * Hh, -1))).view(K, Hh, 4)
                elif a.chain: r_ = read(chain(K, cmd)[-1])[:, None]
                else: r_ = read(imagine(ctx.expand(K, -1, -1), ti.expand(K, -1), cmd, ci.expand(K, -1)))[:, None]
                if rb is not None and a.hand_sense == "bras": r_ = r_.clone(); r_[..., 2:4] = arm_hand(K, cmd, a.cost_all and not raw)
            if raw: return r_[:, -1]                                     # positions imaginées (objet x,y, main x,y) à t+Hh
            if a.goal_img and phase != 1:       # (avec --subgoal : d'abord AMENER LA MAIN derrière l'objet — l'image-but ne guide pas une main loin du T)
                with torch.no_grad(): return gcost((chain(K, cmd)[-1] if a.chain else imagine(ctx.expand(K, -1, -1), ti.expand(K, -1), cmd, ci.expand(K, -1))).view(K, npf, dR), Hh)
            if phase == 1: c_ = (r_[..., 2:4] - sub).norm(dim=-1)                              # sous-but : la MAIN au point
            elif a.task == "bouger": c_ = -(r_[..., :2] - gt).norm(dim=-1)                     # imaginer l'objet DÉPLACÉ
            elif a.task == "direction": c_ = -((r_[..., :2] - gt) @ torch.from_numpy(env.u).to(dev))   # …dans la direction demandée
            else: c_ = (r_[..., slice(0, 2) if a.task in ("objet", "pusht") else slice(2, 4)] - gt).norm(dim=-1)   # objet (pousser) ou MAIN (atteindre, toucher)
            return c_.mean(1)
        def traj(plan):                         # plan (Hh, 2) -> positions IMAGINÉES (Hh, 4) à t+1 .. t+Hh (une requête par horizon)
            cmd = cmd0.expand(Hh, -1, -1).clone(); cmd[:, t + 1:] = 0; cmd[:, t + 1:t + 1 + Hh] = plan[None] / 0.05
            with torch.no_grad():
                r_ = torch.cat([read(z_) for z_ in chain(1, cmd[:1])]) if a.chain else read(imagine(ctx.expand(Hh, -1, -1), tis, cmd, ci.expand(Hh, -1)))
                if rb is not None and a.hand_sense == "bras": r_ = r_.clone(); r_[:, 2:4] = rb(imagine(ctx.expand(Hh, -1, -1), tib, cmd, ci.expand(Hh, -1))[:, 0])
                return r_
        cost.traj, cost.now, cost.phase, cost.sub, cost.gt = traj, now, phase, sub, gt
        return cost
    Kp_ = (a.propose_k or m.goal.kmax) if a.propose else 0
    if a.propose: assert hasattr(m, "goal"), "--propose : bébé entraîné avec --goal_w (tête GESTES MANQUANTS)"; assert a.task in ("objet", "pusht"), "--propose : objet / pusht"
    def propose(env):                                                    # GESTES MANQUANTS : maintenant (passé seul) + image-but Kp plus loin -> Kp gestes bruts
        s0p = max(0, env.t + Kp_ - (T - 1)); tp = env.t - s0p; b1 = env.batch(s0p)
        Xg = env.X[s0p:s0p + T].copy(); Xg[tp + Kp_] = env.frame(env.g, env.Hp, getattr(env, "g_ang", None))   # le BUT à l'image t+Kp (comme --goal_img)
        bg = dict(b1); bg["X"] = torch.from_numpy((Xg * 255).round().astype(np.uint8))[None]
        tok, _ = to_tok0(b1, st, v, P_, dev); tkg, _ = to_tok0(bg, st, v, P_, dev)
        cg_np = np.where(fr <= tp)[0]; cg = torch.from_numpy(cg_np).to(dev)[None]
        vpos = torch.from_numpy(np.searchsorted(cg_np, np.arange(tp * npf, (tp + 1) * npf))).to(dev); apos = int(np.searchsorted(cg_np, ia + tp))
        gi = torch.arange((tp + Kp_) * npf, (tp + Kp_ + 1) * npf, device=dev)[None]
        with torch.no_grad():
            zc = m.enc(torch.gather(tok, 1, cg.unsqueeze(-1).expand(-1, -1, tok.size(-1))), cg)
            zg = m.enc(torch.gather(tkg, 1, gi.unsqueeze(-1).expand(-1, -1, tkg.size(-1))), gi)
            pr = m.goal(zc[:, vpos], zg, zc[:, apos:apos + 1], Kp_)[0, :Kp_].float() * 0.05      # commandes normalisées -> gestes bruts (cmd = geste / 0.05)
        return pr.clamp(-VMAX, VMAX)
    def real_cost(env, cand):
        out = []
        for cc in cand.cpu().numpy():
            e2 = copy.deepcopy(env)
            for ac in cc: e2.step(ac, render=False)
            out.append(e2.dist())
        return np.array(out)
    def oracle_real(env, Hh=6, pop=48, iters=3, K=2):
        """ORACLE PUSH-T : le même planificateur (CEM) mais qui essaie ses gestes dans le VRAI monde (copies) = plafond d'un modèle parfait."""
        Hh = min(Hh, env.L - 1 - env.t); rep_ = -(-Hh // K); mu_, sd_ = np.zeros((K, 2)), np.full((K, 2), 0.06); rg = np.random.default_rng(env.t)
        for _ in range(iters):
            seg = np.clip(mu_ + sd_ * rg.normal(size=(pop, K, 2)), -VMAX, VMAX); cand = np.repeat(seg, rep_, 1)[:, :Hh]
            el = seg[np.argsort(real_cost(env, torch.from_numpy(cand)))[:max(4, pop // 8)]]; mu_, sd_ = el.mean(0), el.std(0) + 0.01
        return mu_[0].astype(np.float32)
    has_obj = not a.task.startswith("main")
    def qty(p4, ref):                           # la quantité visée par la tâche (pour le film) : distance au but / déplacement
        if a.task in ("objet", "bol", "pusht"): return float(np.linalg.norm(p4[:2] - ref))
        if a.task in ("bouger", "direction"): return float(np.linalg.norm(p4[:2] - ref))
        return float(np.linalg.norm(p4[2:] - ref))
    def run(ep, policy, rec=None, stats=None, film=None):
        env = Push0(ep, task=a.task, blocks=bool(a.blocks), damp=wk_.get("damp", 0.0), echo=wk_.get("echo", 0.0), shape_phys=bool(wk_.get("shape_phys", False)), L=EL + (max(a.plan_h, Kp_) if a.slide else 0)); rng = np.random.default_rng(30_000 + ep); d0 = env.dist()
        for _ in range(a.plan_c): env.step(np.clip(rng.normal(0, 0.03, 2), -VMAX, VMAX))
        def ok_now():
            if a.task == "pusht": return float(np.linalg.norm(env.P - env.g)) < 0.06 and env.ang_err() < math.radians(20)
            return env.touched if a.task == "toucher" else (env.dist() <= 0 if a.task in ("bouger", "direction") else env.dist() < succ)
        while env.t < min(EL - 1, a.plan_c + a.plan_steps):         # même nombre de gestes quelle que soit T
            if a.stop_success and ok_now(): break                    # réussi : l'épisode s'arrête (règle du vrai Push-T)
            Hh = a.plan_h if a.slide else min(a.plan_h, T - 1 - env.t)
            if policy == "diag":
                from scipy.stats import spearmanr
                cost = imag_cost(env, Hh); torch.manual_seed(ep); K = 48
                cem = (torch.randn(K, Hh, 2, device=dev) * 0.06).clamp(-VMAX, VMAX)
                an = torch.rand(K, 1, device=dev) * 2 * math.pi; spd = 0.03 + 0.07 * torch.rand(K, 1, device=dev)
                bb = (spd * torch.cat([an.cos(), an.sin()], -1)).unsqueeze(1).expand(K, Hh, 2).contiguous()
                res = {}
                for fam, cand in (("CEM", cem), ("bébé", bb), ("tous", torch.cat([cem, bb]))):
                    ci_, cr_ = cost(cand).cpu().numpy(), real_cost(env, cand)
                    res[fam] = (spearmanr(ci_, cr_).correlation if cr_.std() > 1e-6 else np.nan, cr_[ci_.argmin()], cr_.min(), np.median(cr_), (cr_ < env.dist() - 0.01).mean())
                return res
            if policy == "hasard": act = rng.uniform(-VMAX, VMAX, 2)
            elif policy == "oracle": act = oracle_real(env) if a.task == "pusht" else oracle(env)
            elif a.propose == 2: act = propose(env)[0].cpu().numpy()        # INTUITION seule : le 1er geste proposé, sans imaginer
            else:
                cost = imag_cost(env, Hh); K = a.plan_seg if a.plan_seg > 0 else Hh; rep = -(-Hh // K)
                mu_, sd_ = torch.zeros(K, 2, device=dev), torch.full((K, 2), 0.06, device=dev)
                if a.propose == 1:                                   # la recherche PART de la proposition (moyenne par segment), plus resserrée
                    pr_ = propose(env); pr_ = torch.cat([pr_, pr_[-1:].expand(max(0, Hh - len(pr_)), -1)])[:Hh]
                    mu_ = torch.stack([pr_[i * rep:(i + 1) * rep].mean(0) for i in range(K)]); sd_ = torch.full((K, 2), 0.04, device=dev)
                for it_ in range(a.iters):
                    seg = (mu_ + sd_ * torch.randn(a.pop, K, 2, device=dev)).clamp(-VMAX, VMAX)
                    if a.propose == 1 and it_ == 0: seg[0] = mu_                 # la proposition elle-même est jugée
                    cand = seg.repeat_interleave(rep, 1)[:, :Hh]
                    el = seg[cost(cand).argsort()[:max(4, a.pop // 8)]]; mu_, sd_ = el.mean(0), el.std(0) + 0.01
                act = mu_[0].cpu().numpy()
                if rec is not None:                                      # ANIMATION : où le bébé IMAGINE que tout sera à la fin de son plan
                    plan = mu_.repeat_interleave(rep, 0)[:Hh]; gh = cost(plan[None], raw=True)[0].cpu().numpy()
                    imt = cost.traj(plan).cpu().numpy(); e3 = copy.deepcopy(env); ret = []           # le plan : IMAGINÉ pas à pas vs EXÉCUTÉ (copie du monde)
                    for ac in plan.cpu().numpy(): e3.step(ac, render=False); ret.append([float(x) for x in np.concatenate([e3.P, e3.Hp])])
                    rec.append(dict(t=int(env.t), act=[float(x) for x in act], ghost=[float(x) for x in gh], Hh=int(Hh),
                                    now=[float(x) for x in cost.now.cpu().numpy()], imag=imt.round(4).tolist(), real=ret))
                if stats is not None or film is not None:                # le plan ENTIER : imaginé pas à pas vs exécuté pour de vrai (copie du monde)
                    plan = mu_.repeat_interleave(rep, 0)[:Hh]; im = cost.traj(plan).cpu().numpy(); now = cost.now.cpu().numpy()
                    e2 = copy.deepcopy(env); re = []
                    for ac in plan.cpu().numpy(): e2.step(ac, render=False); re.append(np.concatenate([e2.P, e2.Hp]))
                    re = np.array(re, np.float32)
                    if stats is not None:
                        for hh in range(Hh):
                            mv = float(np.linalg.norm(re[hh, :2] - env.P)) > 0.01
                            stats.append((hh + 1, np.linalg.norm(im[hh, :2] - re[hh, :2]), np.linalg.norm(im[hh, 2:] - re[hh, 2:]),
                                          np.linalg.norm(now[:2] - re[hh, :2]), np.linalg.norm(now[2:] - re[hh, 2:]), mv, cost.phase))
                    if film is not None:
                        ref = cost.gt.cpu().numpy()
                        film.append(dict(t=int(env.t), X=env.X[env.t].copy(), P=env.P.copy(), Hp=env.Hp.copy(), now=now, im=im, re=re, act=np.asarray(act),
                                         phase=cost.phase, sub=None if cost.sub is None else cost.sub.cpu().numpy(), g=ref,
                                         qi=qty(im[-1], ref), qr=qty(re[-1], env.P if a.task in ("bouger", "direction") else ref)))
            env.step(np.asarray(act, np.float32))
        ok = ok_now()                                                    # PUSH-T : position ET orientation
        if a.stop_success: steps_used.append(env.t - a.plan_c if ok else -1)
        return d0, env.dist(), float(ok)
    steps_used = []                                                     # --stop_success : gestes utilisés par épisode réussi
    EL = a.plan_c + a.plan_steps + 1 if a.slide else T                  # durée d'un épisode (fenêtre qui glisse si > T)
    succ = 0.04 if a.task.startswith("main") else 0.06
    near = 3.0 / H                                                       # critère « PRÈS » (bébé de 4–6 mois : arrive près puis corrige) : à moins de 3 px
    if a.instr > 0:                                                      # ---------- MESURE DE L'INSTRUMENT (lecteur) vs IMAGINATION
        tp = time.time(); Hh = a.plan_h; K = 16; fams = ("babillage", "CEM", "extrêmes", "plans"); rows = {f: [] for f in fams}
        nvv = T * npf
        for ep in range(a.instr):
            env = Push0(1000 + ep, task="main"); rng = np.random.default_rng(40_000 + ep)
            for _ in range(a.plan_c + int(rng.integers(0, 4))): env.step(np.clip(rng.normal(0, 0.03, 2), -VMAX, VMAX))
            Hh_ = min(Hh, T - 1 - env.t); cost = imag_cost(env, Hh_); t0 = env.t
            an = torch.rand(K, 1, device=dev) * 2 * math.pi; spd = 0.03 + 0.07 * torch.rand(K, 1, device=dev)
            cands = {"babillage": (spd * torch.cat([an.cos(), an.sin()], -1)).unsqueeze(1).expand(K, Hh_, 2).contiguous(),
                     "CEM": (torch.randn(K, 2, 2, device=dev) * 0.06).clamp(-VMAX, VMAX).repeat_interleave(-(-Hh_ // 2), 1)[:, :Hh_],
                     "extrêmes": (VMAX * torch.sign(torch.randn(K, 2, 2, device=dev))).repeat_interleave(-(-Hh_ // 2), 1)[:, :Hh_]}
            pl = []
            for _ in range(K // 4):                                     # les plans que le PLANIFICATEUR choisit vraiment (CEM complet)
                Kc = a.plan_seg if a.plan_seg > 0 else Hh_; rep = -(-Hh_ // Kc)
                mu_, sd_ = torch.zeros(Kc, 2, device=dev), torch.full((Kc, 2), 0.06, device=dev)
                for _ in range(a.iters):
                    seg = (mu_ + sd_ * torch.randn(a.pop, Kc, 2, device=dev)).clamp(-VMAX, VMAX); cand = seg.repeat_interleave(rep, 1)[:, :Hh_]
                    el = seg[cost(cand).argsort()[:max(4, a.pop // 8)]]; mu_, sd_ = el.mean(0), el.std(0) + 0.01
                pl.append(mu_.repeat_interleave(rep, 0)[:Hh_])
            cands["plans"] = torch.stack(pl)
            for f in fams:
                for plan in cands[f]:
                    with torch.no_grad():
                        b1 = env.batch(); tok, cmd0 = to_tok0(b1, st, v, P_, dev)
                        ci = torch.from_numpy(np.where(fr <= t0)[0]).to(dev)[None]
                        tis = torch.stack([torch.from_numpy(np.where((fr == t0 + hh) & (md == 0))[0]) for hh in range(1, Hh_ + 1)]).to(dev)
                        ctx = m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci)
                        cmd = cmd0.expand(Hh_, -1, -1).clone(); cmd[:, t0 + 1:] = 0; cmd[:, t0 + 1:t0 + 1 + Hh_] = plan[None] / 0.05
                        zi = imagine(ctx.expand(Hh_, -1, -1), tis, cmd, ci.expand(Hh_, -1)).float()          # (Hh, npf, d) IMAGINÉ
                        e2 = copy.deepcopy(env)
                        for ac in plan.cpu().numpy(): e2.step(ac)
                        tk2, _ = to_tok0(e2.batch(), st, v, P_, dev); ix = torch.arange(nvv, device=dev)[None]
                        zr = teachN(tk2[:, :nvv], ix).view(T, npf, dR)[t0 + 1:t0 + 1 + Hh_]   # VRAIES images
                        ri, rr = read(zi)[:, 2:4].cpu().numpy(), read_now(zr)[:, 2:4].cpu().numpy()
                        if rb is not None and a.hand_sense == "bras":       # la main du BRAS : imaginée vs sentie pour de vrai
                            tib_ = torch.arange(ia + t0 + 1, ia + t0 + Hh_ + 1, device=dev)[:, None]
                            ri = rb(imagine(ctx.expand(Hh_, -1, -1), tib_, cmd, ci.expand(Hh_, -1))[:, 0]).cpu().numpy()
                            rr = rb(LN(tgt(tk2[:, ia:ia + T], torch.arange(ia, ia + T, device=dev)[None]))[0, t0 + 1:t0 + 1 + Hh_]).cpu().numpy()
                    tru = e2.HAND[t0 + 1:t0 + 1 + Hh_]
                    rows[f].append(dict(ep=ep, zi=zi.half().cpu(), zr=zr.half().cpu(), tru=tru.copy(), ri=ri, rr=rr, h0=env.HAND[t0].copy()))
        print(f"MESURE DE L'INSTRUMENT ({a.instr} épisodes « main », {K} gestes par famille, horizon {Hh}) | {time.time() - tp:.0f}s", flush=True)
        print(f"   {'famille':>10s} | {'lecteur sur VRAIES images':>25s} | {'lecteur sur IMAGINATION':>23s} | {'lecteur NEUF sur imagination':>28s} | {'écart latent imag./vrai':>23s} | {'imaginé vs vrai (lecteur)':>25s}")
        for f in fams:
            R_ = rows[f]; tru = np.concatenate([r["tru"] for r in R_]); ri = np.concatenate([r["ri"] for r in R_]); rr = np.concatenate([r["rr"] for r in R_])
            Zi = torch.cat([r["zi"] for r in R_]).float(); Zr = torch.cat([r["zr"] for r in R_]).float(); eps_ = np.concatenate([[r["ep"]] * len(r["tru"]) for r in R_])
            tr_, te_ = eps_ < a.instr // 2, eps_ >= a.instr // 2               # lecteur NEUF : moitié des épisodes pour apprendre, l'autre pour tester
            Y = torch.from_numpy(tru).float(); mu2, sd2 = Y[tr_].mean(0), Y[tr_].std(0) + 1e-6; torch.manual_seed(0)
            r2 = mk_reader(2, mu2, sd2); o2 = torch.optim.AdamW(r2.parameters(), 3e-4, weight_decay=0.05); Ztr, Ytr = Zi[tr_], ((Y[tr_] - mu2) / sd2)
            for _ in range(1500):
                bi = torch.randint(0, len(Ztr), (128,)); l = F.mse_loss(r2(Ztr[bi].to(dev)), Ytr[bi].to(dev)); o2.zero_grad(); l.backward(); o2.step()
            with torch.no_grad(): pn = (r2(Zi[te_].to(dev)).cpu() * sd2 + mu2).numpy()
            px = lambda u, w: float(np.linalg.norm(u - w, axis=-1).mean() * 32)
            dl = float(((Zi - Zr).norm(dim=-1) / Zr.norm(dim=-1).clamp_min(1e-6)).mean())
            print(f"   {f:>10s} | {px(rr, tru):22.2f} px | {px(ri, tru):20.2f} px | {px(pn, tru[te_]):25.2f} px | {dl:20.2f}    | {px(ri, rr):22.2f} px", flush=True)
            cal = []                                                     # CALIBRATION : le déplacement imaginé de la main va-t-il dans le bon sens, de la bonne longueur ?
            for k in (0, len(R_[0]["tru"]) - 1):
                Dr = np.stack([r["tru"][k] - r["h0"] for r in R_]); Di = np.stack([r["ri"][k] - r["h0"] for r in R_]); ok_ = np.linalg.norm(Dr, axis=1) > 0.5 / 32
                cos_ = (Dr * Di).sum(1)[ok_] / (np.linalg.norm(Dr, axis=1)[ok_] * np.linalg.norm(Di, axis=1)[ok_] + 1e-9)
                cal.append(f"h={k + 1} : direction cos {cos_.mean():+.2f}, longueur imaginée/réelle {np.median(np.linalg.norm(Di, axis=1)[ok_] / np.linalg.norm(Dr, axis=1)[ok_]):.2f}")
            print(f"   {'':>10s}   calibration du déplacement de la main : " + " | ".join(cal), flush=True)
        return
    if a.diag > 0:
        tp = time.time(); R = [run(e, "diag") for e in range(a.diag)]
        print(f"3 DIAG (c) {a.diag} situations, 48 gestes par famille, horizon {a.plan_h} : l'imagination classe-t-elle comme la réalité ? ({time.time() - tp:.0f}s)")
        print(f"   {'famille':>7s} | {'corr. rang':>10s} | {'réel du choix':>13s} | {'meilleur réel':>13s} | {'médiane réelle':>14s} | {'gestes qui rapprochent':>22s}")
        for fam in ("CEM", "bébé", "tous"):
            x = np.array([r[fam] for r in R], dtype=np.float64)
            print(f"   {fam:>7s} | {np.nanmean(x[:, 0]):+10.2f} | {x[:, 1].mean():13.3f} | {x[:, 2].mean():13.3f} | {x[:, 3].mean():14.3f} | {x[:, 4].mean():22.0%}", flush=True)
    titre = {"main": "amener SA MAIN sur un point (pièce vide, 0.15–0.35)", "main_loin": "amener SA MAIN sur un point LOINTAIN (0.35–0.6)",
             "toucher": "TOUCHER un objet immobile (départ à 0.25–0.45)", "objet": "pousser l'objet sur une cible à 0.12–0.22",
             "bouger": "FAIRE BOUGER l'objet (≥ 0.05)", "bol": "pousser le BLOC dans le BOL (cible à 0.12–0.22, réussite < 0.06)" + (" — BUT DONNÉ EN IMAGE" if a.goal_img else " — but lu par le lecteur supervisé"), "direction": "le POUSSER dans une direction donnée (≥ 0.05 vers gauche/droite/haut/bas)",
             "pusht": "PUSH-T : amener le T sur une position ET une orientation cibles (réussite : < 0.06 et < 20°)" + (" — BUT DONNÉ EN IMAGE" if a.goal_img else " — but lu par le lecteur (position seulement !)")}[a.task]
    print(f"4 PLANIFICATION : {titre} ({a.episodes} épisodes, {min(EL - 1, a.plan_c + a.plan_steps) - a.plan_c} gestes{' (ÉPISODE LONG, fenêtre qui glisse)' if a.slide else ''}, CEM {a.pop}×{a.iters}, horizon {a.plan_h}, segments {a.plan_seg})")
    print(f"   {'politique':>10s} | {'distance finale':>15s} | {('contact atteint' if a.task == 'toucher' else 'objet déplacé' if a.task in ('bouger', 'direction') else 'réussite (< ' + str(succ) + ')'):>17s} | {'progrès moyen':>13s} | {'près (< 3 px)':>13s}")
    for pol in ("hasard", "oracle"):
        steps_used.clear(); tp = time.time(); res = np.array([run(e, pol) for e in range(a.episodes)])
        print(f"   {pol:>10s} | {res[:, 1].mean():15.3f} | {np.mean(res[:, 2]):17.0%} | {np.mean(res[:, 0] - res[:, 1]):+13.3f} | {np.mean(res[:, 1] < near):13.0%}  ({time.time() - tp:.0f}s)", flush=True)
        if a.stop_success and steps_used: su_ = [x for x in steps_used if x >= 0]; print(f"   {'':>10s}   gestes jusqu'à la réussite : " + (f"médiane {int(np.median(su_))}, min {min(su_)}, max {max(su_)} ({len(su_)} réussites)" if su_ else "aucune réussite"), flush=True)
    hs = [int(x) for x in a.plan_hs.split(",")] if a.plan_hs else [a.plan_h]
    sgs = [int(x) for x in a.subgoals.split(",")] if a.subgoals else [a.subgoal]
    cas = [int(x) for x in a.cost_alls.split(",")] if a.cost_alls else [a.cost_all]
    for h_, sg, ca in [(h_, sg, ca) for h_ in hs for sg in sgs for ca in cas]:
        if True:
            a.plan_h, a.subgoal, a.cost_all = h_, sg, ca; nm = ("INTUITION seule" if a.propose == 2 else f"MPC h={h_}" + (" +ss-buts" if sg else "") + (" +coût-tous-pas" if ca else "") + (" +proposé" if a.propose == 1 else ""))
            recs = [[] for _ in range(a.dump_n)]; films = [[] for _ in range(a.film)]; S = []
            steps_used.clear(); tp = time.time(); res = np.array([run(e, "MPC", recs[e] if e < a.dump_n else None, S, films[e] if e < a.film else None) for e in range(a.episodes)])
            print(f"   {nm:>10s} | {res[:, 1].mean():15.3f} | {np.mean(res[:, 2]):17.0%} | {np.mean(res[:, 0] - res[:, 1]):+13.3f} | {np.mean(res[:, 1] < near):13.0%}  ({time.time() - tp:.0f}s)", flush=True)
            if a.stop_success and steps_used: su_ = [x for x in steps_used if x >= 0]; print(f"   {'':>10s}   gestes jusqu'à la réussite : " + (f"médiane {int(np.median(su_))}, min {min(su_)}, max {max(su_)} ({len(su_)} réussites)" if su_ else "aucune réussite"), flush=True)
            S = np.array(S, dtype=np.float64)                             # (h, err objet, err main, copie objet, copie main, l'objet bouge, phase)
            if S.ndim < 2 or not len(S): continue                         # --propose 2 : rien d'imaginé, pas de dérive à mesurer
            print(f"      DÉRIVE de l'imagination (plan entier exécuté pour de vrai, px) : h | objet qui BOUGE imaginé / copie (n) "
                  f"| objet IMMOBILE imaginé / copie | main imaginée / copie")
            for hh in sorted(set(S[:, 0].astype(int))):
                r = S[S[:, 0] == hh]; mv = r[:, 5] > 0; f_ = lambda x: f"{x.mean() * 32:5.1f}" if len(x) else "  —  "
                print(f"      {hh} | " + (f"{f_(r[mv, 1])} / {f_(r[mv, 3])} ({int(mv.sum()):4d}) | {f_(r[~mv, 1])} / {f_(r[~mv, 3])} | " if has_obj else "")
                      + f"{f_(r[:, 2])} / {f_(r[:, 4])}", flush=True)
            if a.subgoal: print(f"      gestes en phase « main au sous-but » {np.mean(S[S[:, 0] == 1, 6] == 1):.0%}, « pousser » {np.mean(S[S[:, 0] == 1, 6] == 2):.0%}")
            if a.dump_n > 0:
                import json; print("DUMP_JSON " + json.dumps(dict(task=a.task, plan_c=a.plan_c, h=h_, subgoal=sg, cost_all=ca, episodes=[dict(ep=e, steps=r_) for e, r_ in enumerate(recs)])))
            if a.film > 0:
                import pickle; out = f"{a.film_dir}/film_{a.task}_h{h_}{'_ss' if sg else ''}{'_ca' if ca else ''}"
                pickle.dump(dict(films=films, has_obj=has_obj, res=res[:a.film]), open(out + ".pkl", "wb"))
                for i, fl in enumerate(films):
                    film_fig([fl], f"{out}_ep{i}.png", f"{titre} — {nm} — épisode {i} : {'RÉUSSI' if res[i, 2] else 'raté'} (distance finale {res[i, 1]:.2f})", has_obj,
                             {"objet": "objet→cible", "pusht": "T→cible", "bouger": "déplacement objet", "direction": "déplacement objet"}.get(a.task, "main→but"))
                print(f"      films -> {out}_ep*.png", flush=True)
    print(f"total {time.time() - t0:.0f}s")

def film_fig(films, path, title, has_obj, qlab="fin du plan"):
    """une case par geste planifié : image vue AVANT le geste ; pointillés = le plan IMAGINÉ (rouge objet, jaune main),
    traits pleins = le MÊME plan exécuté pour de vrai (orange objet, cyan main) ; flèche blanche = le geste joué ;
    croix verte = but, x magenta = sous-but. Titre : quantité visée imaginée vs réelle à la fin du plan."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    wrap = 6; nc = max(len(f) for f in films); rows_per = -(-nc // wrap); ne = len(films) * rows_per
    fig, ax = plt.subplots(ne, wrap, figsize=(2.6 * wrap, 2.85 * ne + 0.7), squeeze=False)
    for i, fl in enumerate(films):
        for j in range(rows_per * wrap):
            A_ = ax[i * rows_per + j // wrap, j % wrap]; A_.set_xticks([]); A_.set_yticks([])
            if j >= len(fl): A_.axis("off"); continue
            s = fl[j]; A_.imshow(s["X"], extent=(0, 1, 1, 0), interpolation="nearest"); im, re = s["im"], s["re"]
            if has_obj:
                A_.plot(np.r_[s["P"][0], re[:, 0]], np.r_[s["P"][1], re[:, 1]], "-", c="orange", lw=1.3)
                A_.plot(np.r_[s["now"][0], im[:, 0]], np.r_[s["now"][1], im[:, 1]], ":", c="red", lw=1.6, marker=".", ms=3)
            A_.plot(np.r_[s["Hp"][0], re[:, 2]], np.r_[s["Hp"][1], re[:, 3]], "-", c="cyan", lw=1.1)
            A_.plot(np.r_[s["now"][2], im[:, 2]], np.r_[s["now"][3], im[:, 3]], ":", c="yellow", lw=1.6, marker=".", ms=3)
            A_.arrow(s["Hp"][0], s["Hp"][1], 1.5 * s["act"][0], 1.5 * s["act"][1], color="w", width=0.008, head_width=0.04, length_includes_head=True)
            A_.plot(*s["g"], "+", c="lime", ms=9, mew=2)
            if s["sub"] is not None: A_.plot(*s["sub"], "x", c="magenta", ms=7, mew=2)
            ph = {0: "", 1: " · main→ss-but", 2: " · pousse"}[s["phase"]]
            A_.set_title(f"t={s['t']}{ph}\n{qlab} : imaginé {s['qi']:.2f} | réel {s['qr']:.2f}", fontsize=8); A_.set_xlim(0, 1); A_.set_ylim(1, 0)
    fig.suptitle(title + "\npointillés = plan IMAGINÉ (rouge objet, jaune main) · plein = même plan RÉEL (orange objet, cyan main) · flèche = geste joué · + but · x sous-but", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close(fig)

if __name__ == "__main__":
    main()
