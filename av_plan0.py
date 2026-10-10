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
from av_dev_long import _PosReader
from av_phase0 import Baby0, baby_from_cfg, to_tok0, to_np, to_torch, layout, VIEW, T, H

VMAX = 0.1

class Push0:
    """un épisode de la tâche dans la PHYSIQUE du monde 0e (copie de gen_world0, 1 objet au repos, main R 0.07, bras limité)."""
    def __init__(s, ep, a_sub=2, hum=0.15, dmin=0.12, dmax=0.22, fric=0.004, task="objet"):
        s.task = task                                                   # « main » : amener SA MAIN sur un point, pièce VIDE (idée user)
        rng = np.random.default_rng(10_000 + ep); s.rng = np.random.default_rng(20_000 + ep); s.a_sub, s.hum = a_sub, hum
        s.Rh, s.Lr = 0.07, 0.9; s.kind = int(rng.integers(0, 4))
        s.sz = rng.uniform(0.12, 0.16) if s.kind == 3 else rng.uniform(0.08, 0.14); s.rc = (0.95 if s.kind == 3 else 0.9) * s.sz
        s.col = np.clip(PAL[rng.integers(len(PAL))] * rng.uniform(0.7, 1.0) + rng.normal(0, 0.08, 3), 0.1, 0.95).astype(np.float32)
        s.m = float(np.exp(rng.uniform(np.log(1 / 3), np.log(3)))); s.mat = int(rng.integers(0, 3)); s.fric = fric   # frottement « s'arrête vite » (vécu en babillant) : sinon l'objet glisse comme sur la glace
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
        s.t = 0; s.X = np.zeros((T, H, H, 3), np.float32); s.POS = np.zeros((T, 2), np.float32); s.HAND = np.zeros((T, 2), np.float32)
        s.CMD = np.zeros((T, 2), np.float32); s.TOUCH = np.zeros((T, 8), np.float32); s.PROP = np.zeros((T, 4), np.float32)
        s.vact = np.zeros((T, 2), np.float32); s.ev = []; s.yy, s.xx = (np.mgrid[0:H, 0:H].astype(np.float32) + 0.5) / H
        s.PROP[0] = np.concatenate([s.Hp, s.vact[0]]) + s.rng.normal(0, 0.005, 4); s.record(0)
    def place(s, h):
        h = np.clip(h, s.Rh, 1 - s.Rh); d = h - SHOULDER; nd = float(np.linalg.norm(d))
        return (SHOULDER + d * s.Lr / nd if nd > s.Lr else h).astype(np.float32)
    def dist(s):
        if s.task == "toucher": return max(0.0, float(np.linalg.norm(s.Hp - s.P)) - (s.rc + s.Rh))   # écart avant le contact
        if s.task == "bouger": return max(0.0, 0.05 - float(np.linalg.norm(s.P - s.P0)))                 # il reste à le déplacer de…
        if s.task == "direction": return max(0.0, 0.05 - float((s.P - s.P0) @ s.u))                       # …dans la bonne direction
        return float(np.linalg.norm((s.Hp if s.task.startswith("main") else s.P) - s.g))
    def record(s, t, render=True):
        s.POS[t], s.HAND[t] = s.P, s.Hp
        if render:
            al = shape_alpha(s.kind, s.sz, s.ang, s.P[0], s.P[1], s.xx, s.yy, H)[..., None]; img = s.col * al
            hx = np.clip((s.Rh * 0.85 - np.maximum(abs(s.xx - s.Hp[0]), abs(s.yy - s.Hp[1]))) * H + 0.5, 0, 1)[..., None]
            s.X[t] = img * (1 - hx) + hx
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
        if 1e-6 < dist < lim:                                           # main <-> objet (même règle que le monde vécu)
            nv = dv / dist; vh_n = float(vh @ nv); vo_n = float(s.V @ nv); s_ = vo_n - vh_n
            side = (0 if nv[0] > 0 else 1) if abs(nv[0]) >= abs(nv[1]) else (2 if nv[1] > 0 else 3); J = 0.0
            if s_ < 0:
                J = -2 * s_ * m / (1 + m); s.V = s.V + J / m * nv; s.om += s.rng.normal(0, 0.1); s.ev.append((t - 0.5, 0, J, float((s.P[0] + s.Hp[0]) / 2)))
            ov = lim - dist; w_ = m / (1 + m)
            s.Hp = s.place(s.Hp - nv * ov * w_); s.P = np.clip(s.P + nv * ov * (1 - w_), rc, 1 - rc).astype(np.float32)
            s.TOUCH[t, 2 * side] += J; s.TOUCH[t, 2 * side + 1] = max(s.TOUCH[t, 2 * side + 1], w_); s.touched = True
        v_ = float(np.linalg.norm(s.V)); s.V = (s.V * np.clip(1 - s.fric / (v_ + 1e-9), 0, 1)).astype(np.float32)
        s.om *= 0.97 if s.fric < 0.003 else 0.9
        s.vact[t] = s.Hp - H0; s.PROP[t] = np.concatenate([s.Hp, s.vact[t]]) + s.rng.normal(0, 0.005, 4); s.record(t, render)
    def batch(s):
        """vécu jusqu'à t -> lot (1, T, ...) ; le futur est rempli « immobile » (hors contexte de toute façon)."""
        t = s.t; Pp, Hh = s.POS.copy(), s.HAND.copy(); Pp[t + 1:], Hh[t + 1:] = s.POS[t], s.HAND[t]; vact = s.vact.copy(); vact[t + 1:] = 0
        extra = cont_sound(2 * s.hum * np.linalg.norm(vact, axis=1) / 0.08, Hh[:, 0], T, s.rng, "froisse")
        extra = extra + hum_signal(Pp[:, None], np.array([s.mat]), np.array([s.m]), T, s.hum, 1, "fric")
        Ls = SPF // s.a_sub; A = render_audio([e for e in s.ev], np.array([s.mat]), np.array([s.m]), T, s.a_sub, s.rng, band_matrix(Ls),
                                               np.hanning(Ls).astype(np.float32), np.arange(T * SPF) / SR, 1, np.zeros(T, bool), extra)
        w = dict(X=s.X[None], A=A[None], TOUCH=s.TOUCH[None], PROP=s.PROP[None], CMD=s.CMD[None])
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
    if np.linalg.norm(Hp - behind) < 0.02:                              # en place : vitesse de main pour la bonne glissade
        vh = math.sqrt(2 * env.fric * max(dist - 0.005, 0)) * (1 + env.m) / 2; return np.clip(u * max(vh, 0.012), -VMAX, VMAX).astype(np.float32)
    if rel @ u > -(env.rc + env.Rh) * 0.5: side = perp if rel @ perp > 0 else -perp; tgt = P + side * (env.rc + env.Rh + 0.06) - u * 0.05
    else: tgt = behind
    d = tgt - Hp; spd = min(VMAX, float(np.linalg.norm(d))); return np.clip(spd * d / max(np.linalg.norm(d), 1e-6), -VMAX, VMAX).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--episodes", type=int, default=60); p.add_argument("--diag", type=int, default=30)
    p.add_argument("--plan_c", type=int, default=3); p.add_argument("--plan_h", type=int, default=6); p.add_argument("--plan_seg", type=int, default=2)
    p.add_argument("--plan_steps", type=int, default=12, help="gestes par épisode (12 = comme à T=16)"); p.add_argument("--pop", type=int, default=64); p.add_argument("--iters", type=int, default=4)
    p.add_argument("--n_read", type=int, default=6000); p.add_argument("--read_steps", type=int, default=4000)
    p.add_argument("--dump_n", type=int, default=0, help="enregistre les N premiers épisodes MPC (gestes + positions IMAGINÉES) pour les animations")
    p.add_argument("--task", type=str, default="objet", choices=["objet", "main", "main_loin", "toucher", "bouger", "direction"], help="main / main_loin : SA MAIN sur un point (pièce vide, 0.15–0.35 / 0.35–0.6) ; toucher : toucher un objet immobile")
    p.add_argument("--obj_diag", type=int, default=0, help="1 = diag de l'objet dans l'imagination (2b) puis arrêt")
    p.add_argument("--obj_diag_fig", type=str, default="/content/obj_diag.png")
    p.add_argument("--plan_hs", type=str, default="", help="balayage d'horizons pour le MPC, ex. 1,2,6 (même lecteur, un seul chargement)")
    p.add_argument("--subgoal", type=int, default=0, help="1 = SOUS-BUTS (objet : se placer derrière puis pousser ; bouger : aller à l'objet puis pousser)")
    p.add_argument("--subgoals", type=str, default="", help="balayage, ex. 0,1")
    p.add_argument("--behind", type=float, default=0.21, help="sous-but « derrière » : distance au centre de l'objet perçu (> rayon objet + main ≈ 0.2, sinon inatteignable)"); p.add_argument("--sg_tol", type=float, default=0.06)
    p.add_argument("--reach", type=float, default=0.2, help="bouger : sous-but « aller à l'objet » tant que main–objet perçus > reach")
    p.add_argument("--cost_all", type=int, default=0, help="1 = coût moyen sur TOUS les pas du plan (sinon seulement la fin : le planificateur peut « remettre à plus tard »)")
    p.add_argument("--cost_alls", type=str, default="", help="balayage, ex. 0,1")
    p.add_argument("--film", type=int, default=0, help="FILM : une image par geste planifié pour les N premiers épisodes MPC (plan imaginé vs plan exécuté pour de vrai)")
    p.add_argument("--film_dir", type=str, default="/content")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    global T; T = int(cfg.get("T", 16)); import av_phase0; av_phase0.T = T                 # longueur des séquences de l'instantané
    h = T // 2; nf = min(8, T - h)                                                   # contexte frames < h, imaginé h .. h+nf-1 (8 images comme à T=16)
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; md, fr = layout(nv, npf); v = VIEW["0e"]; d = cfg["d"]
    m = baby_from_cfg(cfg, dev)
    m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    print(f"bébé de phase 0, pas {ck['state']['it']} | {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def vis_lat(b, bs=50):                      # latents CIBLES normalisés (= ce que le prédicteur imite) : (n, T, npf, d)
        out = []
        for i in range(0, len(b["X"]), bs):
            tok, _ = to_tok0({k: x[i:i + bs] for k, x in b.items()}, st, v, P_, dev); ix = torch.arange(nv, device=dev).expand(len(tok), -1)
            out.append(F.layer_norm(tgt(tok[:, :nv], ix).float(), (d,)).view(len(tok), T, npf, d).half().cpu())
        return torch.cat(out)
    # ---------- 1 LECTEUR (instrument) : objet + main, monde 0e à UN objet
    w = gen_world0(a.n_read, "0e", T, H, seed=777, **cfg.get("wkw", {})); keep = np.where(w["NOBJ"] == 1)[0]
    wb = to_np([{k: w[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}]); b = to_torch(wb); n = len(keep)
    Y = torch.from_numpy(np.concatenate([w["POS"][keep, :, 0], w["HAND"][keep]], -1)).float()       # (n, T, 4)
    Z = vis_lat(b); ntr = int(0.8 * n); Ztr, Ytr = Z[:ntr].flatten(0, 1), Y[:ntr].flatten(0, 1)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6; torch.manual_seed(0); ro = _PosReader(d, npf, nout=4).to(dev)
    opt = torch.optim.AdamW(ro.parameters(), 3e-4, weight_decay=0.05); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.read_steps)
    for _ in range(a.read_steps):
        bi = torch.randint(0, len(Ztr), (256,)); l = F.mse_loss(ro(Ztr[bi].to(dev).float()), ((Ytr[bi] - mu) / sd).to(dev))
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
    ro.eval(); mu, sd = mu.to(dev), sd.to(dev)
    read = lambda z: ro(z.float()) * sd + mu
    with torch.no_grad():
        pt = torch.cat([read(Z[ntr:].flatten(0, 1)[i:i + 512].to(dev)).cpu() for i in range(0, (n - ntr) * T, 512)])
    e = (pt - Y[ntr:].flatten(0, 1)).view(-1, 2, 2).norm(dim=-1) * 32
    print(f"1 LECTEUR sur vrais latents ({n} séq. à 1 objet) : objet {e[:, 0].mean():.2f} px, main {e[:, 1].mean():.2f} px  ({time.time() - t0:.0f}s)", flush=True)
    # ---------- 2 IMAGINATION DÉCODÉE : contexte 0..7, futur 8..15
    te = slice(ntr, n); bt = {k: x[te] for k, x in b.items()}; Yt = Y[te]; cm, tm = fr < h, (fr >= h) & (fr < h + nf) & (md == 0)
    ci0, ti0 = torch.from_numpy(np.where(cm)[0]).to(dev), torch.from_numpy(np.where(tm)[0]).to(dev)
    def imagined(lie):
        out = []
        with torch.no_grad():
            for i in range(0, len(bt["X"]), 50):
                tok, cmd = to_tok0({k: x[i:i + 50] for k, x in bt.items()}, st, v, P_, dev); B = len(tok)
                if lie: cmd = cmd.roll(1, 0)
                ci, ti = ci0.expand(B, -1), ti0.expand(B, -1)
                zp = m.pred(m.enc(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci), ti, cmd, ci).float()
                out.append(read(zp.view(B * nf, npf, d)).view(B, nf, 4).cpu())
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
                    out.append(m.pred(m.enc(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci), ti, cmd, ci).float().view(B, nf, npf, d).half().cpu())
            return torch.cat(out)
        Ztr_i = imag_lat({k: x[:ntr] for k, x in b.items()}).flatten(0, 1); Zte_i = imag_lat(bt).flatten(0, 1)
        Ytr_i, Yte_i = Y[:ntr, h:h + nf].flatten(0, 1), Yt[:, h:h + nf].flatten(0, 1)
        torch.manual_seed(0); r2_ = _PosReader(d, npf, nout=4).to(dev); o2 = torch.optim.AdamW(r2_.parameters(), 3e-4, weight_decay=0.05)
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
    # ---------- 3 & 4 : épisodes
    def imag_cost(env, Hh):
        t = env.t; b1 = env.batch(); tok, cmd0 = to_tok0(b1, st, v, P_, dev)
        ci = torch.from_numpy(np.where(fr <= t)[0]).to(dev)[None]
        tis = torch.stack([torch.from_numpy(np.where((fr == t + hh) & (md == 0))[0]) for hh in range(1, Hh + 1)]).to(dev); ti = tis[-1:]
        vi = torch.from_numpy(np.where((fr <= t) & (md == 0))[0]).to(dev)[None]
        with torch.no_grad():
            ctx = m.enc(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci)
            zt = F.layer_norm(tgt(torch.gather(tok, 1, vi.unsqueeze(-1).expand(-1, -1, tok.size(-1))), vi).float(), (d,))[:, -npf:]
            now = read(zt)[0]                                            # objet x,y, main x,y PERÇUS maintenant (perception, pas imagination)
        gt = torch.from_numpy(env.g).to(dev)
        if a.task in ("toucher", "bouger", "direction"): gt = now[:2]   # où est l'objet MAINTENANT
        phase, sub = 0, None                                             # SOUS-BUTS (le bébé décompose : d'abord la main, ensuite l'objet)
        if a.subgoal and a.task in ("objet", "bouger"):
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
        def cost(cand, raw=False):              # cand (K, Hh, 2) gestes bruts -> distance objet–cible IMAGINÉE à t+Hh
            K = len(cand); cmd = cmd0.expand(K, -1, -1).clone(); cmd[:, t + 1:] = 0; cmd[:, t + 1:t + 1 + Hh] = cand / 0.05
            with torch.no_grad():
                if a.cost_all and not raw:                               # coût à CHAQUE pas du plan (arriver tôt, pas « plus tard ») : (K, Hh, 4)
                    r_ = read(m.pred(ctx.expand(K * Hh, -1, -1), tis.repeat(K, 1), cmd.repeat_interleave(Hh, 0), ci.expand(K * Hh, -1))).view(K, Hh, 4)
                else: r_ = read(m.pred(ctx.expand(K, -1, -1), ti.expand(K, -1), cmd, ci.expand(K, -1)))[:, None]
            if raw: return r_[:, -1]                                     # positions imaginées (objet x,y, main x,y) à t+Hh
            if phase == 1: c_ = (r_[..., 2:4] - sub).norm(dim=-1)                              # sous-but : la MAIN au point
            elif a.task == "bouger": c_ = -(r_[..., :2] - gt).norm(dim=-1)                     # imaginer l'objet DÉPLACÉ
            elif a.task == "direction": c_ = -((r_[..., :2] - gt) @ torch.from_numpy(env.u).to(dev))   # …dans la direction demandée
            else: c_ = (r_[..., slice(0, 2) if a.task == "objet" else slice(2, 4)] - gt).norm(dim=-1)   # objet (pousser) ou MAIN (atteindre, toucher)
            return c_.mean(1)
        def traj(plan):                         # plan (Hh, 2) -> positions IMAGINÉES (Hh, 4) à t+1 .. t+Hh (une requête par horizon)
            cmd = cmd0.expand(Hh, -1, -1).clone(); cmd[:, t + 1:] = 0; cmd[:, t + 1:t + 1 + Hh] = plan[None] / 0.05
            with torch.no_grad(): return read(m.pred(ctx.expand(Hh, -1, -1), tis, cmd, ci.expand(Hh, -1)))
        cost.traj, cost.now, cost.phase, cost.sub, cost.gt = traj, now, phase, sub, gt
        return cost
    def real_cost(env, cand):
        out = []
        for cc in cand.cpu().numpy():
            e2 = copy.deepcopy(env)
            for ac in cc: e2.step(ac, render=False)
            out.append(e2.dist())
        return np.array(out)
    has_obj = not a.task.startswith("main")
    def qty(p4, ref):                           # la quantité visée par la tâche (pour le film) : distance au but / déplacement
        if a.task == "objet": return float(np.linalg.norm(p4[:2] - ref))
        if a.task in ("bouger", "direction"): return float(np.linalg.norm(p4[:2] - ref))
        return float(np.linalg.norm(p4[2:] - ref))
    def run(ep, policy, rec=None, stats=None, film=None):
        env = Push0(ep, task=a.task); rng = np.random.default_rng(30_000 + ep); d0 = env.dist()
        for _ in range(a.plan_c): env.step(np.clip(rng.normal(0, 0.03, 2), -VMAX, VMAX))
        while env.t < min(T - 1, a.plan_c + a.plan_steps):          # même nombre de gestes quelle que soit T
            Hh = min(a.plan_h, T - 1 - env.t)
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
            elif policy == "oracle": act = oracle(env)
            else:
                cost = imag_cost(env, Hh); K = a.plan_seg if a.plan_seg > 0 else Hh; rep = -(-Hh // K)
                mu_, sd_ = torch.zeros(K, 2, device=dev), torch.full((K, 2), 0.06, device=dev)
                for _ in range(a.iters):
                    seg = (mu_ + sd_ * torch.randn(a.pop, K, 2, device=dev)).clamp(-VMAX, VMAX); cand = seg.repeat_interleave(rep, 1)[:, :Hh]
                    el = seg[cost(cand).argsort()[:max(4, a.pop // 8)]]; mu_, sd_ = el.mean(0), el.std(0) + 0.01
                act = mu_[0].cpu().numpy()
                if rec is not None:                                      # ANIMATION : où le bébé IMAGINE que tout sera à la fin de son plan
                    plan = mu_.repeat_interleave(rep, 0)[:Hh]; gh = cost(plan[None], raw=True)[0].cpu().numpy()
                    rec.append(dict(t=int(env.t), act=[float(x) for x in act], ghost=[float(x) for x in gh], Hh=int(Hh)))
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
        ok = env.touched if a.task == "toucher" else (env.dist() <= 0 if a.task in ("bouger", "direction") else env.dist() < succ)
        return d0, env.dist(), float(ok)
    succ = 0.04 if a.task.startswith("main") else 0.06
    if a.diag > 0:
        tp = time.time(); R = [run(e, "diag") for e in range(a.diag)]
        print(f"3 DIAG (c) {a.diag} situations, 48 gestes par famille, horizon {a.plan_h} : l'imagination classe-t-elle comme la réalité ? ({time.time() - tp:.0f}s)")
        print(f"   {'famille':>7s} | {'corr. rang':>10s} | {'réel du choix':>13s} | {'meilleur réel':>13s} | {'médiane réelle':>14s} | {'gestes qui rapprochent':>22s}")
        for fam in ("CEM", "bébé", "tous"):
            x = np.array([r[fam] for r in R], dtype=np.float64)
            print(f"   {fam:>7s} | {np.nanmean(x[:, 0]):+10.2f} | {x[:, 1].mean():13.3f} | {x[:, 2].mean():13.3f} | {x[:, 3].mean():14.3f} | {x[:, 4].mean():22.0%}", flush=True)
    titre = {"main": "amener SA MAIN sur un point (pièce vide, 0.15–0.35)", "main_loin": "amener SA MAIN sur un point LOINTAIN (0.35–0.6)",
             "toucher": "TOUCHER un objet immobile (départ à 0.25–0.45)", "objet": "pousser l'objet sur une cible à 0.12–0.22",
             "bouger": "FAIRE BOUGER l'objet (≥ 0.05)", "direction": "le POUSSER dans une direction donnée (≥ 0.05 vers gauche/droite/haut/bas)"}[a.task]
    print(f"4 PLANIFICATION : {titre} ({a.episodes} épisodes, {min(T - 1, a.plan_c + a.plan_steps) - a.plan_c} gestes, CEM {a.pop}×{a.iters}, horizon {a.plan_h}, segments {a.plan_seg})")
    print(f"   {'politique':>10s} | {'distance finale':>15s} | {('contact atteint' if a.task == 'toucher' else 'objet déplacé' if a.task in ('bouger', 'direction') else 'réussite (< ' + str(succ) + ')'):>17s} | {'progrès moyen':>13s}")
    for pol in ("hasard", "oracle"):
        tp = time.time(); res = np.array([run(e, pol) for e in range(a.episodes)])
        print(f"   {pol:>10s} | {res[:, 1].mean():15.3f} | {np.mean(res[:, 2]):17.0%} | {np.mean(res[:, 0] - res[:, 1]):+13.3f}  ({time.time() - tp:.0f}s)", flush=True)
    hs = [int(x) for x in a.plan_hs.split(",")] if a.plan_hs else [a.plan_h]
    sgs = [int(x) for x in a.subgoals.split(",")] if a.subgoals else [a.subgoal]
    cas = [int(x) for x in a.cost_alls.split(",")] if a.cost_alls else [a.cost_all]
    for h_, sg, ca in [(h_, sg, ca) for h_ in hs for sg in sgs for ca in cas]:
        if True:
            a.plan_h, a.subgoal, a.cost_all = h_, sg, ca; nm = f"MPC h={h_}" + (" +ss-buts" if sg else "") + (" +coût-tous-pas" if ca else "")
            recs = [[] for _ in range(a.dump_n)]; films = [[] for _ in range(a.film)]; S = []
            tp = time.time(); res = np.array([run(e, "MPC", recs[e] if e < a.dump_n else None, S, films[e] if e < a.film else None) for e in range(a.episodes)])
            print(f"   {nm:>10s} | {res[:, 1].mean():15.3f} | {np.mean(res[:, 2]):17.0%} | {np.mean(res[:, 0] - res[:, 1]):+13.3f}  ({time.time() - tp:.0f}s)", flush=True)
            S = np.array(S, dtype=np.float64)                             # (h, err objet, err main, copie objet, copie main, l'objet bouge, phase)
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
                             {"objet": "objet→cible", "bouger": "déplacement objet", "direction": "déplacement objet"}.get(a.task, "main→but"))
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
