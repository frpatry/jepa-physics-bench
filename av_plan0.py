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
from av_phase0 import Baby0, to_tok0, to_np, to_torch, layout, VIEW, T, H

VMAX = 0.1

class Push0:
    """un épisode de la tâche dans la PHYSIQUE du monde 0e (copie de gen_world0, 1 objet au repos, main R 0.07, bras limité)."""
    def __init__(s, ep, a_sub=2, hum=0.15, dmin=0.12, dmax=0.22, fric=0.004):
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
        s.t = 0; s.X = np.zeros((T, H, H, 3), np.float32); s.POS = np.zeros((T, 2), np.float32); s.HAND = np.zeros((T, 2), np.float32)
        s.CMD = np.zeros((T, 2), np.float32); s.TOUCH = np.zeros((T, 8), np.float32); s.PROP = np.zeros((T, 4), np.float32)
        s.vact = np.zeros((T, 2), np.float32); s.ev = []; s.yy, s.xx = (np.mgrid[0:H, 0:H].astype(np.float32) + 0.5) / H
        s.PROP[0] = np.concatenate([s.Hp, s.vact[0]]) + s.rng.normal(0, 0.005, 4); s.record(0)
    def place(s, h):
        h = np.clip(h, s.Rh, 1 - s.Rh); d = h - SHOULDER; nd = float(np.linalg.norm(d))
        return (SHOULDER + d * s.Lr / nd if nd > s.Lr else h).astype(np.float32)
    def dist(s): return float(np.linalg.norm(s.P - s.g))
    def record(s, t, render=True):
        s.POS[t], s.HAND[t] = s.P, s.Hp
        if render:
            al = shape_alpha(s.kind, s.sz, s.ang, s.P[0], s.P[1], s.xx, s.yy, H)[..., None]; img = s.col * al
            hx = np.clip((s.Rh * 0.85 - np.maximum(abs(s.xx - s.Hp[0]), abs(s.yy - s.Hp[1]))) * H + 0.5, 0, 1)[..., None]
            s.X[t] = img * (1 - hx) + hx
    def step(s, cmd, render=True):
        t = s.t = s.t + 1; rc, m = s.rc, s.m; s.CMD[t] = np.clip(cmd, -VMAX, VMAX)
        H0 = s.Hp.copy(); s.Hp = s.place(s.Hp + s.CMD[t]); P0 = s.P.copy(); s.P = s.P + s.V; s.ang += s.om
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
            s.TOUCH[t, 2 * side] += J; s.TOUCH[t, 2 * side + 1] = max(s.TOUCH[t, 2 * side + 1], w_)
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
    p.add_argument("--pop", type=int, default=64); p.add_argument("--iters", type=int, default=4)
    p.add_argument("--n_read", type=int, default=6000); p.add_argument("--read_steps", type=int, default=4000)
    p.add_argument("--obj_diag", type=int, default=0, help="1 = diag de l'objet dans l'imagination (2b) puis arrêt")
    p.add_argument("--obj_diag_fig", type=str, default="/content/obj_diag.png")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; md, fr = layout(nv, npf); v = VIEW["0e"]; d = cfg["d"]
    m = Baby0(cfg["din"], nv, npf, d, cfg["nl"], cfg["nh"], cfg["pred_layers"], bool(cfg["sep"]), cfg.get("inv_head", "attn")).to(dev)
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
    w = gen_world0(a.n_read, "0e", T, H, seed=777); keep = np.where(w["NOBJ"] == 1)[0]
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
    te = slice(ntr, n); bt = {k: x[te] for k, x in b.items()}; Yt = Y[te]; cm, tm = fr <= 7, (fr > 7) & (md == 0)
    ci0, ti0 = torch.from_numpy(np.where(cm)[0]).to(dev), torch.from_numpy(np.where(tm)[0]).to(dev)
    def imagined(lie):
        out = []
        with torch.no_grad():
            for i in range(0, len(bt["X"]), 50):
                tok, cmd = to_tok0({k: x[i:i + 50] for k, x in bt.items()}, st, v, P_, dev); B = len(tok)
                if lie: cmd = cmd.roll(1, 0)
                ci, ti = ci0.expand(B, -1), ti0.expand(B, -1)
                zp = m.pred(m.enc(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci), ti, cmd).float()
                out.append(read(zp.view(B * 8, npf, d)).view(B, 8, 4).cpu())
        return torch.cat(out)
    with torch.no_grad():
        Zt = Z[te]; true_r = torch.cat([read(Zt[:, 8:].flatten(0, 1)[i:i + 512].to(dev)).cpu() for i in range(0, len(Zt) * 8, 512)]).view(-1, 8, 4)
        copy_r = read(Zt[:, 7].to(dev)).cpu()[:, None].expand(-1, 8, -1)
    im_t, im_l = imagined(False), imagined(True); tgt_y = Yt[:, 8:]
    moved = (Yt[:, 15, :2] - Yt[:, 7, :2]).norm(dim=-1) * 32 > 1.5                   # l'objet a bougé (poussé) entre 7 et 15
    def err(pr, sl, k): return float(((pr - tgt_y)[sl].view(-1, 8, 2, 2).norm(dim=-1)[..., k] * 32).mean())
    print(f"2 IMAGINATION DÉCODÉE (px, frames 8–15 ; {int(moved.sum())} séq. où l'objet est poussé) :")
    for nm, k in (("objet (poussé)", 0), ("main", 1)):
        sl = moved if k == 0 else slice(None)
        print(f"   {nm:>15s} | lecture du vrai futur {err(true_r, sl, k):5.2f} | COPIE {err(copy_r, sl, k):5.2f} | imaginé avec SES gestes {err(im_t, sl, k):5.2f} "
              f"| avec les gestes d'un AUTRE {err(im_l, sl, k):5.2f}", flush=True)
    if a.obj_diag:                              # 2b QUE FAIT L'IMAGINATION DE L'OBJET ? (positions lues, px ; départ = frame 7 lue)
        cos = lambda u, w: float((F.cosine_similarity(u, w, dim=-1)).mean())
        st0 = copy_r[:, 0]; Tdisp, Idisp = (Yt[:, 15] - Yt[:, 7]) * 32, (im_t[:, -1] - st0) * 32
        still = (Yt[:, 8:, :2] - Yt[:, 7:8, :2]).norm(dim=-1).amax(1) * 32 < 0.3
        print(f"2b L'OBJET DANS L'IMAGINATION (frames 7 -> 15) :")
        print(f"   objet POUSSÉ ({int(moved.sum())}) : déplacement réel {Tdisp[moved, :2].norm(dim=-1).mean():.2f} px | imaginé {Idisp[moved, :2].norm(dim=-1).mean():.2f} px "
              f"| cos(imaginé, réel) {cos(Idisp[moved, :2], Tdisp[moved, :2]):+.2f} | cos(objet imaginé, MAIN réelle) {cos(Idisp[moved, :2], Tdisp[moved, 2:]):+.2f} "
              f"| cos(objet réel, main réelle) {cos(Tdisp[moved, :2], Tdisp[moved, 2:]):+.2f}")
        di, dr = (im_t[:, -1, :2] - im_t[:, -1, 2:]).norm(dim=-1) * 32, (Yt[:, 15, :2] - Yt[:, 15, 2:]).norm(dim=-1) * 32
        print(f"   distance objet–main à la frame 15 : imaginée {di[moved].mean():.2f} px vs réelle {dr[moved].mean():.2f} px (poussé) ; "
              f"{di[still].mean():.2f} vs {dr[still].mean():.2f} (immobile)")
        print(f"   objet IMMOBILE ({int(still.sum())}) : dérive imaginée {Idisp[still, :2].norm(dim=-1).mean():.2f} px (réelle 0) "
              f"| erreur imaginée {err(im_t, still, 0):.2f} vs copie {err(copy_r, still, 0):.2f}")
        eh = lambda pr: ((pr - tgt_y)[moved].view(-1, 8, 2, 2).norm(dim=-1)[..., 0] * 32).mean(0)
        print("   objet poussé, erreur par horizon 1..8 : imaginé " + " ".join(f"{x:.1f}" for x in eh(im_t)) + " | copie " + " ".join(f"{x:.1f}" for x in eh(copy_r)))
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        idx = torch.where(moved)[0][:8].numpy(); fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
        for j, i in enumerate(idx):
            A_ = ax.flat[j]; A_.imshow(bt["X"][i, 15].numpy(), extent=(0, 1, 1, 0), alpha=0.45)
            A_.plot(Yt[i, 7:, 0], Yt[i, 7:, 1], "w.-", lw=2, label="objet réel"); A_.plot(im_t[i, :, 0], im_t[i, :, 1], "r.-", label="objet imaginé")
            A_.plot(Yt[i, 7:, 2], Yt[i, 7:, 3], "c.-", label="main réelle"); A_.plot(im_t[i, :, 2], im_t[i, :, 3], "y.--", label="main imaginée")
            A_.plot(*Yt[i, 7, :2], "wo", ms=9, mfc="none"); A_.set_xlim(0, 1); A_.set_ylim(1, 0); A_.set_xticks([]); A_.set_yticks([])
        ax.flat[0].legend(fontsize=7, loc="lower left"); fig.suptitle("frames 7 -> 15 : l'objet poussé, réel (blanc) vs imaginé (rouge) ; la main, réelle (cyan) vs imaginée (jaune) ; image = frame 15")
        plt.tight_layout(); plt.savefig(a.obj_diag_fig, dpi=80); print(f"   figure -> {a.obj_diag_fig}  ({time.time() - t0:.0f}s)", flush=True)
        return
    # ---------- 3 & 4 : épisodes
    def imag_cost(env, Hh):
        t = env.t; b1 = env.batch(); tok, cmd0 = to_tok0(b1, st, v, P_, dev)
        ci = torch.from_numpy(np.where(fr <= t)[0]).to(dev)[None]; ti = torch.from_numpy(np.where((fr == t + Hh) & (md == 0))[0]).to(dev)[None]
        with torch.no_grad(): ctx = m.enc(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci)
        gt = torch.from_numpy(env.g).to(dev)
        def cost(cand):                         # cand (K, Hh, 2) gestes bruts -> distance objet–cible IMAGINÉE à t+Hh
            K = len(cand); cmd = cmd0.expand(K, -1, -1).clone(); cmd[:, t + 1:] = 0; cmd[:, t + 1:t + 1 + Hh] = cand / 0.05
            with torch.no_grad(): return (read(m.pred(ctx.expand(K, -1, -1), ti.expand(K, -1), cmd))[:, :2] - gt).norm(dim=-1)
        return cost
    def real_cost(env, cand):
        out = []
        for cc in cand.cpu().numpy():
            e2 = copy.deepcopy(env)
            for ac in cc: e2.step(ac, render=False)
            out.append(e2.dist())
        return np.array(out)
    def run(ep, policy):
        env = Push0(ep); rng = np.random.default_rng(30_000 + ep); d0 = env.dist()
        for _ in range(a.plan_c): env.step(np.clip(rng.normal(0, 0.03, 2), -VMAX, VMAX))
        while env.t < T - 1:
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
            env.step(np.asarray(act, np.float32))
        return d0, env.dist()
    if a.diag > 0:
        tp = time.time(); R = [run(e, "diag") for e in range(a.diag)]
        print(f"3 DIAG (c) {a.diag} situations, 48 gestes par famille, horizon {a.plan_h} : l'imagination classe-t-elle comme la réalité ? ({time.time() - tp:.0f}s)")
        print(f"   {'famille':>7s} | {'corr. rang':>10s} | {'réel du choix':>13s} | {'meilleur réel':>13s} | {'médiane réelle':>14s} | {'gestes qui rapprochent':>22s}")
        for fam in ("CEM", "bébé", "tous"):
            x = np.array([r[fam] for r in R], dtype=np.float64)
            print(f"   {fam:>7s} | {np.nanmean(x[:, 0]):+10.2f} | {x[:, 1].mean():13.3f} | {x[:, 2].mean():13.3f} | {x[:, 3].mean():14.3f} | {x[:, 4].mean():22.0%}", flush=True)
    print(f"4 PLANIFICATION : pousser l'objet sur une cible à 0.12–0.22 ({a.episodes} épisodes, {T - 1 - a.plan_c} gestes, CEM {a.pop}×{a.iters}, horizon {a.plan_h}, segments {a.plan_seg})")
    print(f"   {'politique':>10s} | {'distance finale':>15s} | {'réussite (< 0.06)':>17s} | {'progrès moyen':>13s}")
    for pol in ("hasard", "oracle", "MPC bébé"):
        tp = time.time(); res = np.array([run(e, pol) for e in range(a.episodes)])
        print(f"   {pol:>10s} | {res[:, 1].mean():15.3f} | {np.mean(res[:, 1] < 0.06):17.0%} | {np.mean(res[:, 0] - res[:, 1]):+13.3f}  ({time.time() - tp:.0f}s)", flush=True)
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
