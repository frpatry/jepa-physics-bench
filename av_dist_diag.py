"""
JUGE APPRIS SUR SON PROPRE JEU (plan validé 2026-11-10 ; leçons ARC : « distance au but » + « toc », où le monde a changé).
Le score image-but actuel (écart des latents) est FAUX même avec une imagination parfaite : « sa main au but » vaut « le T au but »
(av_goalcost_diag, duel 52 %). Ici, bébé GELÉ ; une petite tête NEUVE regarde, case par case, (la scène maintenant, l'image-but sans main)
et dit COMBIEN DE GESTES les séparent. Elle apprend sur son JEU LIBRE seulement (aucune démonstration, aucun état d'oracle) :
  deux instants a, b d'une même séquence vécue ; scène = l'image de l'instant a (avec sa main), but = l'image de l'instant b SANS sa main ;
  étiquette = le nombre d'images entre a et le premier instant où le monde (sans sa main) est déjà celui du but — 0 si rien n'a bougé,
  lue dans les images elles-mêmes (écart de pixels ≤ --thr) ; dans les deux sens du temps (après une poussée, sa main est là où était l'objet).
Les latents sont pris comme dans le planificateur : œil-cible, vrai passé ≤ t puis la scène « pour toujours », image t+1.
JUGÉ sur (1) du jeu libre à part, (2) les 8 scènes Push-T fabriquées d'octobre (jamais vues), (3) le PROGRÈS : T à 0, 25, 50, 75, 100 % du chemin.
Repère « pixels » : la même tête sur les cases de pixels au lieu des latents (ce que la tâche permet, sans le bébé).
CRITÈRE (fixé avant le run) : « main au but » bat « T au but » dans ≤ 10 % des situations (52 % aujourd'hui), « T au but » bat le départ
dans ≥ 90 %, et la distance jugée baisse le long du chemin.
  python av_dist_diag.py --ckpt /kaggle/input/.../phase0_rD17n_0e_190k.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_torch, to_np, H, KEYS
from av_world0 import gen_world0, VIEW
import av_plan0
from av_plan0 import Push0, VMAX
from av_judge import DistHead, sep_labels, auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=3000, help="séquences de jeu libre (80 % pour apprendre, 20 % à part)")
    p.add_argument("--n_sit", type=int, default=200, help="situations Push-T fabriquées"); p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=3, help="têtes entraînées (graines)"); p.add_argument("--thr", type=float, default=0.15, help="écart de pixel sous lequel le monde est « le même »")
    p.add_argument("--p_out", type=float, default=0.0, help="part des séquences main hors champ"); p.add_argument("--plan_c", type=int, default=3); p.add_argument("--chunk", type=int, default=48)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t_ = time.time()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    T = int(cfg.get("T", 16)); av_phase0.T = T; av_plan0.T = T
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; d = cfg["d"]; v = VIEW["0e"]; pd = P_ * P_ * 3
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"], strict=False); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    LN = lambda u: F.layer_norm(u.float(), (d,))
    t0 = a.plan_c; TS = np.array([t0 + k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12) if t0 + k < T]); nt = len(TS)   # t0 = l'instant « maintenant » des situations Push-T

    def encode(bb, t):                          # lot uint8 -> image t+1 : latents de l'œil-cible (B, npf, d) et cases de pixels (B, npf, pd)
        Z, PX = [], []
        with torch.no_grad():
            for i in range(0, len(bb["X"]), a.chunk):
                tok, _ = to_tok0({k: x[i:i + a.chunk] for k, x in bb.items()}, st, v, P_, dev); ix = torch.arange(nv, device=dev).expand(len(tok), -1)
                Z.append(LN(tgt(tok[:, :nv], ix)).view(len(tok), T, npf, d)[:, t + 1].half().cpu()); PX.append(tok[:, (t + 1) * npf:(t + 2) * npf, :pd].half().cpu())
        return torch.cat(Z), torch.cat(PX)

    # ---- 1. son jeu libre : scènes (avec sa main) et buts (sans sa main) à 9 instants, étiquettes lues dans les images
    wkw = {**cfg.get("wkw", {}), "p_out": a.p_out, "nohand": True}; wkw.pop("circ", None)
    w = to_np([gen_world0(a.n, "0e", T, H, seed=4242, **wkw)])
    LAB = np.stack([sep_labels(w["XNH"][i], TS, round(a.thr * 255)) for i in range(a.n)])                      # (n, nt, nt)
    print(f"{a.n} séquences de jeu libre, maintenant = {t0}, instants {TS.tolist()} | paires où le monde a changé : {np.mean(LAB > 0):.0%} "
          f"(séquences : {np.mean(LAB.reshape(a.n, -1).max(1) > 0):.0%}) | gestes : " + " ".join(f"{k}:{np.mean(LAB == k):.0%}" for k in range(int(LAB.max()) + 1)) + f" | {time.time() - t_:.0f}s", flush=True)
    FS, FG = {"bébé": [], "pixels": []}, {"bébé": [], "pixels": []}
    for i in range(0, a.n, 8):
        b = to_torch(w, slice(i, i + 8)); B = len(b["X"]); Xv = []
        for src in ("X", "XNH"):
            for u in TS: x = b["X"].clone(); x[:, t0 + 1:] = b[src][:, u:u + 1]; Xv.append(x)                  # vrai passé ≤ t0, puis l'image u « pour toujours »
        bb = {k: b[k].repeat_interleave(2 * nt, 0) for k in KEYS}; bb["X"] = torch.stack(Xv, 1).flatten(0, 1)
        z, px = encode(bb, t0)
        for k, f in (("bébé", z), ("pixels", px)): f = f.view(B, 2, nt, npf, -1); FS[k].append(f[:, 0]); FG[k].append(f[:, 1])
    FS, FG = ({k: torch.cat(x) for k, x in F_.items()} for F_ in (FS, FG)); ntr = int(0.8 * a.n)
    print(f"latents du jeu libre prêts | {time.time() - t_:.0f}s", flush=True)
    si, ai, bi = (torch.from_numpy(x.reshape(-1)) for x in np.meshgrid(np.arange(a.n), np.arange(nt), np.arange(nt), indexing="ij")); lab = torch.from_numpy(LAB.reshape(-1))
    tr0, tr1 = torch.where((si < ntr) & (lab == 0))[0], torch.where((si < ntr) & (lab > 0))[0]; te = torch.where(si >= ntr)[0]

    # ---- 2. les situations Push-T fabriquées (jamais vues par la tête) : but, 8 scènes d'octobre, puis le T à 0 / 25 / 50 / 75 % du chemin (main à l'écart)
    names = ["1 départ", "2 T parti, vide", "3 T parti, MAIN au but", "4 T resté, MAIN au but", "5 T AU BUT", "6 T BLANC au but", "7 T autre couleur au but", "8 DISQUE au but"]
    fr_ = (0.0, 0.25, 0.5, 0.75); SZ, SP, OLD = [], [], []; rng = np.random.default_rng(0)
    for ep in range(a.n_sit):
        env = Push0(ep, task="pusht", shape_phys=True, L=T); r_ = np.random.default_rng(30_000 + ep)
        for _ in range(a.plan_c): env.step(np.clip(r_.normal(0, 0.03, 2), -VMAX, VMAX))
        t = env.t; assert t == t0; b1 = env.batch(0); cur = env.X[t]
        gf = env.frame(env.g, env.Hp, env.g_ang, hand=False)
        gmask = torch.from_numpy(np.abs(gf - cur).max(-1).reshape(nP, P_, nP, P_).max((1, 3)).reshape(-1) > 0.15)
        if not bool(gmask.any()): continue
        g = np.linspace(0.1, 0.9, 17); cand = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
        far = lambda pts, lo: cand[np.argmax(np.min([np.linalg.norm(cand - q, axis=1) for q in pts], 0) * (np.all((cand > lo) & (cand < 1 - lo), 1)))]
        Pfar = far([env.P, env.g], env.rc + 0.02)                                  # loin de l'ancienne place et du but
        Hfar = env.place(far([env.P, env.g, Pfar], 0.08))                          # main dans un coin libre (dans la portée du bras)
        def scene(P, ang, Hp, col=None, kind=None):
            e = copy.copy(env)
            if col is not None: e.col = np.asarray(col, np.float32)
            if kind is not None: e.kind = kind
            return e.frame(P, Hp, ang, hand=True)
        sc = [cur, scene(Pfar, env.ang, Hfar), scene(Pfar, env.ang, env.g), scene(env.P, env.ang, env.g), scene(env.g, env.g_ang, Hfar),
              scene(env.g, env.g_ang, Hfar, col=(1.0, 1.0, 1.0)), scene(env.g, env.g_ang, Hfar, col=np.clip(rng.uniform(0.15, 0.95, 3), 0, 1)),
              scene(env.g, env.g_ang, Hfar, kind=0)]
        sc += [scene(env.P + f * (env.g - env.P), env.ang + f * (env.g_ang - env.ang), Hfar) for f in fr_]
        seqs = []
        for S in [gf] + sc:                                                        # le passé vrai ≤ t, puis la scène « pour toujours »
            Xs = env.X[:T].copy(); Xs[t + 1:] = S; seqs.append(torch.from_numpy((Xs * 255).round().astype(np.uint8)))
        bb = {k: x.expand(len(seqs), *x.shape[1:]).clone() for k, x in b1.items()}; bb["X"] = torch.stack(seqs)
        z, px = encode(bb, t); SZ.append(z); SP.append(px)
        OLD.append(((z[1:].float() - z[:1].float()) ** 2).sum(-1)[:, gmask].mean(-1).numpy())                    # l'ancien score (écart des latents, cases qui doivent changer)
    SC = {"bébé": torch.stack(SZ), "pixels": torch.stack(SP)}; OLD = np.array(OLD); n = len(OLD); ns = len(names)
    print(f"{n} situations Push-T prêtes | {time.time() - t_:.0f}s", flush=True)

    # ---- 3. la tête : apprise sur le jeu libre, jugée sur le jeu libre à part puis sur les scènes
    def fit(kind, seed):
        fs, fg = FS[kind], FG[kind]; torch.manual_seed(seed); net = DistHead(fs.size(-1), npf, norm=kind == "bébé").to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
        for s in range(a.steps):
            sel = torch.cat([tr0[torch.randint(0, len(tr0), (128,))], tr1[torch.randint(0, len(tr1), (128,))]])   # moitié « rien n'a bougé », moitié « le monde a changé »
            l = F.smooth_l1_loss(net(fs[si[sel], ai[sel]].to(dev), fg[si[sel], bi[sel]].to(dev)), lab[sel].to(dev)); opt.zero_grad(); l.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pr = torch.cat([net(fs[si[c], ai[c]].to(dev), fg[si[c], bi[c]].to(dev)).cpu() for c in te.split(1024)]).numpy()
            S = SC[kind]; D = net(S[:, 1:].flatten(0, 1).to(dev), S[:, :1].expand(-1, S.size(1) - 1, -1, -1).flatten(0, 1).to(dev)).view(n, -1).cpu().numpy()
        y = lab[te].numpy(); mv = y > 0
        print(f"   {kind:>6s} graine {seed} | jeu libre à part : erreur {np.abs(pr - y).mean():.2f} geste (repère « toujours la médiane » {np.abs(y - np.median(y)).mean():.2f}) | "
              f"monde changé : erreur {np.abs(pr - y)[mv].mean():.2f} (repère {np.abs(y - np.median(y[mv]))[mv].mean():.2f}), corrélation {np.corrcoef(pr[mv], y[mv])[0, 1]:+.2f} | "
              f"« déjà au but » reconnu : AUC {auc(-pr, ~mv):.2f} | {time.time() - t_:.0f}s", flush=True)
        return D

    def report(title, C, unit):
        print(f"{title}\n   {'scène':>26s} | {unit:>13s} | {'meilleur que le départ':>22s}")
        for i, nm in enumerate(names): print(f"   {nm:>26s} | {C[..., i].mean():13.2f} | {np.mean(C[..., i] < C[..., 0]):22.0%}")
        vs = lambda i, j: f"{np.mean(C[..., i] < C[..., j]):.0%}"
        print(f"   DUELS (part des situations où la 1re scène est jugée PLUS PROCHE du but que la 2e) :\n"
              f"      main au but (T parti) vs T AU BUT : {vs(2, 4)} | main au but (T resté) vs T AU BUT : {vs(3, 4)}   <- critère ≤ 10 %\n"
              f"      T AU BUT vs départ : {vs(4, 0)}   <- critère ≥ 90 % | main au but (T parti) vs zone vide (T parti) : {vs(2, 1)}   <- le raccourci paie-t-il encore ?\n"
              f"      T BLANC vs T de sa couleur : {vs(5, 4)} | DISQUE vs T : {vs(7, 4)}")
        f0, f25, f50, f75, f100 = (C[..., i] for i in (ns, ns + 1, ns + 2, ns + 3, 4))
        print(f"   PROGRÈS (T à 0 / 25 / 50 / 75 / 100 % du chemin, main à l'écart) : " + " / ".join(f"{x.mean():.2f}" for x in (f0, f25, f50, f75, f100))
              + f" | baisse à chaque étape : {np.mean((f0 > f25) & (f25 > f50) & (f50 > f75) & (f75 > f100)):.0%} | 50 % plus proche que 0 % : {np.mean(f50 < f0):.0%} | 75 % plus proche que 50 % : {np.mean(f75 < f50):.0%}", flush=True)

    report(f"ANCIEN SCORE (écart des latents, bébé {ck['state']['it']}, {n} situations ; plus BAS = plus proche du but pour lui)", OLD, "score")
    for kind in ("bébé", "pixels"):
        print(f"JUGE APPRIS sur {'les latents du bébé' if kind == 'bébé' else 'les PIXELS (repère)'} — {a.seeds} têtes, {a.steps} pas :", flush=True)
        D = np.stack([fit(kind, s) for s in range(a.seeds)])
        report(f"   -> scènes Push-T (moyenne des {a.seeds} têtes)", D, "gestes jugés")


if __name__ == "__main__":
    main()
