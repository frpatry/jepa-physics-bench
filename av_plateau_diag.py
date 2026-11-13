"""
AU PALIER, UN GESTE PEUT-IL ENCORE AMÉLIORER ? (13 nov. 2026 ; bébé GELÉ, VRAI monde à la place de son imagination)
Constat : avec le vrai monde (av_plan0 --real_wm) le T monte à ~70 % de couverture puis n'est plus touché ; son juge voit pourtant très
bien que 70 % n'est pas le but (av_cov_diag). Hypothèse du user : « elle ne veut pas perdre des points » — tous les plans ESSAYÉS font
baisser la couverture, donc ne plus toucher est le meilleur plan.
États : tâche Push-T, le T posé à une couverture choisie (bande « palier » 60–80 %, bande « bas » 20–40 %), immobile, la main juste à
côté (côté au hasard), passé immobile de t0 = T−1−h images. Depuis chaque état, pour trois TAILLES DE GESTES (écart-type du tirage) :
  tirage au hasard (comme le 1er tour du planificateur : 2 segments × 3 pas) : part des plans qui touchent le T / améliorent / abîment ;
  planificateur + SON JUGE sur la vraie scène obtenue (CEM 64 × 4, même recherche qu'av_plan0 --real_wm) : couverture après son plan ;
  planificateur + la VRAIE couverture (ORACLE de la même recherche) à 6 pas, et à 12 pas (4 segments) : ce qui était atteignable ;
  accord du juge : parmi les plans qui touchent le T, son classement suit-il la couverture obtenue (corrélation de rang) ?
  python av_plateau_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17r_0e_250k.pt
"""
import argparse, copy, math, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0, av_plan0
from av_phase0 import baby_from_cfg, to_tok0, VIEW, H
from av_plan0 import Push0
from av_judge import build_judge
VMAX = av_plan0.VMAX


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=60, help="états par bande"); p.add_argument("--plan_h", type=int, default=6)
    p.add_argument("--bands", type=str, default="0.6-0.8,0.2-0.4"); p.add_argument("--sds", type=str, default="0.06,0.02,0.007", help="tailles de gestes (0.06 = celle du planificateur)")
    p.add_argument("--task_scale", type=float, default=0.0, help="taille des objets (0 = celle du monde vécu)"); p.add_argument("--pop", type=int, default=64); p.add_argument("--iters", type=int, default=4)
    p.add_argument("--judge", type=int, default=2); p.add_argument("--judge_n", type=int, default=3000); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    av_phase0.T = av_plan0.T = T
    teachN = lambda x, ix: F.layer_norm(tgt(x, ix).float(), (d,)); P_ = cfg["P"]; npf = (H // P_) ** 2; nv = T * npf; v = VIEW["0e"]; ix = torch.arange(nv, device=dev)
    wk_ = cfg.get("wkw", {}); sc = a.task_scale or float(wk_.get("obj_scale", 1.0)); Hh = a.plan_h; t0 = T - 1 - Hh
    log(f"AU PALIER (bébé gelé, pas {ck['state']['it']}) : {a.n} états par bande, objets × {sc}, horizon {Hh}, recherche {a.pop} × {a.iters}")
    judge = build_judge(m, teachN, m.pred, cfg, st, dev, d, Hh, mode=a.judge, n=a.judge_n, log=log)
    from scipy.stats import spearmanr

    def roll(env, cand, frames=False):           # cand (K, h, 2) -> couverture après le plan (K,), [images (K, T, H, H, 3)]
        cv, X = [], []
        for cc in cand:
            e2 = copy.deepcopy(env)
            for ac in cc: e2.step(ac, render=frames)
            cv.append(e2.coverage()); X.append(e2.X[:T].copy()) if frames else None
        return np.array(cv), (np.array(X) if frames else None)

    def judge_cost(env, b1, zg, cand):           # son juge sur la VRAIE scène obtenue (dernière image de la fenêtre)
        cv, X = roll(env, cand, True); out = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                K = len(X[i:i + 64]); bK = {k: x.expand(K, *x.shape[1:]).clone() for k, x in b1.items()}
                bK["X"] = torch.from_numpy((X[i:i + 64] * 255).round().astype(np.uint8)); tok, _ = to_tok0(bK, st, v, P_, dev)
                out.append(judge(teachN(tok[:, :nv], ix.expand(K, -1)).view(K, T, npf, d)[:, -1], zg.expand(K, -1, -1)).cpu().numpy())
        return np.concatenate(out), cv

    def cem(costfn, sd0, h, rg):                 # la recherche d'av_plan0 (segments de 3 pas) ; rend le plan final (h, 2) et le 1er tirage (cand, coût, couv.)
        K = h // 3; mu, sd = np.zeros((K, 2)), np.full((K, 2), sd0); first = None
        for _ in range(a.iters):
            seg = np.clip(mu + sd * rg.normal(size=(a.pop, K, 2)), -VMAX, VMAX); cand = np.repeat(seg, 3, 1).astype(np.float32); c, cv = costfn(cand)
            first = first or (cand, c, cv); el = seg[np.argsort(c)[:max(4, a.pop // 8)]]; mu, sd = el.mean(0), el.std(0) + 0.01 * sd0 / 0.06
        return np.repeat(mu, 3, 0).astype(np.float32), first

    sds = [float(x) for x in a.sds.split(",")]
    for band in a.bands.split(","):
        lo, hi = (float(x) for x in band.split("-")); R = {s_: [] for s_ in sds}; c0s = []; ep = 0
        while len(c0s) < a.n and ep < 20 * a.n:
            env = Push0(ep, task="pusht", shape_phys=True, damp=wk_.get("damp", 0.0), echo=wk_.get("echo", 0.0), obj_scale=sc, L=T + 14); rng = np.random.default_rng(50_000 + ep); ep += 1
            ok = False
            for _ in range(300):                 # le T à une couverture de la bande, la main juste à côté (hors du cercle du T)
                an = rng.uniform(0, 2 * math.pi); P = (env.g + rng.uniform(0, 0.8 * env.sz) * np.array([math.cos(an), math.sin(an)])).astype(np.float32); ang = float(env.g_ang + rng.uniform(-0.9, 0.9))
                if not (np.all(P > env.rc) and np.all(P < 1 - env.rc)) or not lo <= env.coverage(P, ang) < hi: continue
                ah = rng.uniform(0, 2 * math.pi); want = P + (env.rc + env.Rh + rng.uniform(0.02, 0.06)) * np.array([math.cos(ah), math.sin(ah)]); Hp = env.place(want)
                if np.linalg.norm(Hp - want) < 1e-4: ok = True; break
            if not ok: continue
            env.P, env.ang, env.Hp = P, ang, Hp; env.V[:] = 0; env.om = 0.0; env.record(0)
            for _ in range(t0): env.step(np.zeros(2, np.float32))
            c0 = env.coverage(); c0s.append(c0); b1 = env.batch(0)
            Xg = env.X[:T].copy(); Xg[t0 + 1:] = env.frame(env.g, env.Hp, env.g_ang, hand=False); bg = dict(b1); bg["X"] = torch.from_numpy((Xg * 255).round().astype(np.uint8))[None]
            with torch.no_grad(): tkg, _ = to_tok0(bg, st, v, P_, dev); zg = teachN(tkg[:, :nv], ix[None]).view(T, npf, d)[-1:]
            for s_ in sds:
                rg = np.random.default_rng(ep)
                pj, (cand, cj, cv) = cem(lambda c_: judge_cost(env, b1, zg, c_), s_, Hh, rg)
                po, _ = cem(lambda c_: (lambda r_: (-r_, r_))(roll(env, c_)[0]), s_, Hh, np.random.default_rng(ep))
                po2, _ = cem(lambda c_: (lambda r_: (-r_, r_))(roll(env, c_)[0]), s_, 2 * Hh, np.random.default_rng(ep))
                dcv = cv - c0; tch = np.abs(dcv) > 1e-4
                rho = spearmanr(-cj[tch], cv[tch]).correlation if tch.sum() >= 8 and cv[tch].std() > 1e-6 else np.nan
                R[s_].append((tch.mean(), (dcv > 0.01).mean(), (dcv < -0.01).mean(), dcv.max(), roll(env, pj[None])[0][0] - c0, roll(env, po[None])[0][0] - c0,
                              roll(env, po2[None])[0][0] - c0, rho, float(abs(roll(env, pj[None])[0][0] - c0) < 1e-4)))
        log(f"\n=== départ à {lo:.0%}–{hi:.0%} de couverture : {len(c0s)} états (moyenne {np.mean(c0s):.0%}) | {time.time() - t_:.0f}s")
        log(f"   {'taille des gestes':>20s} | 1er tirage : touchent le T / améliorent / abîment | meilleur du tirage | SON JUGE : couv. après son plan (ne touche pas) | ORACLE 6 pas | ORACLE 12 pas | juge ↔ couverture (rang)")
        for s_ in sds:
            r = np.array(R[s_], float); mn = np.nanmean(r, 0)
            log(f"   {f'{s_} ({s_ * H:.1f} px/pas)':>20s} | {mn[0]:^10.0%} / {mn[1]:^10.0%} / {mn[2]:^7.0%}            | {mn[3]:^+17.0%} | {mn[4]:^+26.0%} ({mn[8]:.0%})          | {mn[5]:^+12.0%} | {mn[6]:^+13.0%} | {mn[7]:^+10.2f}")
            log(f"   {'':>20s}   son juge : améliore (> +1 pt) dans {np.mean(r[:, 4] > 0.01):.0%} des états, abîme (< −1 pt) dans {np.mean(r[:, 4] < -0.01):.0%} | oracle 6 pas améliore dans {np.mean(r[:, 5] > 0.01):.0%} | "
                f"oracle 12 pas atteint ≥ 95 % dans {np.mean(np.array(c0s) + r[:, 6] >= 0.95):.0%}")
    log("   lecture : oracle haut + juge bas ⇒ le juge choisit mal ; oracle bas à 6 pas mais haut à 12 ⇒ horizon ; tout bas en gros gestes mais haut en fins ⇒ taille des gestes")


if __name__ == "__main__":
    main()
