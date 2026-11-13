"""
LE PALIER DE PUSH-T (13 nov. 2026) : son ŒIL et son JUGE distinguent-ils un T à 70 % de couverture d'un T à 100 % ? (bébé GELÉ)
Constat (av_plan0 --real_wm) : même avec le vrai monde à la place de son imagination, le T monte à ~70 % de couverture puis n'est plus
touché. Hypothèse : à ~70 % (≈ 1 px du but) la scène est, pour lui, déjà le but (vue 32 px en cases de 4 px ; juge à ± 1.7 geste ;
étiquette « déjà au but » du juge = pixels égaux à 15 % près).
Scènes : tâche Push-T (av_plan0.Push0), le T posé à des écarts CONTRÔLÉS du but (décalage 0..~3 px × rotation 0..35°, + le but exact),
la main loin (coin le plus éloigné du but) ; lues comme dans le planificateur à la fenêtre qui glisse : scène immobile, image-but sans
la main « pour toujours » après t = T−1−h, latents de l'œil-cible à la dernière image.
Trois façons de dire « laquelle est la plus près du but » (paires de scènes du MÊME épisode, classées par couverture) :
  pixels    : écart moyen des pixels à l'image-but (REPÈRE : l'information est dans l'image) ;
  son œil   : écart de ses latents à ceux de l'image-but (l'ancien score du planificateur, --goal_img sans juge) ;
  son juge  : « gestes restants » (av_judge, celui du planificateur --judge 2) ;
  sonde     : INSTRUMENT — une tête neuve apprise à lire la couverture dans ses latents (épisodes à part) = ce que l'œil CONTIENT.
Mesure : part des paires bien ordonnées (0.50 = hasard) par couple de tranches de couverture ; et ce que dit le juge par tranche.
  python av_cov_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17r_0e_250k.pt
"""
import argparse, copy, math, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0, av_plan0
from av_phase0 import baby_from_cfg, to_tok0, layout, VIEW, H
from av_plan0 import Push0
from av_judge import build_judge, DistHead

BINS = ((0.95, 1.01, "≥ 95 %"), (0.80, 0.95, "80–95 %"), (0.60, 0.80, "60–80 %"), (0.30, 0.60, "30–60 %"), (0.0, 0.30, "< 30 %"))
DUELS = ((0, 1), (0, 2), (1, 2), (0, 3), (2, 3), (3, 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=300, help="épisodes Push-T"); p.add_argument("--poses", type=int, default=15)
    p.add_argument("--plan_h", type=int, default=6); p.add_argument("--judge", type=int, default=2); p.add_argument("--judge_n", type=int, default=3000)
    p.add_argument("--scales", type=str, default="", help="tailles d'objets de la tâche, ex. 1,1.8 (vide = 1 et celle du monde vécu)")
    p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    av_phase0.T = av_plan0.T = T
    teachN = lambda x, ix: F.layer_norm(tgt(x, ix).float(), (d,)); P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]
    wk_ = cfg.get("wkw", {}); lived = float(wk_.get("obj_scale", 1.0)); t0 = T - 1 - a.plan_h
    scales = [float(x) for x in a.scales.split(",")] if a.scales else sorted({1.0, lived})
    log(f"PALIER DE PUSH-T (bébé gelé, pas {ck['state']['it']}) : {a.n} épisodes × {a.poses} poses du T autour du but | objets du monde vécu × {lived} | tailles testées {scales}")
    judge = build_judge(m, teachN, m.pred, cfg, st, dev, d, a.plan_h, mode=a.judge, n=a.judge_n, log=log) if a.judge else None
    ix = torch.arange(nv, device=dev)
    for sc in scales:
        rng = np.random.default_rng(7); COV, EP, PX, EYE, JUD, ZS, ZG, OFF = [], [], [], [], [], [], [], []
        for ep in range(a.n):
            env = Push0(ep, task="pusht", shape_phys=True, obj_scale=sc); b1 = env.batch(0)
            cs = np.array([[0.07, 0.07], [0.93, 0.07], [0.07, 0.93], [0.93, 0.93]], np.float32); Hp = env.place(cs[np.linalg.norm(cs - env.g, axis=1).argmax()])
            K = a.poses; an = rng.uniform(0, 2 * math.pi, K); r = rng.choice([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1], K) * rng.uniform(0.7, 1.3, K)
            da = np.radians(rng.choice([0, 2, 5, 10, 20, 35], K)) * rng.choice([-1, 1], K); r[0] = da[0] = 0.0          # pose 0 = le but exact
            Ps = env.g[None] + r[:, None] * np.stack([np.cos(an), np.sin(an)], 1); angs = env.g_ang + da
            cov = np.array([env.coverage(Ps[k].astype(np.float32), float(angs[k])) for k in range(K)])
            gf = env.frame(env.g, Hp, env.g_ang, hand=False); Xs = np.stack([env.frame(Ps[k].astype(np.float32), Hp, float(angs[k])) for k in range(K)])
            Xq = np.repeat(Xs[:, None], T, 1); Xg = Xq.copy(); Xg[:, t0 + 1:] = gf                                     # scène immobile | passé ≤ t0 puis le but pour toujours
            u8 = lambda X: torch.from_numpy((X * 255).round().astype(np.uint8))
            with torch.no_grad():
                z = []
                for X in (Xq, Xg):
                    bK = {k: x.expand(K, *x.shape[1:]).clone() for k, x in b1.items()}; bK["X"] = u8(X); tok, _ = to_tok0(bK, st, v, P_, dev)
                    z.append(teachN(tok[:, :nv], ix.expand(K, -1)).view(K, T, npf, d)[:, -1])
                zs, zg = z; EYE.append(((zs - zg) ** 2).sum(-1).mean(-1).cpu().numpy())
                if judge is not None: JUD.append(judge(zs, zg).cpu().numpy())
            PX.append(np.abs(Xs - gf[None]).mean((1, 2, 3))); COV.append(cov); EP.append(np.full(K, ep)); ZS.append(zs.half().cpu()); ZG.append(zg.half().cpu())
            OFF.append(np.hypot(r * H, 0))
        COV, EP, PX, EYE, OFF = (np.concatenate(x) for x in (COV, EP, PX, EYE, OFF)); JUD = np.concatenate(JUD) if JUD else None; ZS, ZG = torch.cat(ZS), torch.cat(ZG)
        # SONDE (instrument) : couverture lue dans ses latents, apprise sur 80 % des épisodes
        ntr = int(0.8 * a.n); tr = torch.from_numpy(np.where(EP < ntr)[0]); te = EP >= ntr; yc = torch.from_numpy(COV).float()
        torch.manual_seed(0); net = DistHead(d, npf).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=2000, pct_start=0.05)
        for _ in range(2000):
            sel = tr[torch.randint(0, len(tr), (128,))]; l = F.smooth_l1_loss(net(ZS[sel].to(dev), ZG[sel].to(dev)), yc[sel].to(dev)); opt.zero_grad(); l.backward(); opt.step(); sch.step()
        with torch.no_grad(): PRB = torch.cat([net.eval()(ZS[i:i + 512].to(dev), ZG[i:i + 512].to(dev)).cpu() for i in range(0, len(ZS), 512)]).numpy()
        bn = np.full(len(COV), -1)
        for i, (lo, hi, _) in enumerate(BINS): bn[(COV >= lo) & (COV < hi)] = i

        def duel(cost, i, j, sel):               # paires du même épisode : la scène de la tranche i (plus couverte) a-t-elle le coût le plus BAS ?
            ok = tot = 0
            for ep in np.unique(EP[sel]):
                e = sel & (EP == ep); ci, cj = cost[e & (bn == i)], cost[e & (bn == j)]
                ok += (ci[:, None] < cj[None]).sum() + 0.5 * (ci[:, None] == cj[None]).sum(); tot += len(ci) * len(cj)
            return (ok / tot if tot else np.nan), tot
        al = np.ones(len(COV), bool); hd = " | ".join(f"{BINS[i][2]} vs {BINS[j][2]}" for i, j in DUELS)
        rows = [("pixels (repère)", PX, al), ("son ŒIL (écart des latents)", EYE, al)] + ([("son JUGE (gestes restants)", JUD, al)] if JUD is not None else []) \
            + [("sonde de couverture (à part)", -PRB, te)]
        log(f"\n=== objets de la tâche × {sc} (T de {0.14 * sc * H:.1f} px env.) | scènes par tranche : " + ", ".join(f"{b_[2]} {int((bn == i).sum())}" for i, b_ in enumerate(BINS))
            + f" | décalage moyen par tranche (px, hors rotation) : " + ", ".join(f"{OFF[bn == i].mean():.1f}" for i in range(len(BINS))) + f" | {time.time() - t_:.0f}s")
        log(f"   PAIRES BIEN ORDONNÉES (même épisode ; 0.50 = hasard) :\n   {'':>30s} | {hd}")
        for nm, c_, sel in rows: log(f"   {nm:>30s} | " + " | ".join(f"{duel(c_, i, j, sel)[0]:^{len(BINS[i][2]) + len(BINS[j][2]) + 4}.2f}" for i, j in DUELS))
        log(f"   {'paires (toutes)':>30s} | " + " | ".join(f"{duel(PX, i, j, al)[1]:^{len(BINS[i][2]) + len(BINS[j][2]) + 4}d}" for i, j in DUELS))
        if JUD is not None:
            log("   CE QUE DIT LE JUGE (gestes restants, moyenne ± écart-type) : " + " | ".join(f"{b_[2]} {JUD[bn == i].mean():.2f} ± {JUD[bn == i].std():.2f}" for i, b_ in enumerate(BINS))
                + f" | au but EXACT {JUD[COV > 0.999].mean():.2f}")
        log("   ÉCART DES LATENTS au but (son œil), moyenne : " + " | ".join(f"{b_[2]} {EYE[bn == i].mean():.3f}" for i, b_ in enumerate(BINS)) + f" | au but EXACT {EYE[COV > 0.999].mean():.3f} (la main seule)")
        log(f"   sonde de couverture : erreur moyenne {np.abs(PRB - COV)[te].mean():.3f} (repère « toujours la moyenne » {np.abs(COV[te] - COV[~te].mean()).mean():.3f})")
    log("   lecture : pixels hauts + œil/juge ≈ 0.5 entre « ≥ 95 % » et « 60–80 % » ⇒ pour lui 70 % = arrivé (résolution) ; sonde haute ⇒ l'œil le voit, le juge ne s'en sert pas")


if __name__ == "__main__":
    main()
