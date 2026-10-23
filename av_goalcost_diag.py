"""
DIAG DU SCORE IMAGE-BUT (question user 2026-10-22 : « la couleur de l'objet devrait compter ? il a trouvé un détour ? »).
GIF : le bébé pose SA MAIN dans la zone-but (et pousse parfois le T HORS de la zone) au lieu d'y amener le T.
Ici, pas de planification : pour N situations Push-T, on fabrique des SCÈNES et on calcule le score que le planificateur
leur donnerait (av_plan0 --goal_img : écart des latents de l'œil-cible à l'image-but SANS main, sur les cases qui doivent
changer) — comme si l'imagination était PARFAITE (la scène est encodée telle quelle à t+1). Plus bas = « meilleur » pour lui.
  1 départ            : la scène actuelle (T à sa place, main où elle est)
  2 T parti, vide     : T poussé loin (hors des deux zones), main dans un coin libre
  3 T parti, MAIN     : T poussé loin, sa MAIN au centre de la zone-but
  4 T resté, MAIN     : T pas touché, sa MAIN au centre de la zone-but (ép. 2 des GIF)
  5 T AU BUT          : le T exact à sa place voulue, main dans un coin libre (la vraie réussite)
  6 T BLANC au but    : même chose mais le T est BLANC (comme la main) -> la couleur compte-t-elle ?
  7 T autre couleur   : T au but, d'une autre couleur tirée au hasard
  8 DISQUE au but     : un disque de la couleur du T, au centre du but -> la forme compte-t-elle ?
  python av_goalcost_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17n_0e_190k.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, layout, H
from av_world0 import VIEW
import av_plan0
from av_plan0 import Push0, VMAX


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=100); p.add_argument("--plan_c", type=int, default=3)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t_ = time.time()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    T = int(cfg.get("T", 16)); av_phase0.T = T; av_plan0.T = T
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; d = cfg["d"]; v = VIEW["0e"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"], strict=False); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    LN = lambda u: F.layer_norm(u.float(), (d,))
    names = ["1 départ", "2 T parti, vide", "3 T parti, MAIN au but", "4 T resté, MAIN au but", "5 T AU BUT", "6 T BLANC au but", "7 T autre couleur au but", "8 DISQUE au but"]
    C, CS = [], []; rng = np.random.default_rng(0)
    for ep in range(a.n):
        env = Push0(ep, task="pusht", shape_phys=True, L=T); r_ = np.random.default_rng(30_000 + ep)
        for _ in range(a.plan_c): env.step(np.clip(r_.normal(0, 0.03, 2), -VMAX, VMAX))
        t = env.t; b1 = env.batch(0); cur = env.X[t]
        gf = env.frame(env.g, env.Hp, env.g_ang, hand=False)
        gmask = torch.from_numpy(np.abs(gf - cur).max(-1).reshape(nP, P_, nP, P_).max((1, 3)).reshape(-1) > 0.15).to(dev)
        if not bool(gmask.any()): continue
        g = np.linspace(0.1, 0.9, 17); cand = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
        far = lambda pts, lo: cand[np.argmax(np.min([np.linalg.norm(cand - q, axis=1) for q in pts], 0) * (np.all((cand > lo) & (cand < 1 - lo), 1)))]
        Pfar = far([env.P, env.g], env.rc + 0.02)                                  # loin de l'ancienne place et du but
        Hfar = env.place(far([env.P, env.g, Pfar], 0.08))                          # main dans un coin libre (dans la portée du bras)
        hands = []                                                                 # « ÇA, C'EST MOI » : les cases où est la main (dans la scène)
        hcase = lambda wh, nh: torch.from_numpy(np.abs(wh - nh).max(-1).reshape(nP, P_, nP, P_).max((1, 3)).reshape(-1) > 0.05).to(dev)
        def scene(P, ang, Hp, col=None, kind=None):
            e = copy.copy(env)
            if col is not None: e.col = np.asarray(col, np.float32)
            if kind is not None: e.kind = kind
            wh = e.frame(P, Hp, ang, hand=True); hands.append(hcase(wh, e.frame(P, Hp, ang, hand=False))); return wh
        hands.append(hcase(cur, env.frame(env.P, env.Hp, env.ang, hand=False))); h_now = hands[0]
        sc = [cur, scene(Pfar, env.ang, Hfar), scene(Pfar, env.ang, env.g), scene(env.P, env.ang, env.g), scene(env.g, env.g_ang, Hfar),
              scene(env.g, env.g_ang, Hfar, col=(1.0, 1.0, 1.0)), scene(env.g, env.g_ang, Hfar, col=np.clip(rng.uniform(0.15, 0.95, 3), 0, 1)),
              scene(env.g, env.g_ang, Hfar, kind=0)]
        seqs = []
        for S in [gf] + sc:                                                        # le passé vrai ≤ t, puis la scène « pour toujours »
            Xs = env.X[:T].copy(); Xs[t + 1:] = S; seqs.append(torch.from_numpy((Xs * 255).round().astype(np.uint8)))
        bb = {k: x.expand(len(seqs), *x.shape[1:]).clone() for k, x in b1.items()}; bb["X"] = torch.stack(seqs)
        with torch.no_grad():
            tok, _ = to_tok0(bb, st, v, P_, dev); ix = torch.arange(nv, device=dev).expand(len(seqs), -1)
            z = LN(tgt(tok[:, :nv], ix)).view(len(seqs), T, npf, d)[:, t + 1]       # latents à t+1
        dz = ((z[1:] - z[:1]) ** 2).sum(-1)
        C.append(dz[:, gmask].mean(-1).cpu().numpy())
        CS.append(np.array([float(dz[i][gmask & ~h_now & ~hands[i]].mean()) if bool((gmask & ~h_now & ~hands[i]).any()) else np.nan
                            for i in range(len(sc))]))    # sans les cases de SA MAIN (maintenant et dans la scène) : il les reconnaît par son bras
    C = np.array(C); n = len(C); CS = np.array(CS)
    print(f"{n} situations Push-T, bébé {ck['state']['it']} | score image-but (plus BAS = plus proche du but pour lui ; imagination supposée parfaite) | {time.time() - t_:.0f}s")
    print(f"   {'scène':>26s} | {'score moyen':>11s} | {'relatif au départ':>17s} | {'meilleur que le départ':>22s}")
    for i, nm in enumerate(names):
        print(f"   {nm:>26s} | {C[:, i].mean():11.1f} | {np.mean(C[:, i] / C[:, 0]):17.2f} | {np.mean(C[:, i] < C[:, 0]):22.0%}")
    vs = lambda i, j: f"{np.mean(C[:, i] < C[:, j]):.0%}"
    print(f"   DUELS (part des situations où la 1re scène a un MEILLEUR score que la 2e) :")
    print(f"      main au but (T parti) vs zone vide (T parti) : {vs(2, 1)}   <- le raccourci paie-t-il ?")
    print(f"      main au but (T parti) vs T AU BUT             : {vs(2, 4)}   <- le raccourci bat-il la vraie réussite ?")
    print(f"      T parti + zone vide vs départ                 : {vs(1, 0)}   <- vider l'ancienne place paie-t-il ?")
    print(f"      T BLANC au but vs T de sa couleur au but      : {vs(5, 4)}   (écart moyen {np.mean(C[:, 5] - C[:, 4]):+.1f}) <- la couleur compte-t-elle ?")
    print(f"      DISQUE au but vs T au but                     : {vs(7, 4)}   (écart moyen {np.mean(C[:, 7] - C[:, 4]):+.1f}) <- la forme compte-t-elle ?")
    print(f"      T BLANC au but vs main au but (T parti)       : {vs(5, 2)}")
    print(f"   « ÇA, C'EST MOI » — même score SANS les cases de sa main (maintenant + dans la scène) :")
    for i, nm in enumerate(names):
        print(f"   {nm:>26s} | {np.nanmean(CS[:, i]):11.1f} | {np.nanmean(CS[:, i] / CS[:, 0]):17.2f} | {np.nanmean(CS[:, i] < CS[:, 0]):22.0%}")
    vs2 = lambda i, j: f"{np.nanmean(CS[:, i] < CS[:, j]):.0%}"
    print(f"      main au but (T parti) vs T AU BUT : {vs2(2, 4)} | T AU BUT vs départ : {vs2(4, 0)} | T BLANC vs T coloré : {vs2(5, 4)} | DISQUE vs T : {vs2(7, 4)}")


if __name__ == "__main__":
    main()
