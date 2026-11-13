"""
D'OÙ VIENT LE CHANGEMENT PERMANENT DE SES LATENTS ? (13 nov. 2026 ; bébé GELÉ, sans entraînement)
Constat (av_transport_diag) : deux images d'un objet IMMOBILE diffèrent déjà de 0.42–0.47 dans ses latents ; un déplacement de > 8 px n'ajoute que +0.13.
Ici : la même scène (un objet qui ne bouge JAMAIS, tâche « objet » d'av_plan0.Push0, main loin de lui) dans des conditions contrôlées :
  A  tout est immobile (images identiques au pixel près)            -> ce que fait le simple passage du TEMPS dans son œil ;
  B  sa main bouge LOIN de l'objet, il la voit                       -> + l'effet de la main ailleurs ;
  C  sa main bouge loin de l'objet, il ne la VOIT pas (il la sent)   -> la part de B qui passe par la vue de la main.
Mesure : écart moyen entre les latents de la MÊME case à l'image t et à l'image t+k (œil-cible, normalisés), sur les cases de cœur d'objet et sur
des cases de fond ; lu « vue seule » (comme le planificateur) et « tous les sens » (comme à l'entraînement). Repères d'échelle : deux objets
DIFFÉRENTS (scènes différentes), objet contre fond dans la même image. Par instant t (son œil est causal : peu de passé au début).
  python av_floor_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17s_0e_270k.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0, av_plan0
from av_phase0 import baby_from_cfg, to_tok0, H
from av_world0 import VIEW
from av_plan0 import Push0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=300); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); av_phase0.T = av_plan0.T = T
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; wk_ = cfg.get("wkw", {})
    LN = lambda u: F.layer_norm(u.float(), (d,)); CONDS = ("A tout immobile", "B main bouge loin, VUE", "C main bouge loin, PAS vue")
    Z = {(c, s_): [] for c in range(3) for s_ in ("vue", "tous")}; CORE, BG = [], []
    for ep in range(a.n):
        for c in range(3):
            env = Push0(ep, task="objet", damp=wk_.get("damp", 0.0), echo=wk_.get("echo", 0.0), shape_phys=bool(wk_.get("shape_phys", False))); rng = np.random.default_rng(70_000 + ep)
            cs = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]], np.float32); env.Hp = env.place(cs[np.linalg.norm(cs - env.P, axis=1).argmax()]); env.record(0); far = env.rc + env.Rh + 0.12
            for _ in range(T - 1):
                cmd = np.zeros(2, np.float32)
                if c > 0:                         # marche au hasard, toujours loin de l'objet (jamais de contact)
                    for _try in range(20):
                        cmd = np.clip(rng.normal(0, 0.05, 2), -av_plan0.VMAX, av_plan0.VMAX).astype(np.float32)
                        if np.linalg.norm(env.place(env.Hp + cmd) - env.P) > far: break
                    else: cmd = np.zeros(2, np.float32)
                env.step(cmd)
            assert float(np.linalg.norm(env.POS[T - 1] - env.POS[0])) < 1e-6, "l'objet a bougé"
            b = env.batch(0, nohand=(c == 2))
            with torch.no_grad():
                tok, _ = to_tok0(b, st, v, P_, dev); N = tok.size(1)
                Z[(c, "vue")].append(LN(tgt(tok[:, :nv], torch.arange(nv, device=dev)[None])).view(T, npf, d).cpu())
                Z[(c, "tous")].append(LN(tgt(tok, torch.arange(N, device=dev)[None]))[:, :nv].view(T, npf, d).cpu())
            if c == 1:                            # cases : cœur d'objet (≥ 50 % de pixels d'objet), fond (aucun pixel d'objet ni de main dans tout l'épisode B)
                fo = (env.XN[0].max(-1) > 0.05).reshape(nP, P_, nP, P_).mean((1, 3)).reshape(-1)
                hand_any = (np.abs(env.X[:T] - env.XN[:T]).max((0, 3)) > 0.02).reshape(nP, P_, nP, P_).max((1, 3)).reshape(-1)
                CORE.append(torch.from_numpy(fo >= 0.5)); BG.append(torch.from_numpy((fo == 0) & ~hand_any))
    CORE, BG = torch.stack(CORE), torch.stack(BG); Z = {k: torch.stack(x) for k, x in Z.items()}                    # (n, npf) ; (n, T, npf, d)
    log(f"PLANCHER DES LATENTS (bébé gelé, pas {ck['state']['it']}) : {a.n} scènes à objet immobile | cases de cœur par scène {float(CORE.float().sum(1).mean()):.1f}, de fond {float(BG.float().sum(1).mean()):.1f} | {time.time() - t_:.0f}s")
    def dz(Zc, t, k, M):                         # écart moyen entre la même case à t et t+k, sur les cases M
        e = (Zc[:, t] - Zc[:, t + k]).abs().mean(-1); return float((e * M).sum() / M.sum().clamp_min(1))
    Za = Z[(0, "vue")]; perm = torch.randperm(a.n, generator=torch.Generator().manual_seed(0)); t = 7
    ia = [int(torch.where(CORE[i])[0][0]) if CORE[i].any() else -1 for i in range(a.n)]; ok = [i for i in range(a.n) if ia[i] >= 0 and ia[int(perm[i])] >= 0 and int(perm[i]) != i]
    ref_obj = float(torch.stack([(Za[i, t, ia[i]] - Za[int(perm[i]), t, ia[int(perm[i])]]).abs().mean() for i in ok]).mean())
    ib = [int(torch.where(BG[i])[0][0]) if BG[i].any() else -1 for i in range(a.n)]; ok2 = [i for i in range(a.n) if ia[i] >= 0 and ib[i] >= 0]
    ref_bg = float(torch.stack([(Za[i, t, ia[i]] - Za[i, t, ib[i]]).abs().mean() for i in ok2]).mean())
    ref_2c = float(torch.stack([(Za[i, t, torch.where(CORE[i])[0][0]] - Za[i, t, torch.where(CORE[i])[0][-1]]).abs().mean() for i in range(a.n) if int(CORE[i].sum()) >= 2]).mean())
    log(f"   REPÈRES D'ÉCHELLE (image {t}, vue seule) : deux objets DIFFÉRENTS {ref_obj:.3f} | objet contre fond, même image {ref_bg:.3f} | deux cases du MÊME objet, même image {ref_2c:.3f}")
    for s_ in ("vue", "tous"):
        log(f"\n   lu « {'vue seule (comme le planificateur)' if s_ == 'vue' else 'tous les sens (comme à l entraînement)'} » — écart entre la MÊME case à t et à t+k")
        log(f"   {'':>28s} | {'CŒUR D OBJET : t=7 k=1':>22s}   k=2    k=4    k=8 | t=2 k=1  t=12 k=1 | {'FOND : t=7 k=1':>14s}   k=2    k=4    k=8")
        for c in range(3):
            Zc = Z[(c, s_)]
            log(f"   {CONDS[c]:>28s} | {dz(Zc, 7, 1, CORE):22.3f} {dz(Zc, 7, 2, CORE):6.3f} {dz(Zc, 7, 4, CORE):6.3f} {dz(Zc, 7, 8, CORE):6.3f} | {dz(Zc, 2, 1, CORE):7.3f} {dz(Zc, 12, 1, CORE):9.3f} | "
                f"{dz(Zc, 7, 1, BG):14.3f} {dz(Zc, 7, 2, BG):6.3f} {dz(Zc, 7, 4, BG):6.3f} {dz(Zc, 7, 8, BG):6.3f}")
    Zc = Z[(0, "vue")]; e = (Zc[:, 7] - Zc[:, 8]).abs()                                                               # A : quelles dimensions changent ?
    pd_ = (e * CORE[:, :, None]).sum((0, 1)) / CORE.sum().clamp_min(1); srt = pd_.sort(descending=True).values
    log(f"\n   A, cœur d'objet, images 7→8 : part du changement portée par les 10 % de dimensions qui changent le plus {float(srt[:d // 10].sum() / srt.sum()):.0%} "
        f"(si ≈ 10 % : tout change un peu ; si ≫ : quelques dimensions « horloge »)")
    mt = Zc.mean(2)                                                                                                    # (n, T, d) moyenne de l'image : la part COMMUNE à toutes les cases
    e_c = ((Zc[:, 7] - mt[:, 7:8]) - (Zc[:, 8] - mt[:, 8:9])).abs().mean(-1)
    log(f"   A, cœur d'objet, images 7→8, une fois retirée la MOYENNE de l'image (ce qui change pareil dans toutes les cases) : {float((e_c * CORE).sum() / CORE.sum()):.3f} "
        f"(brut {dz(Zc, 7, 1, CORE):.3f})")
    log("   lecture : A déjà haut ⇒ le TEMPS seul fait bouger ses latents (son œil date chaque image) ; B ≫ A ⇒ la main ailleurs se répand dans toutes les cases ; C ≈ A ⇒ c'est la VUE de la main")
    log(f"total {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
