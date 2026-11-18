"""
QUE VIT LE BÉBÉ EN JEU LIBRE ? (17 nov. 2026 ; aucun modèle — statistiques du monde, instruments = positions et toucher du générateur)
Constat : avec les bons gestes, l'imagination déplace la main mais presque pas l'objet poussé. Hypothèse user : « garbage in, garbage out » — son jeu libre contient
trop peu de vraies poussées. Ici, dans le monde EXACT de son entraînement (réglages lus dans la sauvegarde) : combien d'images où un objet bouge, de combien, pendant
combien de temps d'affilée, poussé par lui ou non — comparé aux parties Push-T jouées par l'utilisateur (recordings/) et au mouvement de sa main.
  python av_play_stats.py --ckpt bebe3c.pt --rec recordings
"""
import argparse, glob, json, time
import numpy as np, torch
import av_plan0
from av_world0 import gen_world0
from av_plan0 import Push0

H = 32


def runs(mv):
    """mv (T,) bool -> [(début, longueur)] des suites d'images consécutives où ça bouge."""
    out, t = [], 0
    while t < len(mv):
        if mv[t]:
            t1 = t
            while t1 < len(mv) and mv[t1]: t1 += 1
            out.append((t, t1 - t)); t = t1
        else: t += 1
    return out


def stats(nm, tracks, log):
    """tracks : [(pos (T,2) px, mine (T-1,) bool = déplacement dû à SA main, hand (T,2) px)]"""
    D = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) for p, _, _ in tracks]); M = np.concatenate([m for _, m, _ in tracks]); Dh = np.concatenate([np.linalg.norm(np.diff(h, axis=0), axis=1) for _, _, h in tracks])
    R = [(l, float(np.linalg.norm(p[t0 + l] - p[t0]))) for p, _, _ in tracks for t0, l in runs(np.linalg.norm(np.diff(p, axis=0), axis=1) > 0.2)]; L = np.array([r[0] for r in R] or [0]); A = np.array([r[1] for r in R] or [0.0])
    W = np.concatenate([np.linalg.norm(p[8:] - p[:-8], axis=1) for p, _, _ in tracks if len(p) > 8]); Wh = np.concatenate([np.linalg.norm(h[8:] - h[:-8], axis=1) for _, _, h in tracks if len(h) > 8])
    nseq = len(tracks); nfr = len(D); pc = lambda x: f"{100 * x:5.1f} %"
    log(f"\n=== {nm} : {nseq} suivis d'objet, {nfr} images")
    log(f"  OBJET, d'une image à la suivante : bouge (> 0.05 px) {pc(np.mean(D > 0.05))} | > 0.5 px {pc(np.mean(D > 0.5))} | > 1 px {pc(np.mean(D > 1))} | > 2 px {pc(np.mean(D > 2))} | déplacement moyen {D.mean():.3f} px par image"
        f" | part du mouvement due à SA main {pc(D[M].sum() / max(D.sum(), 1e-9))}")
    log(f"  MAIN, d'une image à la suivante  : bouge (> 0.05 px) {pc(np.mean(Dh > 0.05))} | > 0.5 px {pc(np.mean(Dh > 0.5))} | > 1 px {pc(np.mean(Dh > 1))} | > 2 px {pc(np.mean(Dh > 2))} | déplacement moyen {Dh.mean():.3f} px par image")
    log(f"  OBJET sur 8 images : immobile (< 0.1 px) {pc(np.mean(W < 0.1))} | 0.1–1 px {pc(np.mean((W >= 0.1) & (W < 1)))} | 1–2 px {pc(np.mean((W >= 1) & (W < 2)))} | 2–4 px {pc(np.mean((W >= 2) & (W < 4)))} | 4–8 px {pc(np.mean((W >= 4) & (W < 8)))} | > 8 px {pc(np.mean(W >= 8))}"
        f" || MAIN sur 8 images : < 1 px {pc(np.mean(Wh < 1))} | 1–4 px {pc(np.mean((Wh >= 1) & (Wh < 4)))} | 4–8 px {pc(np.mean((Wh >= 4) & (Wh < 8)))} | > 8 px {pc(np.mean(Wh >= 8))}")
    log(f"  DÉPLACEMENTS D'AFFILÉE de l'objet (suites d'images où il avance de > 0.2 px) : {len(R) / max(nfr, 1) * 100:.2f} pour 100 images | durée moyenne {L.mean():.1f} images (≥ 4 images : {pc(np.mean(L >= 4))}, ≥ 8 : {pc(np.mean(L >= 8))})"
        f" | trajet moyen {A.mean():.2f} px (≥ 2 px : {pc(np.mean(A >= 2))}, ≥ 4 px : {pc(np.mean(A >= 4))}, ≥ 8 px : {pc(np.mean(A >= 8))})")
    return dict(mv=np.mean(D > 0.5), mean=D.mean(), w48=np.mean(W >= 4))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--rec", default="recordings"); p.add_argument("--n", type=int, default=4000)
    a = p.parse_args(); t0 = time.time(); log = lambda s_: print(s_, flush=True)
    cfg = torch.load(a.ckpt, map_location="cpu", weights_only=False)["cfg"]; wkw = dict(cfg.get("wkw", {})); T = int(cfg.get("T", 16)); log(f"monde du jeu libre (étape 0e), réglages de la sauvegarde : {wkw}")
    tr = []; ncon = 0; nobj = []
    for c0 in range(0, a.n, 500):
        w = gen_world0(min(500, a.n - c0), "0e", T, H, seed=31_000 + c0 // 500, **wkw); n = len(w["X"]); act = (w["TSRC"] == 1) if "TSRC" in w else (np.abs(w["TOUCH"]).sum(-1) > 0)
        ncon += int((np.abs(w["TOUCH"]).sum(-1) > 0).any(1).sum()); nobj += list(w["NOBJ"])
        for i in range(n):
            hd = w["HAND"][i] * H
            for j in range(int(w["NOBJ"][i])):
                ps = w["POS"][i, :, j] * H
                if np.isnan(ps).any(): continue
                near = np.linalg.norm(ps - hd, axis=1) < 9; mine = (act[i, :-1] | act[i, 1:]) & (near[:-1] | near[1:]); tr.append((ps, mine, hd))
    log(f"{a.n} séquences de {T} images | avec au moins un contact senti {ncon / a.n:.0%} | objets par séquence {np.mean(nobj):.2f} | {time.time() - t0:.0f}s")
    s1 = stats("JEU LIBRE DU BÉBÉ (monde de son entraînement)", tr, log)
    av_plan0.T = T; tr2 = []
    for f in sorted(glob.glob(a.rec + "/*.json")):
        r = json.load(open(f)); env = Push0(r["ep"], task="pusht", damp=wkw.get("damp", 0.0), echo=wkw.get("echo", 0.0), shape_phys=bool(wkw.get("shape_phys", False)), L=r["n"] + 2, obj_scale=wkw.get("obj_scale", 1.0))
        for c in r["cmd"]: env.step(np.array(c, np.float32), render=False)
        n = r["n"]; tc = np.abs(env.TOUCH[:n + 1]).sum(-1) > 0; tr2.append((env.POS[:n + 1] * H, tc[:-1] | tc[1:], env.HAND[:n + 1] * H))
    if tr2:
        s2 = stats("PARTIES PUSH-T JOUÉES PAR L'UTILISATEUR", tr2, log)
        log(f"\nRÉSUMÉ : images où l'objet avance de > 0.5 px — jeu libre {s1['mv']:.1%}, utilisateur {s2['mv']:.1%} (× {s2['mv'] / max(s1['mv'], 1e-9):.1f}) | déplacement moyen par image — {s1['mean']:.3f} px contre {s2['mean']:.3f} px")
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
