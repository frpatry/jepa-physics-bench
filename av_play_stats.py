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
    """tracks : [(pos (T,2) px, cause 0 lui / 1 parent / 2 objet lancé, hand (T,2) px, contact (T,) bool)]"""
    pc = lambda x: f"{100 * x:5.1f} %"; dd = lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1)
    D = np.concatenate([dd(p) for p, _, _, _ in tracks]); C = np.concatenate([np.full(len(p) - 1, c) for p, c, _, _ in tracks]); Dh = np.concatenate([dd(h) for _, _, h, _ in tracks]); CT = np.concatenate([ct for _, _, _, ct in tracks])
    tot = max(D.sum(), 1e-9); me = [t_ for t_ in tracks if t_[1] == 0]; Dm = np.concatenate([dd(p) for p, _, _, _ in me]); mvf = Dm[Dm > 0.02]
    W = np.concatenate([np.linalg.norm(p[8:] - p[:-8], axis=1) for p, _, _, _ in me if len(p) > 8]); Wh = np.concatenate([np.linalg.norm(h[8:] - h[:-8], axis=1) for _, _, h, _ in tracks if len(h) > 8])
    ep = []                                     # épisodes de contact (trous de ≤ 2 images comblés) : durée, trajet de l'objet pendant l'épisode + 3 images
    for p, c, _, ct in me:
        t = 0; n = len(ct)
        while t < n:
            if ct[t]:
                t1 = t; gap = 0
                while t1 + 1 < n and (ct[t1 + 1] or gap < 2): gap = 0 if ct[t1 + 1] else gap + 1; t1 += 1
                while t1 > t and not ct[t1]: t1 -= 1
                ep.append((t1 - t + 1, float(np.linalg.norm(p[min(t1 + 3, n - 1)] - p[max(t - 1, 0)])))); t = t1 + 1
            else: t += 1
    L = np.array([e[0] for e in ep] or [0]); A = np.array([e[1] for e in ep] or [0.0]); ncs = np.mean([bool(ct.any()) for _, _, _, ct in tracks])
    log(f"\n=== {nm} : {len(tracks)} suivis d'objet, {len(D)} images")
    log(f"  D'OÙ VIENT LE MOUVEMENT DES OBJETS (part du trajet total) : de LUI {pc(D[C == 0].sum() / tot)} | main du parent {pc(D[C == 1].sum() / tot)} | objets lancés seuls {pc(D[C == 2].sum() / tot)}")
    log(f"  CONTACT : images où sa main touche un objet {pc(CT.mean())} | suivis avec au moins un contact {pc(ncs)} | épisodes de contact : {len(ep) / max(len(me), 1):.2f} par suivi, durée moyenne {L.mean():.1f} images "
        f"(1–2 images {pc(np.mean(L <= 2))}, 3–7 {pc(np.mean((L >= 3) & (L < 8)))}, ≥ 8 {pc(np.mean(L >= 8))}) | trajet de l'objet par épisode : < 0.5 px {pc(np.mean(A < 0.5))}, 0.5–2 px {pc(np.mean((A >= 0.5) & (A < 2)))}, 2–6 px {pc(np.mean((A >= 2) & (A < 6)))}, > 6 px {pc(np.mean(A >= 6))}")
    log(f"  OBJET qu'il fait bouger, par image : bouge (> 0.02 px) {pc(np.mean(Dm > 0.02))} des images | quand il bouge : lent (< 0.3 px) {pc(np.mean(mvf < 0.3))}, moyen (0.3–1 px) {pc(np.mean((mvf >= 0.3) & (mvf < 1)))}, rapide (> 1 px) {pc(np.mean(mvf >= 1))}"
        f" || sur 8 images : immobile (< 0.1 px) {pc(np.mean(W < 0.1))} | 0.1–1 px {pc(np.mean((W >= 0.1) & (W < 1)))} | 1–4 px {pc(np.mean((W >= 1) & (W < 4)))} | > 4 px {pc(np.mean(W >= 4))}")
    log(f"  MAIN, par image : < 0.3 px {pc(np.mean(Dh < 0.3))} | 0.3–1 px {pc(np.mean((Dh >= 0.3) & (Dh < 1)))} | 1–2 px {pc(np.mean((Dh >= 1) & (Dh < 2)))} | > 2 px {pc(np.mean(Dh >= 2))} | moyenne {Dh.mean():.2f} px")
    return dict(lui=D[C == 0].sum() / tot, ct=CT.mean(), l8=np.mean(L >= 8), ncs=ncs)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--rec", default="recordings"); p.add_argument("--n", type=int, default=4000); p.add_argument("--set", type=str, default="", help="réglages du monde à changer, ex. push=1,p_parent=0.1"); p.add_argument("--gif", type=str, default="", help="GIF de 12 séquences de ce jeu libre")
    a = p.parse_args(); t0 = time.time(); log = lambda s_: print(s_, flush=True)
    cfg = torch.load(a.ckpt, map_location="cpu", weights_only=False)["cfg"]; wkw = dict(cfg.get("wkw", {})); T = int(cfg.get("T", 16)); wkw.update({k_: float(v_) for k_, v_ in (x.split("=") for x in a.set.split(",") if x)}); log(f"monde du jeu libre (étape 0e), réglages : {wkw}")
    tr = []; ncon = 0; nobj = []
    for c0 in range(0, a.n, 500):
        w = gen_world0(min(500, a.n - c0), "0e", T, H, seed=31_000 + c0 // 500, **wkw); n = len(w["X"])
        if a.gif and c0 == 0:
            from PIL import Image
            S_ = 6; fr = [Image.fromarray((np.concatenate([np.concatenate([np.pad(w["X"][r * 6 + c, t], ((1, 1), (1, 1), (0, 0)), constant_values=0.25) for c in range(6)], 1) for r in range(2)], 0) * 255).astype(np.uint8)).resize((6 * 34 * S_, 2 * 34 * S_), Image.NEAREST) for t in range(T)]
            fr[0].save(a.gif, save_all=True, append_images=fr[1:], duration=300, loop=0); act = (w["TSRC"] == 1) if "TSRC" in w else (np.abs(w["TOUCH"]).sum(-1) > 0)
        ncon += int((np.abs(w["TOUCH"]).sum(-1) > 0).any(1).sum()); nobj += list(w["NOBJ"])
        for i in range(n):
            hd = w["HAND"][i] * H; ct = w["TSRC"][i] > 0; par = bool(np.isfinite(w["PAR"][i]).any()) if "PAR" in w else False; t1 = int(np.argmax(ct)) if ct.any() else T
            for j in range(int(w["NOBJ"][i])):
                ps = w["POS"][i, :, j] * H
                if np.isnan(ps).any(): continue
                auto = t1 >= 2 and float(np.linalg.norm(np.diff(ps[:t1], axis=0), axis=1).max()) > 0.05      # il bougeait AVANT tout contact : lancé seul
                near = np.linalg.norm(ps - hd, axis=1) < 10; tr.append((ps, 1 if par else (2 if auto else 0), hd, ct & near))
    log(f"{a.n} séquences de {T} images | avec au moins un contact senti {ncon / a.n:.0%} | objets par séquence {np.mean(nobj):.2f} | {time.time() - t0:.0f}s")
    stats("JEU LIBRE DU BÉBÉ", tr, log)
    av_plan0.T = T; tr2 = []
    for f in sorted(glob.glob(a.rec + "/*.json")):
        r = json.load(open(f)); env = Push0(r["ep"], task="pusht", damp=wkw.get("damp", 0.0), echo=wkw.get("echo", 0.0), shape_phys=bool(wkw.get("shape_phys", False)), L=r["n"] + 2, obj_scale=wkw.get("obj_scale", 1.0))
        for c in r["cmd"]: env.step(np.array(c, np.float32), render=False)
        n = r["n"]; mvd = np.linalg.norm(np.diff(env.POS[:n + 1], axis=0), axis=1) > 0.005; tc = np.concatenate([[False], mvd]) | np.concatenate([mvd, [False]])   # contact ≈ le T avance (il ne bouge que poussé)
        tr2.append((env.POS[:n + 1] * H, 0, env.HAND[:n + 1] * H, tc))
    if tr2: stats("PARTIES PUSH-T JOUÉES PAR L'UTILISATEUR", tr2, log)
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
