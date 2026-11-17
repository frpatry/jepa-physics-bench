"""
L'IMAGINATION EST-ELLE INTERROGÉE AU JEU COMME À L'ENTRAÎNEMENT ? (17 nov. 2026 ; bébé GELÉ, aucun entraînement)
Intuition user : « lorsqu'on joue on interroge et / ou lit mal l'info ». Constat dans le code : l'œil traite toute la séquence d'un bloc, SANS masque de temps -> à
l'entraînement (et à l'examen) le latent de l'image t est calculé avec les images t+1..15 sous les yeux (l'AVENIR) ; au jeu (av_plan0) il n'a que les images 0..t.
Deux risques : (1) FUITE — l'imagination par objet a pu apprendre à lire dans l'image t ce qui va se passer (l'objet sera poussé) ; (2) DÉCALAGE — au jeu elle reçoit
des latents qu'elle n'a jamais vus. Ici, le MÊME examen (objet poussé : amplitude, direction ; immobile : dérive ; main) avec l'image de départ calculée :
  A. comme à l'entraînement (toute la séquence)        B. comme au jeu (les images 0..t seulement, vue seule — exactement l'appel du planificateur)
et l'écart entre les deux latents de départ, comparé au tremblement d'une image à la suivante.
  python av_leak_diag.py --ckpt bebe3c.pt
"""
import argparse, copy, time
import torch, torch.nn.functional as F
import av_phase0, av_objimag
from av_phase0 import baby_from_cfg, to_tok0
from av_world0 import VIEW
from av_phase0 import H


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--n_eval", type=int, default=2000)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]; av_phase0.T = T
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); assert hasattr(m, "oi")
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; TS = (3, 7); D = {"n": 0, "pf": 0.0, "jit": 0.0}
    wkw = {k: x for k, x in cfg.get("wkw", {}).items() if k not in ("nohand", "same_col", "circ")}; sc = float(wkw.get("obj_scale", 1.0))

    def lat(past):
        @torch.no_grad()
        def f(b):
            tok, cmd = to_tok0(b, st, v, P_, dev); B = len(tok); za = F.layer_norm(tgt(tok, torch.arange(tok.size(1), device=dev).expand(B, -1)).float(), (d,)); Z = za[:, :nv].view(B, T, npf, d).clone()
            if past:
                for t in TS:                    # l'image t comme au jeu : la vue des images 0..t seulement
                    vi = torch.arange((t + 1) * npf, device=dev).expand(B, -1); zp = F.layer_norm(tgt(tok[:, :(t + 1) * npf], vi).float(), (d,))[:, -npf:]
                    D["pf"] += float((zp - Z[:, t]).abs().mean()) * B; D["jit"] += float((Z[:, t] - Z[:, t - 1]).abs().mean()) * B; D["n"] += B; Z[:, t] = zp
            return Z, cmd, tok[:, nv + 2 * T:, :4], tok[:, nv + T:nv + 2 * T, :8], za[:, nv:nv + 3 * T].view(B, 3, T, d)
        return f
    R = {}
    for nm, past in (("A. COMME À L'ENTRAÎNEMENT (image de départ calculée avec toute la séquence, avenir compris)", False), ("B. COMME AU JEU (image de départ calculée avec le passé seulement)", True)):
        log(f"\n################ {nm}"); ex = av_objimag.build_exam(m.oi, lat(past), wkw, sc, T, nP, P_, npf, a.n_eval, 12, 1, 0, 0, log, t0); R[nm[0]] = ex(nm[:2] + f" pas {ck['state']['it']}")
    log(f"\nÉCART entre les deux images de départ (latents, |·| moyen) : {D['pf'] / max(1, D['n']):.3f} | pour comparer, changement d'une image à la suivante (toute la séquence) : {D['jit'] / max(1, D['n']):.3f}")
    g = lambda r, k: r.get(k, float("nan"))
    log("RÉSUMÉ (objet poussé de 4–8 px, 8 pas) : amplitude | direction | objet immobile")
    for k_ in ("A", "B"): log(f"   {k_} : {g(R[k_], 'amp_k8'):.0%} | {g(R[k_], 'cos_k8'):+.2f} | {g(R[k_], 'immobile_k8'):.2f} px")
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
