"""
DIAG MOBILE (instrument, quelques minutes GPU) — l'examen « ruban coupé » du run phase 0 moyenne l'erreur sur TOUTE l'image
alors que le mobile n'occupe que ~6 px sur 32 : le signal est noyé. Ici, erreur restreinte aux patches AUTOUR DU MOBILE :
  A ÉCOUTE SES GESTES pour le mobile : erreur près du mobile avec SES commandes vs celles d'une autre séquence,
    mobile RELIÉ (devrait monter) vs mobile AUTONOME (ne devrait pas bouger) ;
  B RUBAN COUPÉ : même contexte, futur où le mobile répond vs ne répond plus -> % de séquences plus « surprenantes » coupées ;
  C RETARD de 3 frames.
Bébé de l'instantané vs même bébé à l'INIT (graine 0 = mêmes poids initiaux que le run).

  python av_mob_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_0c_22k.pt
"""
import argparse, copy
import numpy as np, torch
from av_world0 import gen_world0
from av_phase0 import Baby0, errs, layout, to_np, to_torch, view_at, T, H

def near_mobile(MOB, cols, nv, npf, nP, R=1):
    """(n, len(cols)) : patch (frame, case) à ≤ R case du mobile."""
    cell = (np.arange(nv) % npf)[cols]; fr = (np.arange(nv) // npf)[cols]; gy, gx = cell // nP, cell % nP
    mc = np.clip((MOB * nP).astype(int), 0, nP - 1)                       # (n, T, 2) case (x, y)
    return (np.abs(gx[None] - mc[:, fr, 0]) <= R) & (np.abs(gy[None] - mc[:, fr, 1]) <= R)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=400)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    P = cfg["P"]; nP = H // P; npf = nP * nP; nv = T * npf; md, fr = layout(nv, npf); v = view_at("0c", 1.0)
    mk = lambda: Baby0(cfg["din"], nv, npf, cfg["d"], cfg["nl"], cfg["nh"], cfg["pred_layers"], bool(cfg["sep"]), cfg.get("inv_head", "attn")).to(dev)
    torch.manual_seed(0); m0 = mk().eval(); t0 = copy.deepcopy(m0.enc).eval()
    m1 = mk(); m1.load_state_dict(ck["m"]); m1.eval(); t1 = copy.deepcopy(m1.enc); t1.load_state_dict(ck["tgt"]); t1.eval()
    W = {k: gen_world0(a.n, "0c", T, H, seed=3003, **kw) for k, kw in
         dict(relie=dict(force_mtype=0), auto=dict(force_mtype=1), coupe=dict(force_mtype=2, force_tcut=8), retard=dict(force_mtype=0, mobile_delay=3)).items()}
    B = {k: to_torch(to_np([w])) for k, w in W.items()}
    cm, tm = fr <= 7, (fr > 7) & (md == 0); cols = np.where(tm)[0]
    print(f"instantané pas {ck['state']['it']} | erreur restreinte aux patches à ≤ 1 case du mobile (frames 8–15)", flush=True)
    for nm, (m, tg) in {"INIT": (m0, t0), "BÉBÉ": (m1, t1)}.items():
        e = lambda b, **kw: errs(m, tg, b, st, v, P, cm, tm, dev, cfg["d"], **kw)
        def near_mean(E, w_list):
            nr = torch.from_numpy(np.any([near_mobile(w["MOB"], cols, nv, npf, nP) for w in w_list], 0)).float()
            return (E * nr).sum(1) / nr.sum(1).clamp_min(1)
        out = []
        for kind in ("relie", "auto"):                                     # A : mentir sur la commande
            et, el = near_mean(e(B[kind]), [W[kind]]), near_mean(e(B[kind], lie=True), [W[kind]])
            out.append(f"mobile {'RELIÉ' if kind == 'relie' else 'AUTONOME'} {float(el.mean() / et.mean() - 1):+.0%}")
        en = near_mean(e(B["relie"]), [W["relie"], W["coupe"]]); ec = near_mean(e(B["relie"], b_tgt=B["coupe"]), [W["relie"], W["coupe"]])
        er, en2 = near_mean(e(B["retard"]), [W["retard"]]), near_mean(e(B["relie"]), [W["relie"]])
        print(f"  {nm:>4s} | A mentir sur ses gestes : {', '.join(out)} | B ruban coupé {float((ec > en).float().mean()):.0%} "
              f"| C retard {float((er > en2).float().mean()):.0%}", flush=True)

if __name__ == "__main__":
    main()
