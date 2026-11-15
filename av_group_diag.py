"""
QUALITÉ DES GROUPES de l'imagination par objet (15 nov. 2026 ; modèles GELÉS, sans entraînement)
Question user : l'amplitude de l'objet poussé imaginé était meilleure avec l'ancien module (53–56 %) qu'avec le bébé v2 (28 %) — est-ce la SÉPARATION DES OBJETS ?
« chose / fond » est identique (0.33 / 0.02 des deux côtés). Reste le REGROUPEMENT (ObjImag.proj : « quelles cases bougent ensemble ») : dans l'ancien module il partait
du mouvement commun de l'ancien bébé (co_proj), dans le bébé v2 il est appris de zéro.
Ici, sur les mêmes scènes (instruments = masques du monde), à l'image 7 :
  - deux cases du MÊME objet se ressemblent-elles plus que deux cases d'objets DIFFÉRENTS / qu'une case d'objet et une case de fond ? (AUC, dans proj et dans les latents bruts)
  - le poids de groupe A (celui qui moyenne le déplacement) d'une case de cœur d'objet : part qui tombe sur SON objet, sur un AUTRE objet, sur la main, sur le fond.
Mondes : objets × 1.8 et × 1 ; couleurs quelconques et MÊME couleur.
  python av_group_diag.py --ckpt bebe2_rampe.pt                                   # bébé v2 (m.oi)
  python av_group_diag.py --ckpt phase0_rD17s_0e_270k.pt --objimag objimag_D17s_v10.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, H
from av_world0 import gen_world0, VIEW


def auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    if len(pos) < 5 or len(neg) < 5: return float("nan")
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--objimag", type=str, default=""); p.add_argument("--n", type=int, default=1200); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); av_phase0.T = T
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; ix = torch.arange(nv, device=dev)
    if a.objimag:
        from av_objimag import ObjImag
        cko = torch.load(a.objimag, map_location=dev, weights_only=False); om = ObjImag(d, nP, T, gate=cko["cfg"].get("gate", 0), audio=cko["cfg"].get("audio", 0), senses=cko["cfg"].get("senses", 0)).to(dev); om.load_state_dict(cko["m"]); om.eval()
    else: om = m.oi.eval()
    wk0 = {k: x for k, x in cfg.get("wkw", {}).items() if k not in ("nohand", "same_col", "circ")}; wk0.update(p_out=0.0, p_parent=0.0, kick=0.0)
    log(f"QUALITÉ DES GROUPES : {a.ckpt.split('/')[-1]} (pas {ck['state']['it']}) | module : {a.objimag.split('/')[-1] if a.objimag else 'dans le bébé (m.oi)'} | netteté du groupe exp(ps) = {float(om.ps.exp()):.1f} | {a.n} scènes par monde")
    t = 7
    for nm, wkw in (("objets × 1.8, couleurs quelconques", {**wk0, "obj_scale": 1.8}), ("objets × 1.8, MÊME couleur", {**wk0, "obj_scale": 1.8, "same_col": 1.0}),
                    ("objets × 1, couleurs quelconques", {**wk0, "obj_scale": 1.0}), ("objets × 1, MÊME couleur", {**wk0, "obj_scale": 1.0, "same_col": 1.0})):
        R = {k: [] for k in ("ps", "pd", "pb", "zs", "zd", "zb", "a_same", "a_other", "a_hand", "a_bg", "c_same", "c_other", "c_hand", "c_bg", "o_obj", "o_bg")}
        with torch.no_grad():
            for c0 in range(0, a.n, 200):
                w0 = gen_world0(min(200, a.n - c0), "0e", T, H, seed=661_000 + c0 // 200, masks=True, **wkw); MK = w0["MASK"]; nc = len(MK); K = MK.shape[2]
                fc = torch.from_numpy((MK.reshape(nc, T, K, nP, P_, nP, P_).mean((4, 6)) / 255).reshape(nc, T, K, npf)[:, t].astype(np.float32))   # (nc, K, npf) ; dernier = main
                w = to_np([w0]); del w0, MK
                for i in range(0, nc, 50):
                    b = to_torch(w, slice(i, i + 50)); tok, _ = to_tok0(b, st, v, P_, dev); B = len(tok)
                    Z = F.layer_norm(tgt(tok[:, :nv], ix.expand(B, -1)).float(), (d,)).view(B, T, npf, d)[:, t]
                    u = F.normalize(om.proj(Z), dim=-1); S = (u @ u.transpose(1, 2)).cpu(); A = torch.softmax(om.ps.exp() * (u @ u.transpose(1, 2)), -1).cpu()
                    zc = Z - om.clock[t]; zn = F.normalize(zc, dim=-1); SZ = (zn @ zn.transpose(1, 2)).cpu()
                    o = om.things(Z, tok[:, nv + 2 * T + t, :4].float()).cpu()
                    f = fc[i:i + 50]; fo, fh = f[:, :-1], f[:, -1]; anyo = fo.amax(1); bg = (anyo < 0.02) & (fh < 0.02); hand = fh >= 0.5
                    for j in range(B):
                        cores = [torch.where((fo[j, k] >= 0.5) & (fh[j] < 0.1))[0] for k in range(K - 1)]; cores = [c for c in cores if len(c) >= 2]; bj = torch.where(bg[j])[0]; hj = torch.where(hand[j])[0]
                        if cores: R["o_obj"].append(o[j, torch.cat(cores)]); R["o_bg"].append(o[j, bj])
                        for q, c in enumerate(cores):
                            iu = torch.triu_indices(len(c), len(c), 1); R["ps"].append(S[j][c][:, c][iu[0], iu[1]]); R["zs"].append(SZ[j][c][:, c][iu[0], iu[1]])
                            R["pb"].append(S[j][c][:, bj].flatten()[::7]); R["zb"].append(SZ[j][c][:, bj].flatten()[::7])
                            oth = [c2 for q2, c2 in enumerate(cores) if q2 != q]
                            if oth: oc = torch.cat(oth); R["pd"].append(S[j][c][:, oc].flatten()); R["zd"].append(SZ[j][c][:, oc].flatten())
                            Ar = A[j][c]                                                                               # (cases de cœur, npf) : où va le poids de groupe de chaque case
                            same = torch.zeros(npf, dtype=torch.bool); same[torch.where(fo[j, q] >= 0.1)[0]] = True
                            other = (torch.cat([fo[j, :q], fo[j, q + 1:]]).amax(0) >= 0.1) & ~same if K > 2 else torch.zeros(npf, dtype=torch.bool)
                            for key, M in (("same", same), ("other", other & ~hand[j]), ("hand", hand[j] & ~same), ("bg", bg[j])):
                                R["a_" + key].append(Ar[:, M].sum(1)); R["c_" + key].append(torch.full((len(c),), float(M.float().mean())))
        C = {k: (torch.cat(x).numpy() if x else np.zeros(0)) for k, x in R.items()}
        log(f"\n=== {nm} | paires même objet {len(C['ps'])}, objets différents {len(C['pd'])} | {time.time() - t_:.0f}s")
        log(f"   deux cases du MÊME objet se ressemblent plus que… (AUC ; 0.50 = hasard) | dans SON regroupement (proj) : …deux cases d'objets DIFFÉRENTS {auc(C['ps'], C['pd']):.2f} | …une case d'objet et une de FOND {auc(C['ps'], C['pb']):.2f} "
            f"|| dans les latents bruts (dé-datés) : objets différents {auc(C['zs'], C['zd']):.2f} | fond {auc(C['zs'], C['zb']):.2f}")
        log(f"   ressemblance moyenne (proj) : même objet {C['ps'].mean():+.2f} | objets différents {C['pd'].mean() if len(C['pd']) else float('nan'):+.2f} | objet–fond {C['pb'].mean():+.2f}")
        log("   POIDS DE GROUPE d'une case de cœur d'objet (part du poids ; entre parenthèses = part des cases, c.-à-d. ce que donnerait un poids uniforme) : " +
            " | ".join(f"{lab} {C['a_' + key].mean():.0%} ({C['c_' + key].mean():.0%})" for key, lab in (("same", "SON objet"), ("other", "un AUTRE objet"), ("hand", "la main"), ("bg", "le fond"))))
        log(f"   « chose » dite sur : cœur d'objet {C['o_obj'].mean():.2f} | fond {C['o_bg'].mean():.2f}")
    log(f"total {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
