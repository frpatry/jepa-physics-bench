"""
SONDE DE MOUVEMENT sur toute une lignée de bébés (15 nov. 2026 ; œil GELÉ, aucun entraînement du bébé)
Question : l'ancien module d'imagination par objet donne 53 % d'amplitude sur l'œil de D17s (270k pas) et 36 % sur celui du bébé v2 (97k). Les deux yeux diffèrent par
beaucoup de choses (durée, pertes tirées des pixels dès 192k, tête « gestes manquants », taille des objets, objets lancés seuls, main effacée). À QUELLE ÉTAPE le sens du
mouvement des objets apparaît-il dans les latents ? Ici, sur les MÊMES scènes pour tous (un objet, monde à frottement, tailles × 1 et × 1.8), pour chaque sauvegarde :
  A. LECTURE DU DÉPLACEMENT (instrument : régression linéaire, moitié des scènes pour l'apprendre) : depuis les latents de l'œil-cible autour d'une case d'objet aux images
     t et t+k (dé-datés par l'horloge moyenne), retrouve-t-on de combien l'objet s'est déplacé ? -> direction (cosinus), part d'amplitude lue, sur les objets poussés ; norme lue
     sur les objets immobiles.
  B. SIGNAL / BRUIT des cibles : changement des latents d'une case d'objet (dé-daté) quand l'objet est immobile, déplacé de 2–4 px, de 4–8 px.
  python av_motion_probe.py --ckpts a.pt,b.pt,...
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, H
from av_world0 import gen_world0, VIEW

WK = dict(p_obj=0.8, mob_size=0.16, p_parent=0.0, blind=True, echo=0.5, damp=0.5, shape_phys=True, p_auto=0.0, p_out=0.0, kick=0.0)
PAIRS = ((3, 4), (7, 4), (11, 4), (3, 8), (7, 8))


def world(n, sc, T, P_):
    nP = H // P_; out = dict(b=[], pos=[], fo=[], fh=[])
    for c0 in range(0, n, 500):
        w0 = gen_world0(500, "0e", T, H, seed=915_000 + int(sc * 10) * 100 + c0 // 500, masks=True, **{**WK, "obj_scale": sc}); keep = np.where(w0["NOBJ"] == 1)[0]; MK = w0["MASK"][keep]
        fc = (MK.reshape(len(keep), T, MK.shape[2], nP, P_, nP, P_).mean((4, 6)) / 255).reshape(len(keep), T, MK.shape[2], nP * nP).astype(np.float32)
        out["fo"].append(torch.from_numpy(fc[:, :, 0])); out["fh"].append(torch.from_numpy(fc[:, :, -1])); out["pos"].append(torch.from_numpy(w0["POS"][keep][:, :, 0] * H).float())
        out["b"].append(to_np([{k: w0[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}])); del w0, MK
    b = {k: np.concatenate([x[k] for x in out["b"]]) for k in out["b"][0]}
    return b, torch.cat(out["pos"]), torch.cat(out["fo"]), torch.cat(out["fh"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True); p.add_argument("--n", type=int, default=9000); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True); T = 16; P_ = 4; nP = H // P_; npf = nP * nP; nv = T * npf; av_phase0.T = T
    W = {sc: world(a.n, sc, T, P_) for sc in (1.0, 1.8)}
    for sc, (b, pos, fo, fh) in W.items():
        dn = (pos[:, 15] - pos[:, 7]).norm(dim=-1); log(f"monde objets × {sc} : {len(pos)} scènes à un objet | déplacé de > 0.5 px entre les images 7 et 15 : {float((dn > 0.5).float().mean()):.0%}, de 4–8 px : {int(((dn >= 4) & (dn < 8)).sum())} scènes | {time.time() - t_:.0f}s")
    log("par sauvegarde et par monde : A = LECTURE DU DÉPLACEMENT (objets poussés ≥ 0.5 px, scènes jamais vues par la sonde) : R², direction (cos) et part d'amplitude lue pour 2–4 px et 4–8 px, norme lue sur objet immobile "
        "| B = changement des latents d'une case d'objet : immobile / 2–4 px / 4–8 px (rapport 4–8 px sur immobile)")
    for ck_ in a.ckpts.split(","):
        ck = torch.load(ck_, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; d = cfg["d"]; assert int(cfg.get("T", 16)) == T and cfg["P"] == P_
        m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); v = VIEW["0e"]; ix = torch.arange(nv, device=dev)
        line = f"{ck_.split('/')[-1][:-3]:>28s} (pas {ck['state']['it']:>6d})"
        for sc, (b, pos, fo, fh) in W.items():
            n = len(pos); Z = []
            with torch.no_grad():
                for i in range(0, n, 50):
                    tok, _ = to_tok0(to_torch(b, slice(i, i + 50)), st, v, P_, dev); Z.append(F.layer_norm(tgt(tok[:, :nv], ix.expand(len(tok), -1)).float(), (d,)).view(-1, T, npf, d).half().cpu())
            Z = torch.cat(Z).float(); Z = Z - Z.mean(0, keepdim=True)                                                     # dé-daté : horloge moyenne par instant et par case
            X, Y, SC, DZ = [], [], [], []
            for t, k in PAIRS:
                dz = (Z[:, t + k] - Z[:, t]).view(n, nP, nP, d).permute(0, 3, 1, 2)                                       # (n, d, nP, nP)
                nb = F.unfold(F.pad(dz, (1, 1, 1, 1), mode="replicate"), 3).view(n, d * 9, npf).permute(0, 2, 1)           # voisinage 3×3 du changement
                core = (fo[:, t] >= 0.5) & (fh[:, t] < 0.1) & (fh[:, t + k] < 0.1); si, ci = torch.where(core)
                X.append(torch.cat([nb[si, ci], Z[si, t, ci]], -1)); Y.append((pos[si, t + k] - pos[si, t])); SC.append(si); DZ.append(dz.permute(0, 2, 3, 1).reshape(n, npf, d)[si, ci].abs().mean(-1))
            X, Y, SC, DZ = torch.cat(X), torch.cat(Y), torch.cat(SC), torch.cat(DZ); tr = SC < int(0.5 * n); te = ~tr; yn = Y.norm(dim=-1)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = ((X - mu) / sd).to(dev).double(); Yd = Y.to(dev).double()
            wt = torch.where(yn[tr] >= 0.5, 5.0, 1.0).to(dev).double()                                                     # les objets poussés comptent plus (ils sont rares)
            A_ = (Xn[tr.to(dev)] * wt[:, None]).T @ Xn[tr.to(dev)] + 300.0 * torch.eye(Xn.size(1), device=dev, dtype=torch.double); Wm = torch.linalg.solve(A_, (Xn[tr.to(dev)] * wt[:, None]).T @ Yd[tr.to(dev)])
            Pr = (Xn @ Wm).float().cpu(); pn = Pr.norm(dim=-1); mv = te & (yn >= 0.5); r2 = float(1 - ((Pr[mv] - Y[mv]) ** 2).sum() / ((Y[mv] - Y[mv].mean(0)) ** 2).sum())
            seg = lambda lo, hi: te & (yn >= lo) & (yn < hi); f_ = lambda q: (f"cos {float(F.cosine_similarity(Pr[q], Y[q], dim=-1).mean()):+.2f} amp {float(pn[q].mean() / yn[q].mean()):.0%} (n={int(q.sum())})" if int(q.sum()) >= 20 else "trop peu")
            im = yn < 0.05; s24, s48 = (yn >= 2) & (yn < 4), (yn >= 4) & (yn < 8)
            line += (f" || × {sc} A : R² {r2:+.2f} | 2–4 px {f_(seg(2, 4))} | 4–8 px {f_(seg(4, 8))} | immobile lu {float(pn[te & im].mean()):.2f} px"
                     f" | B : {float(DZ[im].mean()):.3f} / {float(DZ[s24].mean()):.3f} / {float(DZ[s48].mean()):.3f} (× {float(DZ[s48].mean() / DZ[im].mean()):.2f})")
            del Z, X, Xn
        log(line + f" | {time.time() - t_:.0f}s"); del m, tgt, ck; torch.cuda.empty_cache()
    log(f"total {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
