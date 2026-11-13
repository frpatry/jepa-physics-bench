"""
POURQUOI DÉCALER TOUTE L'IMAGE FAIT-IL BAISSER L'ERREUR ? (13 nov. 2026 ; bébé GELÉ, sans entraînement)
Constat (av_objimag v2, v3) : dès qu'il le peut, le module d'imagination par objet décale / floute tous ses latents (~2 px) au lieu de déplacer un objet.
Ici, sur ses VRAIS latents (œil-cible, dé-datés par l'horloge moyenne mesurée : latent moyen de chaque case à chaque instant), on impose nous-mêmes :
  rien (référence : dé-daté puis re-daté) | un RÉTRÉCISSEMENT (× α) | un FLOU symétrique 3×3 | un DÉCALAGE global (dx, dy) en cases | rétrécissement + décalage,
et on regarde l'écart aux vrais latents de t+k : partout, sur le fond, sur l'objet, sur la main ; données comme à l'entraînement du module (table qui déborde,
main parfois hors champ) et données d'examen (tout dans le champ) ; plusieurs instants.
Lectures : le décalage gagne dans TOUTES les directions autant qu'un flou ⇒ c'est du débruitage (le flou doit être offert autrement) ; une direction gagne
nettement ⇒ ses latents sont décalés d'une image à l'autre (biais de l'œil) ; le gain n'existe que sur certaines données / cases ⇒ c'est là qu'il faut regarder.
Aussi : une case à t+1 ressemble-t-elle autant à ses VOISINES de t qu'à elle-même ? (latents lisses dans l'espace ⇒ un décalage ne coûte presque rien).
  python av_shift_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17s_0e_270k.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, H
from av_world0 import gen_world0, VIEW


def warp(x, dx, dy):
    """x (B, C, n, n) déplacé de (dx, dy) cases (interpolation bilinéaire, bords prolongés)."""
    B, _, n, _ = x.shape; ys, xs = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(n, device=x.device), indexing="ij")
    gx = ((xs - dx + 0.5) / n * 2 - 1).float(); gy = ((ys - dy + 0.5) / n * 2 - 1).float()
    return F.grid_sample(x, torch.stack([gx, gy], -1)[None].expand(B, -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=2000); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); av_phase0.T = T
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; ix = torch.arange(nv, device=dev)
    wk0 = {k: x for k, x in cfg.get("wkw", {}).items() if k not in ("nohand", "same_col", "circ")}
    log(f"POURQUOI LE DÉCALAGE GLOBAL PAIE (bébé gelé, pas {ck['state']['it']}) : {a.n} séquences par jeu de données | monde vécu : hors champ {wk0.get('p_out', 0)}, parent {wk0.get('p_parent', 0)}, objets × {wk0.get('obj_scale', 1.0)}")
    grid = lambda Z: Z.reshape(-1, nP, nP, d).permute(0, 3, 1, 2); flat = lambda G: G.permute(0, 2, 3, 1).reshape(-1, npf, d)
    for nm, wkw in (("COMME L'ENTRAÎNEMENT DU MODULE (table qui déborde, main parfois hors champ)", wk0), ("COMME L'EXAMEN (tout dans le champ, pas de parent)", {**wk0, "p_out": 0.0, "p_parent": 0.0, "kick": 0.0})):
        Z, FO, FH = [], [], []
        with torch.no_grad():
            for c0 in range(0, a.n, 250):
                w0 = gen_world0(min(250, a.n - c0), "0e", T, H, seed=881_000 + c0 // 250, masks=True, **wkw); MK = w0["MASK"]; nc = len(MK)
                fc = (MK.reshape(nc, T, MK.shape[2], nP, P_, nP, P_).mean((4, 6)) / 255).reshape(nc, T, MK.shape[2], npf)
                FO.append(torch.from_numpy(fc[:, :, :-1].max(2).astype(np.float32))); FH.append(torch.from_numpy(fc[:, :, -1].astype(np.float32))); w = to_np([w0]); del w0, MK
                for i in range(0, nc, 50):
                    b = to_torch(w, slice(i, i + 50)); tok, _ = to_tok0(b, st, v, P_, dev)
                    Z.append(F.layer_norm(tgt(tok[:, :nv], ix.expand(len(tok), -1)).float(), (d,)).view(-1, T, npf, d).cpu())
        Z, FO, FH = torch.cat(Z), torch.cat(FO), torch.cat(FH); clk = Z.mean(0)                                        # horloge mesurée : latent moyen par instant et par case
        log(f"\n=== {nm} | {time.time() - t_:.0f}s")
        for t, k in ((7, 1), (7, 4), (3, 1), (12, 1)):
            C = grid(Z[:, t] - clk[t]); Y = Z[:, t + k] - clk[t + k]                                                    # contenu dé-daté à t ; cible dé-datée à t+k
            fo = torch.maximum(FO[:, t], FO[:, t + k]); fh = torch.maximum(FH[:, t], FH[:, t + k])
            masks = {"partout": torch.ones_like(fo), "fond": ((fo < 0.02) & (fh < 0.02)).float(), "objet": ((fo >= 0.5) & (fh < 0.1)).float(), "main": (fh >= 0.5).float()}
            def err(G): e = (flat(G) - Y).abs().mean(-1); return {q: float((e * M).sum() / M.sum().clamp_min(1)) for q, M in masks.items()}
            fmt = lambda r: " | ".join(f"{q} {r[q]:.4f}" for q in masks)
            e0 = err(C); log(f"   image {t} -> {t + k} | référence (dé-daté, re-daté) : {fmt(e0)}   [recopier sans horloge : {float((Z[:, t] - Z[:, t + k]).abs().mean()):.4f}]")
            rs = [(al, err(C * al)) for al in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)]; al, rb = min(rs, key=lambda x: x[1]["partout"]); log(f"      meilleur RÉTRÉCISSEMENT × {al} : {fmt(rb)}")
            bs = []
            for wn in (0.03, 0.06, 0.1, 0.15):
                kk = torch.full((3, 3), wn); kk[1, 1] = 1 - 8 * wn; G = F.conv2d(F.pad(C.reshape(-1, 1, nP, nP), (1, 1, 1, 1), mode="replicate"), kk[None, None]).reshape(C.shape)
                for al2 in (1.0, al): bs.append((wn, al2, err(G * al2)))
            wn, al2, bb = min(bs, key=lambda x: x[2]["partout"]); log(f"      meilleur FLOU symétrique (voisin {wn}, × {al2}) : {fmt(bb)}")
            sh = {(dx, dy): err(warp(C, dx, dy) * al) for dx in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0) for dy in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)}
            (bx, by), sb = min(sh.items(), key=lambda x: x[1]["partout"]); log(f"      meilleur DÉCALAGE global (× {al}) : dx {bx:+.2f}, dy {by:+.2f} cases -> {fmt(sb)}")
            log("         selon la direction (demi-case, partout) : " + " | ".join(f"{q} {sh[xy]['partout']:.4f}" for q, xy in (("droite", (0.5, 0.0)), ("gauche", (-0.5, 0.0)), ("bas", (0.0, 0.5)), ("haut", (0.0, -0.5)),
                                                                                    ("diag.", (0.5, 0.5)), ("anti-diag.", (-0.5, -0.5)))) + f" | aucun {sh[(0.0, 0.0)]['partout']:.4f}")
        t = 7; Cn = F.normalize(grid(Z[:, t] - clk[t]), dim=1); Yn = F.normalize(grid(Z[:, t + 1] - clk[t + 1]), dim=1); fo, fh = FO[:, t].view(-1, nP, nP), FH[:, t].view(-1, nP, nP)
        res = []
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0)):
            cs = (Yn[:, :, dy:, dx:] * Cn[:, :, :nP - dy, :nP - dx]).sum(1); fo_, fh_ = fo[:, dy:, dx:], fh[:, dy:, dx:]
            bg = (fo_ < 0.02) & (fh_ < 0.02) & (fo[:, :nP - dy, :nP - dx] < 0.02) & (fh[:, :nP - dy, :nP - dx] < 0.02); ob = (fo_ >= 0.5) & (fo[:, :nP - dy, :nP - dx] >= 0.5)
            res.append(f"({dx},{dy}) fond {float(cs[bg].mean()):+.2f} / objet {float(cs[ob].mean()) if ob.any() else float('nan'):+.2f}")
        log("   RESSEMBLANCE d'une case de l'image 8 avec la case de l'image 7 décalée de (dx, dy) cases (cosinus, dé-datés) : " + " | ".join(res))
    log(f"total {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
