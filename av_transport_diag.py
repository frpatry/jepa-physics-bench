"""
AVANT L'IMAGINATION PAR OBJET (13 nov. 2026) — deux vérifications sans entraînement, bébé GELÉ :
 (1) SÉPARE-T-IL L'OBJET DU FOND ? son propre sens du groupe (tête co_proj du mouvement commun, aucune sonde) : une case d'objet ressemble-t-elle plus
     à une autre case du MÊME objet qu'à une case de FOND ? (jusqu'ici mesuré seulement objet contre objet, av_obj_diag) ; repère : cosinus de ses latents bruts.
     Cases : cœur d'objet (≥ 50 % couvert), bord (10–50 %), fond (< 2 % d'objet et de main).
 (2) TRANSPORTER MARCHE-T-IL À SA RÉSOLUTION ? ses VRAIS latents de l'image t, déplacés du VRAI déplacement de l'objet (positions du monde = instrument),
     comparés aux vrais latents de l'image t+k : mieux que recopier ? Fond recopié, cases quittées remplies du latent moyen du fond, objet transporté
     (interpolation sur la grille des cases). Mesuré là où ça compte (cases que l'objet occupe à t ou à t+k, hors main), par taille de déplacement et par horizon.
     transport ≪ copie ⇒ une imagination « un déplacement par groupe » tient avec ses cases de 4 px ; transport ≈ copie ⇒ ses latents ne se déplacent pas
     tels quels (ils portent l'endroit) : il faudra apprendre l'opération de transport, ou voir plus fin.
  python av_transport_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17s_0e_270k.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, layout, H
from av_world0 import gen_world0, VIEW
from av_judge import auc


def shift(x, d):
    """x (B, C, n, n), d (B, 2) déplacement (dx, dy) en CASES -> x transporté de d (interpolation bilinéaire, zéro hors champ)."""
    B, _, n, _ = x.shape; ys, xs = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(n, device=x.device), indexing="ij")
    gx = (xs[None] - d[:, 0, None, None] + 0.5) / n * 2 - 1; gy = (ys[None] - d[:, 1, None, None] + 0.5) / n * 2 - 1
    return F.grid_sample(x, torch.stack([gx, gy], -1).float(), mode="bilinear", padding_mode="zeros", align_corners=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=3000); p.add_argument("--seed", type=int, default=91)
    p.add_argument("--t0", type=int, default=7); p.add_argument("--ks", type=str, default="1,2,4,8"); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); av_phase0.T = T
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; md, fr = layout(nv, npf); v = VIEW["0e"]; ix = torch.arange(nv, device=dev); ks = [int(x) for x in a.ks.split(",")]
    wkw = {**cfg.get("wkw", {}), "p_out": 0.0, "p_parent": 0.0, "kick": 0.0, "nohand": False}; wkw.pop("circ", None); wkw.pop("same_col", None)
    FR = [a.t0, 11, 15]; id_ = lambda msk: torch.from_numpy(np.where(msk)[0]).to(dev); gat = lambda tok, i_: torch.gather(tok, 1, i_.unsqueeze(-1).expand(-1, -1, tok.size(-1)))
    log(f"AVANT L'IMAGINATION PAR OBJET (bébé gelé, pas {ck['state']['it']}) : {a.n} scènes à UN objet, objets × {wkw.get('obj_scale', 1.0)}")
    ZR, U, FO, FH, POS = [], [], [], [], []
    with torch.no_grad():
        for c0 in range(0, a.n, 250):
            w0 = gen_world0(min(250, a.n - c0), "0e", T, H, seed=a.seed * 1000 + c0 // 250, nobj=(1, 1), masks=True, **wkw); MK = w0["MASK"]; nc = len(MK)
            fc = (MK.reshape(nc, T, MK.shape[2], nP, P_, nP, P_).mean((4, 6)) / 255).reshape(nc, T, MK.shape[2], npf)
            FO.append(fc[:, :, 0].astype(np.float32)); FH.append(fc[:, :, -1].astype(np.float32)); POS.append(w0["POS"][:, :, 0] * H); w = to_np([w0]); del w0, MK
            for i in range(0, nc, 32):
                b = to_torch(w, slice(i, i + 32)); B = len(b["X"]); tok, _ = to_tok0(b, st, v, P_, dev)
                ZR.append(F.layer_norm(tgt(tok[:, :nv], ix.expand(B, -1)).float(), (d,)).view(B, T, npf, d).half().cpu())
                if hasattr(m, "co_proj"):
                    us = []
                    for f in FR:
                        ci = id_(fr <= f).expand(B, -1); zc = m.enc(gat(tok, ci), ci)[:, f * npf:(f + 1) * npf].float()
                        us.append(F.normalize(m.co_proj(F.layer_norm(zc, (d,))), dim=-1).half().cpu())
                    U.append(torch.stack(us, 1))
    ZR = torch.cat(ZR).float(); FO, FH, POS = (torch.from_numpy(np.concatenate(x)) for x in (FO, FH, POS)); U = torch.cat(U).float() if U else None
    log(f"   données : {time.time() - t_:.0f}s | cases de cœur d'objet par image {float((FO >= 0.5).float().sum(-1).mean()):.1f}, de bord {float(((FO >= 0.1) & (FO < 0.5)).float().sum(-1).mean()):.1f}")

    # ---------- (1) OBJET contre FOND : paires (case de cœur, autre case de la même image)
    iu, ju = np.triu_indices(npf, 1); iu, ju = torch.from_numpy(np.concatenate([iu, ju])), torch.from_numpy(np.concatenate([ju, iu]))   # paires ordonnées (i = cœur)
    dist = np.hypot((iu // nP - ju // nP).numpy(), (iu % nP - ju % nP).numpy()); bins = np.digitize(dist, (1.5, 2.9))
    def pair_auc(S):                              # S (n, nF, npf, k) vecteurs normalisés -> AUC par tranche de distance, pour « cœur–cœur » et « cœur–bord » contre « cœur–fond »
        fo = FO[:, FR]; fh = FH[:, FR]; core = (fo >= 0.5) & (fh < 0.1); edge = (fo >= 0.1) & (fo < 0.5) & (fh < 0.1); bg = (fo < 0.02) & (fh < 0.02)
        sim = torch.einsum("nfik,nfjk->nfij", S, S)[:, :, iu, ju]; ci_ = core[:, :, iu]; out = []
        for pos_ in (core, edge):
            row = []
            for b_ in range(3):
                sb = torch.from_numpy(bins == b_); pm = (ci_ & pos_[:, :, ju])[:, :, sb]; nm = (ci_ & bg[:, :, ju])[:, :, sb]; s_ = sim[:, :, sb]
                sp, sn = s_[pm].numpy(), s_[nm].numpy()
                if len(sp) > 20000: sp = np.random.default_rng(0).choice(sp, 20000, replace=False)
                if len(sn) > 20000: sn = np.random.default_rng(1).choice(sn, 20000, replace=False)
                row.append(auc(np.concatenate([sp, sn]), np.concatenate([np.ones(len(sp), bool), np.zeros(len(sn), bool)])) if min(len(sp), len(sn)) >= 20 else np.nan)
            out.append(row)
        return out
    log("\n(1) OBJET contre FOND — une case de cœur d'objet ressemble-t-elle plus à une case du MÊME objet qu'à une case de FOND ? (AUC, 0.50 = hasard)")
    log(f"   {'':>44s} | cœur–cœur : voisines  proches  loin | cœur–BORD : voisines  proches  loin")
    for nm, S in ([("SON sens du groupe (co_proj, aucune sonde)", U)] if U is not None else []) + [("repère : cosinus de ses latents bruts", F.normalize(ZR[:, FR], dim=-1))]:
        r = pair_auc(S); log(f"   {nm:>44s} |             " + "  ".join(f"{x:7.2f}" for x in r[0]) + " |             " + "  ".join(f"{x:7.2f}" for x in r[1]))

    # ---------- (2) TRANSPORT : latents de t déplacés du vrai déplacement, comparés aux latents de t+k
    log(f"\n(2) TRANSPORT — ses vrais latents de l'image {a.t0} déplacés du VRAI déplacement de l'objet, contre recopier (écart moyen aux vrais latents de t+k,\n"
        f"    sur les cases que l'objet occupe à t ou à t+k, hors main ; « cible » = écart entre deux images quand RIEN ne bouge)")
    t0 = a.t0; Z0 = ZR[:, t0].view(-1, nP, nP, d).permute(0, 3, 1, 2); M0 = FO[:, t0].view(-1, 1, nP, nP)
    bgm = ((FO[:, t0] < 0.02) & (FH[:, t0] < 0.02)).float(); bgl = (ZR[:, t0] * bgm.unsqueeze(-1)).sum(1) / bgm.sum(1, keepdim=True).clamp_min(1)   # latent moyen du fond (n, d)
    edges = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 99.0); names = ("< 0.5 px (immobile)", "0.5–1 px", "1–2 px", "2–4 px", "4–8 px", "> 8 px")
    log(f"   {'déplacement réel':>20s} | " + " | ".join(f"k={k} : n  copie → transport" for k in ks))
    rows = {nm: [] for nm in names}; tot = []
    for k in ks:
        D = (POS[:, t0 + k] - POS[:, t0]).float(); dn = D.norm(dim=-1); Dc = D / P_
        Mk = shift(M0, Dc); Zo = shift(Z0 * M0, Dc) / Mk.clamp_min(1e-3)
        base = torch.where(M0 > 0.02, bgl[:, :, None, None].expand(-1, -1, nP, nP), Z0)                      # fond recopié, cases quittées = fond moyen
        pred = (Mk * Zo + (1 - Mk) * base).permute(0, 2, 3, 1).reshape(-1, npf, d); Zk = ZR[:, t0 + k]
        ev = (((FO[:, t0] >= 0.1) | (FO[:, t0 + k] >= 0.1)) & (FH[:, t0] < 0.02) & (FH[:, t0 + k] < 0.02)).float()
        e_t = ((pred - Zk).abs().mean(-1) * ev).sum(1) / ev.sum(1).clamp_min(1); e_c = ((ZR[:, t0] - Zk).abs().mean(-1) * ev).sum(1) / ev.sum(1).clamp_min(1); ok = ev.sum(1) > 0
        for j, nm in enumerate(names):
            s_ = ok & (dn >= edges[j]) & (dn < edges[j + 1]); rows[nm].append(f"{int(s_.sum()):5d}  {float(e_c[s_].mean()):.3f} → {float(e_t[s_].mean()):.3f}" if int(s_.sum()) >= 10 else f"{int(s_.sum()):5d}        —      ")
        mv = ok & (dn >= 1.0); tot.append((k, int(mv.sum()), float(e_c[mv].mean()), float(e_t[mv].mean()), float((e_t[mv] < e_c[mv]).float().mean())))
    for nm in names: log(f"   {nm:>20s} | " + " | ".join(rows[nm]))
    log("   objets déplacés d'au moins 1 px : " + " | ".join(f"k={k} (n={n_}) copie {c_:.3f} → transport {t__:.3f} ({1 - t__ / c_:+.0%}), mieux que la copie dans {b_:.0%}" for k, n_, c_, t__, b_ in tot))
    log("   lecture : transport ≪ copie ⇒ « un déplacement par groupe » tient à sa résolution ; transport ≈ copie ⇒ ses latents portent l'endroit : opération à apprendre ou vue plus fine")
    log(f"total {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
