"""
OÙ EST L'OBJET ? — DANS QUELLES CONDITIONS L'ŒIL LE SITUE-T-IL MAL ? (21 nov. 2026, demande user ; bébé GELÉ, les lecteurs ne sont que des instruments de mesure)
Constat (bébé v4, examens 35k -> 55k) : la sonde « objet » de l'examen ne s'améliore plus (17.7 % -> 19.5 % de la largeur) alors que la main progresse (15.5 % -> 10.6 %).
Mais ce chiffre mélange tout, et l'instrument est fragile : (1) ses scènes viennent de l'ANCIEN jeu libre (gen_world0 sans réglage), pas du jeu de poussée ;
(2) l'objet est mesuré sur 8 images au hasard, même caché sous la main ; (3) lecteur LINÉAIRE sur l'image entière, appris sur ~1 000 images.
Ici, sur le même œil-cible gelé (comme l'examen : tgt, toute la séquence, vue seule), l'erreur de position (px, image de 32 px) est DÉCOUPÉE par condition :
  nombre d'objets (1/2/3) | visibilité (en entier / en partie sous sa main / coupé par le bord / autre / presque invisible) | mouvement (au repos / poussé par lui /
  bouge sans lui) | forme (rond, carré, triangle, T) | taille | distance à sa main — dans DEUX mondes : son JEU DE POUSSÉE (réglages lus dans la sauvegarde) et
  l'ANCIEN jeu libre (le monde de l'examen).
Deux lecteurs, chacun lu aussi sur les PIXELS bruts (repère) :
  LOCAL    : chaque case lit dans SON latent « l'objet demandé est-il chez moi ? » et « où dans ma case ? » (comme LocalReader d'av_plan0) ; l'objet demandé est
             désigné par sa COULEUR moyenne (seule façon de dire « lequel » quand il y en a 2 ou 3). Appris sur les images où ≥ 10 % de l'objet est visible.
  LINÉAIRE : la sonde de l'examen (ridge sur l'image entière), scènes à UN objet seulement, mais avec beaucoup plus d'images.
Visibilité = part visible de l'objet (masques du générateur) / sa plus grande surface visible dans la séquence. Repère « au milieu » = toujours répondre la position
moyenne. Plusieurs sauvegardes -> mêmes scènes pour toutes (comparaison v4 / v3c).
  python av_objpos_diag.py --ckpts "/kaggle/input/**/bebe4.pt" "/kaggle/input/**/bebe3c.pt" --n 2000
"""
import argparse, copy, glob, time
import numpy as np, torch, torch.nn as nn
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, KEYS, H
from av_world0 import gen_world0, VIEW

FORMES = ("rond", "carré", "triangle", "T")
VIS = ("visible en entier (≥ 90 %)", "en partie sous SA MAIN", "coupé par le BORD du champ", "en partie caché, autre (objet, parent, main + bord)", "presque INVISIBLE (< 10 % : sous la main ou hors champ)")
MOUV = ("au repos", "POUSSÉ PAR LUI (contact senti, main à < 10 px)", "bouge sans lui (glisse, lancé, parent, autre objet)")
DIST = ("main à moins de 8 px", "main entre 8 et 16 px", "main à plus de 16 px")
PUSH_WKW = dict(p_obj=0.5, mob_size=0.16, p_parent=0.03, pause=False, echo=0.5, damp=0.5, shape_phys=True, push=1.0)   # run_bebe4.sh (si aucune sauvegarde ne le porte)


def scenes(n, wkw, seed, T, chunk=250):
    """n séquences du stade 0e -> sens (uint8 / float16) + instruments : HAND, RH, POS, NOBJ, SHAPE, TSRC, AREA (n,T,K) px² visibles, COL (n,K,3) couleur moyenne."""
    R = {k: [] for k in KEYS + ("HAND", "RH", "POS", "NOBJ", "SHAPE", "TSRC", "AREA", "COL")}
    for c0 in range(0, n, chunk):
        w0 = gen_world0(min(chunk, n - c0), "0e", T, H, seed=seed + c0 // chunk, masks=True, **wkw)
        MK = w0["MASK"][:, :, :-1].astype(np.float32) / 255; R["AREA"].append(MK.sum((-1, -2)))
        R["COL"].append(np.einsum("ntkhw,nthwc->nkc", MK, w0["X"]) / np.maximum(MK.sum((1, 3, 4)), 1e-6)[..., None]); del MK
        w = to_np([w0])
        for k in KEYS: R[k].append(w[k])
        for k in ("HAND", "RH", "POS", "NOBJ", "SHAPE", "TSRC"): R[k].append(w0[k])
    return {k: np.concatenate(v) for k, v in R.items()}


def samples(W, FR):
    """un échantillon = (séquence, image, objet). -> dict de tableaux alignés : si, fi, y (2), col (3) et les conditions."""
    FRa = np.array(FR); K = W["POS"].shape[2]; pos, prev = W["POS"][:, FRa], W["POS"][:, FRa - 1]
    with np.errstate(invalid="ignore"):
        speed = np.linalg.norm(pos - prev, axis=-1) * H; amax = W["AREA"].max(1)[:, None]; frac = W["AREA"][:, FRa] / np.maximum(amax, 1e-6)
        hand = W["HAND"][:, FRa][:, :, None]; dh = np.linalg.norm(pos - hand, axis=-1) * H; rest = np.sqrt(amax / np.pi) / H
        over = np.abs(pos - hand).max(-1) < W["RH"][:, None, None] * 0.85 + rest; edge = np.minimum(pos, 1 - pos).min(-1) < rest; inside = ((pos > 0) & (pos < 1)).all(-1)
        act = W["TSRC"] == 1; rec = act.copy(); rec[:, 1:] |= act[:, :-1]; rec[:, 2:] |= act[:, :-2]
        pushed = (speed > 0.05) & rec[:, FRa][:, :, None] & (dh < 10)
        vis = np.where(frac >= 0.9, 0, np.where(frac < 0.1, 4, np.where(over & ~edge & inside, 1, np.where(edge & ~over, 2, 3))))
        mot = np.where(speed < 0.05, 0, np.where(pushed, 1, 2)); dist = np.digitize(dh, [8, 16])
    ok = (np.arange(K)[None, None] < W["NOBJ"][:, None, None]) & (amax >= 4) & np.isfinite(pos).all(-1) & np.isfinite(prev).all(-1)
    si, fi, ki = np.where(ok); am = amax[si, 0, ki]; kind = W["SHAPE"][si, ki]; big = np.zeros(len(si), bool)
    for q in (kind == 3, kind != 3):
        if q.any(): big[q] = am[q] >= np.median(am[q])
    return dict(si=si, fi=fi, yall=pos[si, fi].astype(np.float32), y=pos[si, fi, ki].astype(np.float32), col=W["COL"][si, ki].astype(np.float32), nobj=W["NOBJ"][si], kind=kind, big=big,
                vis=vis[si, fi, ki], mot=mot[si, fi, ki], dist=dist[si, fi, ki], frac=frac[si, fi, ki], speed=speed[si, fi, ki])


@torch.no_grad()
def feats(tgt, W, st, P_, dev, T, FR, bs=16):
    """œil-cible gelé sur toute la séquence (vue seule, comme l'examen) -> latents (n, F, npf, d) et pixels des cases (n, F, npf, 48), float16."""
    nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; Z, PX = [], []
    for i in range(0, len(W["X"]), bs):
        tok, _ = to_tok0(to_torch(W, slice(i, i + bs)), st, v, P_, dev); B = len(tok); ix = torch.arange(nv, device=dev).expand(B, -1)
        Z.append(tgt(tok[:, :nv], ix).float().view(B, T, npf, -1)[:, FR].half().cpu()); PX.append(tok[:, :nv, :P_ * P_ * 3].view(B, T, npf, -1)[:, FR].half().cpu())
    return torch.cat(Z), torch.cat(PX)


class Local(nn.Module):
    """chaque case : score « l'objet de cette couleur est chez moi » + décalage dans la case ; position = moyenne pondérée (softmax) des centres + décalages."""
    def __init__(s, din, nP, h=128):
        super().__init__(); s.nP = nP; s.f = nn.Sequential(nn.Linear(din + 3, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.sc, s.off = nn.Linear(h, 1), nn.Linear(h, 2)
        r, c = torch.meshgrid(torch.arange(nP), torch.arange(nP), indexing="ij"); s.register_buffer("ctr", torch.stack([(c.flatten() + 0.5) / nP, (r.flatten() + 0.5) / nP], -1).float())
    def forward(s, x, col):
        f = s.f(torch.cat([x, col[:, None].expand(-1, x.size(1), -1)], -1)); w = s.sc(f).softmax(1)
        return (w * (s.ctr[None] + torch.tanh(s.off(f)) * (0.75 / s.nP))).sum(1)


def fit_local(Zf, S, tr, te, dev, steps, bs, seed):
    """Zf (n, F, npf, din) float16 ; apprend sur tr (indices d'échantillons), -> sur te : erreur (px) vers l'objet demandé, et vers l'objet LE PLUS PROCHE de sa réponse
    (petite alors que la première est grande = il situe bien DES objets mais se trompe d'objet)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); si, fi = torch.from_numpy(S["si"]), torch.from_numpy(S["fi"])
    y, col = torch.from_numpy(S["y"]), (torch.from_numpy(S["col"]) - 0.5) * 2; tr = torch.from_numpy(tr); sub = tr[torch.randperm(len(tr), generator=g)[:2000]]
    x0 = Zf[si[sub], fi[sub]].float().reshape(-1, Zf.size(-1)); mu, sd = x0.mean(0), x0.std(0).clamp_min(1e-3)
    get = lambda idx: (((Zf[si[idx], fi[idx]].float() - mu) / sd).to(dev), col[idx].to(dev), y[idx].to(dev))
    net = Local(Zf.size(-1), int(round(Zf.size(2) ** 0.5))).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, c, t = get(tr[torch.randint(len(tr), (bs,), generator=g)]); loss = (net(x, c) - t).norm(dim=-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    net.eval(); out = []; te = torch.from_numpy(te)
    with torch.no_grad():
        for i in range(0, len(te), 512):
            x, c, t = get(te[i:i + 512]); out.append(net(x, c).cpu())
    pr = torch.cat(out).numpy(); te = te.numpy()
    with np.errstate(invalid="ignore"): dn = np.linalg.norm(pr[:, None] - S["yall"][te], axis=-1) * H
    return np.linalg.norm(pr - S["y"][te], axis=-1) * H, np.nanmin(dn, 1)


def ridge(X, Y, itr, iva, ite, dev):
    """sonde LINÉAIRE de l'examen (ridge exacte, forme duale, λ choisi sur la validation) -> erreur (px) sur ite."""
    Xt = X[itr].to(dev).float(); mu = Xt.mean(0); Xt = Xt - mu; t = Y[itr].to(dev); tm = t.mean(0); K = Xt @ Xt.T; s = K.diagonal().mean(); best = None
    Xv, Xe = X[iva].to(dev).float() - mu, X[ite].to(dev).float() - mu
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        Wr = Xt.T @ torch.linalg.solve(K + lam * s * torch.eye(len(K), device=dev), t - tm); ev = (Xv @ Wr + tm - Y[iva].to(dev)).norm(dim=-1).mean().item()
        if best is None or ev < best[0]: best = (ev, Wr)
    return ((Xe @ best[1] + tm - Y[ite].to(dev)).norm(dim=-1) * H).cpu().numpy()


def rows(S, te):
    g = lambda k: S[k][te]; nb, vis, mot, dist, kind, big = g("nobj"), g("vis"), g("mot"), g("dist"), g("kind"), g("big"); R = [("TOUT", np.ones(len(te), bool))]
    R += [(f"scènes à {k} objet{'s' if k > 1 else ''}", nb == k) for k in (1, 2, 3)] + [(VIS[k], vis == k) for k in range(5)] + [(MOUV[k], mot == k) for k in range(3)]
    sp_ = g("speed"); R += [(f"vitesse de l'objet : {nm_}", (sp_ >= lo) & (sp_ < hi)) for nm_, lo, hi in (("0.05 à 0.5 px par image", 0.05, 0.5), ("0.5 à 1.5 px par image", 0.5, 1.5), ("plus de 1.5 px par image", 1.5, 1e9))]
    R += [(f"forme : {FORMES[k]}", kind == k) for k in range(4)] + [("taille : petits", ~big), ("taille : gros", big)] + [(DIST[k], dist == k) for k in range(3)]
    R += [("CAS FACILE : 1 objet, visible en entier, au repos, main à plus de 8 px", (nb == 1) & (vis == 0) & (mot == 0) & (dist > 0)),
          ("CAS PUSH-T : poussé par lui, au moins à moitié visible", (mot == 1) & (g("frac") >= 0.5)),
          ("poussé par lui, visible en entier", (mot == 1) & (vis == 0)), ("au repos, visible en entier, main à moins de 8 px", (mot == 0) & (vis == 0) & (dist == 0))]
    return R


def table(log, R, cols, nmin=20):
    """cols : [(nom, erreurs px alignées sur le test)]."""
    log(f"      {'condition':<72s} {'n':>6s} | " + " | ".join(f"{nm:>14s}" for nm, _ in cols) + " | médiane (dernière colonne)")
    for nm, q in R:
        if q.sum() < nmin: log(f"      {nm:<72s} {int(q.sum()):6d} | trop peu"); continue
        log(f"      {nm:<72s} {int(q.sum()):6d} | " + " | ".join(f"{e[q].mean():11.2f} px" for _, e in cols) + f" | {np.median(cols[-1][1][q]):.2f} px")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="chemins ou motifs (** permis) ; par motif, la sauvegarde la plus avancée"); p.add_argument("--n", type=int, default=2000, help="séquences par monde")
    p.add_argument("--seed", type=int, default=77_000); p.add_argument("--steps", type=int, default=3000); p.add_argument("--bs", type=int, default=256)
    p.add_argument("--calm", type=float, default=0.35); p.add_argument("--p_out", type=float, default=0.55, help="jeu de poussée : réglages de la fin de l'étape 0e (journal du pas 56 000)")
    p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t0 = time.time(); log = lambda s_: print(s_, flush=True)
    CK = []
    for pat in a.ckpts:                                              # par motif : la plus avancée
        C = {f: torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(pat, recursive=True))}
        if not C: log(f"(aucune sauvegarde pour {pat})"); continue
        f = max(C, key=lambda f_: int(C[f_]["state"]["it"])); CK.append((f, C[f])); del C
    wkw = next(({k: x for k, x in ck["cfg"].get("wkw", {}).items() if k not in ("nohand", "same_col", "circ")} for _, ck in CK if ck["cfg"].get("wkw", {}).get("push", 0) > 0), PUSH_WKW)
    T = int(CK[0][1]["cfg"].get("T", 16)); FR = list(range(1, T, 2))
    MONDES = (("SON JEU DE POUSSÉE", {**wkw, "calm": a.calm, "p_out": a.p_out}), ("ANCIEN JEU LIBRE (monde de l'examen)", {}))
    log(f"OÙ EST L'OBJET ? {a.n} séquences par monde, images {FR} | sauvegardes : " + " ; ".join(f"{f} (pas {ck['state']['it']})" for f, ck in CK) + f"\n   jeu de poussée : {MONDES[0][1]}")
    for mi, (mn, mk) in enumerate(MONDES):
        W = scenes(a.n, mk, a.seed + 1000 * mi, T); S = samples(W, FR); n = len(W["X"]); N = len(S["si"])
        prm = np.random.default_rng(5).permutation(n); part = np.zeros(n, int); part[prm[int(0.6 * n):int(0.75 * n)]] = 1; part[prm[int(0.75 * n):]] = 2; sp = part[S["si"]]
        seen = S["frac"] >= 0.1; tr = np.where((sp < 2) & seen)[0]; te = np.where(sp == 2)[0]; R = rows(S, te)
        one = S["nobj"] == 1; r_tr, r_va, r_te = (np.where((sp == k) & one)[0] for k in range(3)); R1 = rows(S, r_te)
        base = np.linalg.norm(S["y"][te] - S["y"][tr].mean(0), axis=-1) * H; base1 = np.linalg.norm(S["y"][r_te] - S["y"][r_tr].mean(0), axis=-1) * H
        log(f"\n================ MONDE : {mn} | {n} séquences, {N} (image, objet) | objets par séquence {W['NOBJ'].mean():.2f} | {time.time() - t0:.0f}s")
        log("   part des (image, objet) : " + " ; ".join(f"{VIS[k].split(' (')[0]} {np.mean(S['vis'] == k):.0%}" for k in range(5)) + " || " + " ; ".join(f"{MOUV[k].split(' (')[0]} {np.mean(S['mot'] == k):.0%}" for k in range(3))
            + f" || vitesse des poussés par lui : médiane {np.median(S['speed'][S['mot'] == 1]) if (S['mot'] == 1).any() else 0:.2f} px/image")
        Y = torch.from_numpy(S["y"]); px_cols = None
        for f, ck in CK:
            cfg = ck["cfg"]; st = {k: (x.to(dev) if torch.is_tensor(x) else x) for k, x in ck["norm"].items()}; m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); del m
            Z, PX = feats(tgt, W, st, cfg["P"], dev, T, FR); tag = f"{f.split('/')[-1]} pas {ck['state']['it']}"; log(f"\n   ---- {tag} | latents {tuple(Z.shape)} | {time.time() - t0:.0f}s")
            flat = lambda A_, idx: A_[torch.from_numpy(S["si"][idx]), torch.from_numpy(S["fi"][idx])].reshape(len(idx), -1)
            if px_cols is None:                                      # pixels : une seule fois (mêmes scènes pour toutes les sauvegardes ; vue 0e = image nette)
                px_cols = (fit_local(PX, S, tr, te, dev, a.steps, a.bs, 0)[0], ridge(torch.cat([flat(PX, r_tr), flat(PX, r_va), flat(PX, r_te)]), torch.cat([Y[r_tr], Y[r_va], Y[r_te]]),
                           np.arange(len(r_tr)), len(r_tr) + np.arange(len(r_va)), len(r_tr) + len(r_va) + np.arange(len(r_te)), dev))
            e_loc, e_near = fit_local(Z, S, tr, te, dev, a.steps, a.bs, 0); log(f"   LECTEUR LOCAL guidé par la couleur ({len(tr)} exemples d'apprentissage, lu sur {len(te)} d'autres séquences) | {time.time() - t0:.0f}s")
            table(log, R, [("« au milieu »", base), ("pixels", px_cols[0]), ("lat. + proche", e_near), ("LATENTS", e_loc)])
            log("      (« lat. + proche » = distance de sa réponse à l'objet le plus proche, quel qu'il soit : petite quand LATENTS est grande = il se trompe d'objet)")
            e_lin = ridge(torch.cat([flat(Z, r_tr), flat(Z, r_va), flat(Z, r_te)]), torch.cat([Y[r_tr], Y[r_va], Y[r_te]]), np.arange(len(r_tr)), len(r_tr) + np.arange(len(r_va)), len(r_tr) + len(r_va) + np.arange(len(r_te)), dev)
            log(f"   SONDE LINÉAIRE de l'examen, scènes à UN objet ({len(r_tr)} exemples, lu sur {len(r_te)}) : {e_lin.mean() / H:.1%} de la largeur (pixels {px_cols[1].mean() / H:.1%}) | {time.time() - t0:.0f}s")
            table(log, [r_ for r_ in R1 if not r_[0].startswith("scènes à")], [("« au milieu »", base1), ("pixels", px_cols[1]), ("LATENTS", e_lin)])
            del Z, PX, tgt
    log(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
