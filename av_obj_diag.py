"""
LE BÉBÉ SÉPARE-T-IL LES OBJETS AUTREMENT QUE PAR LA COULEUR ? (bébé GELÉ ; les petites têtes ne sont que des instruments de mesure)
Diagnostic récurrent : il « se fie à la couleur » (les objets qui se touchent fusionnent, cf. slots). Ici on le mesure directement.
Scènes du stade 0e à DEUX objets, au repos au départ (sa main babille et les pousse parfois), en 4 conditions :
  {couleurs différentes, MÊME couleur} × {séparés (écart ≥ 2 px), qui SE TOUCHENT / se chevauchent (écart ≤ 1 px)}
(av_world0 : nobj=(2, 2), same_col, touch, masks — défauts = monde inchangé). Le contact est MESURÉ image par image sur les masques
visibles (la main peut écarter ou rapprocher les objets) ; les images entre les deux (1 < écart < 2 px) sont écartées.
SONDE D'AFFINITÉ PAR PAIRES : deux cases (4×4 px) couvertes chacune ≥ 50 % par un objet (et < 10 % par sa main) appartiennent-elles au
MÊME objet ? petite tête MLP sur [z_i, z_j, z_i·z_j, |z_i − z_j|] (symétrisée), lue sur :
  vrais latents       : l'œil-cible sur toute la séquence (comme av_judge.teachN) — il voit la VIDÉO (le mouvement commun compte) ;
  imaginés h = 1..4   : latents imaginés de l'image f depuis le vrai passé ≤ f − h avec ses vrais gestes (voie d'av_move_diag) ;
                        tête apprise SUR l'imaginé (leçon ARC), et la tête du vrai lue sur l'imaginé (moyenne des h) ;
  pixels (image f)    : REPÈRE — les 48 valeurs brutes de la case ; pixels (f−2..f) : la même avec le mouvement (144 valeurs).
DISTANCE CONTRÔLÉE : les latents savent où est la case, une sonde pourrait dire « loin = autre objet ». AUC rapportée par tranche de
distance entre centres de cases (voisines ≤ 1.5 | proches ≤ 2.9 | loin) et apprentissage ÉQUILIBRÉ (autant de « même » que d'« autre »
dans chaque tranche ; tranche sans les deux : ignorée). Deux régimes :
  TRANSFERT     : apprise sur couleurs DIFFÉRENTES (séparés + se touchent ; --train sep : séparés seulement, peu de voisines « autre »),
                  évaluée sur les 4 conditions (scènes à part) — la couleur suffit à l'apprentissage, tient-elle sans couleur ?
  MÊME COULEUR  : apprise sur la même couleur (scènes à part) — l'information « quel objet » est-elle LISIBLE sans couleur ?
CRITÈRE : « MÊME couleur, se touchent, voisines » — les pixels ne peuvent pas séparer deux objets de même couleur collés (≈ 0.5) ;
latents nettement > pixels (au moins le régime MÊME COULEUR, idéalement aussi en TRANSFERT) ⇒ le bébé a une notion d'objet au-delà
de la couleur ; latents ≈ pixels ≈ 0.5 ⇒ le diagnostic « il se fie à la couleur » est confirmé.
BONUS (--count 1) : COMBIEN D'OBJETS (1/2/3) depuis les latents moyennés sur les cases vs pixels (case moyenne, image entière), même
découpage par condition ; « compté en moins » = deux objets collés vus comme un seul.

  python av_obj_diag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17p_0e_210k.pt --n 3000
"""
import argparse, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, layout, H
from av_world0 import gen_world0, VIEW
from av_judge import auc

COND = ("diff., séparés", "diff., se touchent", "MÊME, séparés", "MÊME, se touchent"); ABR = ("diff/sép", "diff/touche", "MÊME/sép", "MÊME/touche")
BINS = ("voisines", "proches", "loin")                                   # distance entre centres de cases : ≤ 1.5 | ≤ 2.9 | au-delà


def dilate(b, r):
    """b (..., H, H) bool -> dilaté de r px (carré)."""
    Hh = b.shape[-1]; p = np.pad(b, [(0, 0)] * (b.ndim - 2) + [(r, r), (r, r)]); out = np.zeros_like(b)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1): out |= p[..., dy:dy + Hh, dx:dx + Hh]
    return out


def contact(MK, thr=77):
    """MK (n, F, K+1, H, H) uint8 parts visibles -> (n, F) int8 : 1 deux objets se touchent (écart ≤ 1 px), 0 tous séparés (≥ 2 px), −1 entre les deux."""
    ob = MK[:, :, :-1] >= thr; d1, d2 = dilate(ob, 1), dilate(ob, 2); t1 = np.zeros(ob.shape[:2], bool); n2 = np.zeros_like(t1)
    for a_ in range(ob.shape[2]):
        for b_ in range(a_ + 1, ob.shape[2]):
            t1 |= (d1[:, :, a_] & ob[:, :, b_]).any((-1, -2)); n2 |= (d2[:, :, a_] & ob[:, :, b_]).any((-1, -2))
    return np.where(t1, 1, np.where(n2, -1, 0)).astype(np.int8)


@torch.no_grad()
def obj_data(m, teachN, imagine, cfg, st, dev, n, seed, nobj, FR, hmax, wkw, imag=True, pool=False, chunk=250, bs=32):
    """scènes d'examen -> dict : ZR (n, nF, npf, d) vrais latents [pool : (n, nF, d) moyennés] ; ZI liste h -> (n, nF, npf, d) imaginés ;
    PX (n, nF, npf, 48) / PX3 (n, nF, npf, 144) pixels des cases (image f / f−2..f) ; IMG (n, nF, H·H·3) si pool ; FC (n, nF, K+1, npf)
    part de chaque case couverte par l'objet k (K = main) ; CT (n, nF) contact ; SC (n,) même couleur ; NO (n,) nombre d'objets."""
    T = int(cfg.get("T", 16)); P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; md, fr = layout(nv, npf); v = VIEW["0e"]; nF = len(FR)
    ix = torch.arange(nv, device=dev); id_ = lambda msk: torch.from_numpy(np.where(msk)[0]).to(dev)
    gat = lambda tok, i_: torch.gather(tok, 1, i_.unsqueeze(-1).expand(-1, -1, tok.size(-1)))
    cells = lambda X: X.reshape(*X.shape[:2], nP, P_, nP, P_, 3).transpose(0, 1, 2, 4, 3, 5, 6).reshape(*X.shape[:2], npf, P_ * P_ * 3)
    FR3 = np.array([[f - 2, f - 1, f] for f in FR]); T0 = sorted({f - h for f in FR for h in range(1, hmax + 1)})
    R = {k: [] for k in ("ZR", "PX", "PX3", "IMG", "FC", "CT", "SC", "NO")}; R["ZI"] = [[] for _ in range(hmax)]
    for c0 in range(0, n, chunk):
        w0 = gen_world0(min(chunk, n - c0), "0e", T, H, seed=seed * 1000 + c0 // chunk, nobj=nobj, same_col=0.5, touch=0.5, masks=True, **wkw)
        MK = w0["MASK"][:, FR]; nc = len(MK); K1 = MK.shape[2]
        R["FC"].append((MK.reshape(nc, nF, K1, nP, P_, nP, P_).mean((4, 6)) / 255).reshape(nc, nF, K1, npf).astype(np.float16))
        R["CT"].append(contact(MK)); R["SC"].append(w0["SAMECOL"]); R["NO"].append(w0["NOBJ"]); w = to_np([w0]); del w0, MK
        R["PX"].append(cells(w["X"][:, FR])); R["PX3"].append(cells(w["X"][:, FR3.reshape(-1)]).reshape(nc, nF, 3, npf, -1).transpose(0, 1, 3, 2, 4).reshape(nc, nF, npf, -1))
        if pool: R["IMG"].append(w["X"][:, FR].reshape(nc, nF, -1))
        for i in range(0, nc, bs):
            b = to_torch(w, slice(i, i + bs)); B = len(b["X"]); tok, cmd = to_tok0(b, st, v, P_, dev)
            zr = teachN(tok[:, :nv], ix.expand(B, -1)).view(B, T, npf, -1)[:, FR]; R["ZR"].append((zr.mean(2) if pool else zr).half().cpu())
            if not imag: continue
            zi = {}
            for t0 in T0:                                                # même passé vu -> un seul contexte pour tous les (f, h) avec f − h = t0
                ci = id_(fr <= t0).expand(B, -1); ctx = m.context(gat(tok, ci), ci)
                for q, f in enumerate(FR):
                    if 1 <= f - t0 <= hmax:
                        ti = id_((fr == f) & (md == 0)).expand(B, -1); c = cmd.clone(); c[:, f + 1:] = 0   # ses vrais gestes jusqu'à f
                        zi[(f - t0, q)] = imagine(ctx, ti, c, ci).float().view(B, npf, -1).half().cpu()
            for h in range(1, hmax + 1): R["ZI"][h - 1].append(torch.stack([zi[(h, q)] for q in range(nF)], 1))
    out = {k: (torch.cat(x) if isinstance(x[0], torch.Tensor) else np.concatenate(x)) for k, x in R.items() if k != "ZI" and len(x)}
    out["ZI"] = [torch.cat(x) for x in R["ZI"] if len(x)]; return out


def make_pairs(FC, CT, SC, nP):
    """paires de cases d'une même image, chacune couverte ≥ 50 % par un objet (et < 10 % par sa main) -> S scène, Q image, I, J cases,
    Y même objet ?, B tranche de distance, C condition (0 diff. séparés, 1 diff. se touchent, 2 MÊME séparés, 3 MÊME se touchent)."""
    lab = np.full(FC.shape[:2] + FC.shape[3:], -1, np.int8); ok_h = FC[:, :, -1] < 0.1
    for k in range(FC.shape[2] - 1): lab[(FC[:, :, k] >= 0.5) & ok_h] = k
    iu, ju = np.triu_indices(FC.shape[-1], 1); li, lj = lab[:, :, iu], lab[:, :, ju]
    S, Q, p_ = np.nonzero((li >= 0) & (lj >= 0) & (CT[:, :, None] >= 0)); I, J = iu[p_], ju[p_]
    D = np.hypot(I // nP - J // nP, I % nP - J % nP)
    return dict(S=S, Q=Q, I=I, J=J, Y=li[S, Q, p_] == lj[S, Q, p_], B=np.digitize(D, (1.5, 2.9)), C=2 * SC[S].astype(int) + CT[S, Q])


class PairHead(nn.Module):
    """(case i, case j) -> logit « même objet » ; [z_i, z_j, z_i·z_j, |z_i − z_j|], symétrisé. Latents : layer_norm ; pixels : standardisés (mu, sd)."""
    def __init__(s, din, mu=None, sd=None, h=256):
        super().__init__(); s.std = mu is not None
        if s.std: s.register_buffer("mu", mu); s.register_buffer("sd", sd)
        s.net = nn.Sequential(nn.Linear(4 * din, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def f(s, z): return (z.float() - s.mu) / s.sd if s.std else F.layer_norm(z.float(), (z.size(-1),))
    def g(s, a, b): return s.net(torch.cat([a, b, a * b, (a - b).abs()], -1)).squeeze(-1)
    def forward(s, zi, zj): a, b = s.f(zi), s.f(zj); return 0.5 * (s.g(a, b) + s.g(b, a))


def norm_stats(Z, scenes):
    """pixels (uint8) : (mu, sd) par valeur sur les cases des scènes d'apprentissage (échantillon) ; latents : (None, None) -> layer_norm."""
    if Z.dtype != torch.uint8: return None, None
    x = Z[scenes.to(Z.device)].reshape(-1, Z.size(-1)); x = x[torch.randperm(len(x), device=x.device)[:200000]].float()
    return x.mean(0), x.std(0) + 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=3000, help="scènes à DEUX objets (4 conditions mêlées)")
    p.add_argument("--seed", type=int, default=31); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hmax", type=int, default=4, help="imaginés à h = 1..hmax"); p.add_argument("--plan_c", type=int, default=3, help="passé vu minimal (images 0..plan_c)")
    p.add_argument("--train", type=str, default="diff", choices=["diff", "sep"], help="régime TRANSFERT : couleurs différentes (séparés + se touchent) ou séparés seulement")
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--count", type=int, default=1, help="1 = BONUS : combien d'objets (1/2/3)"); p.add_argument("--n_count", type=int, default=0, help="scènes du bonus (0 = --n)")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    teachN = lambda x, ix: F.layer_norm(tgt(x, ix).float(), (d,)); imagine = m.pred                       # étage du haut (comme av_plan0 --read_level top)
    nP = H // cfg["P"]; FR = list(range(a.plan_c + a.hmax, T, 2))
    wkw = {**cfg.get("wkw", {}), "p_out": 0.0, "p_parent": 0.0, "kick": 0.0, "nohand": False}; wkw.pop("circ", None); wkw.pop("same_col", None)
    log(f"OBJETS AU-DELÀ DE LA COULEUR (bébé gelé, pas {ck['state']['it']}) : {a.n} scènes à 2 objets, images {FR}, imaginés h = 1..{a.hmax} | {time.time() - t_:.0f}s")
    D = obj_data(m, teachN, imagine, cfg, st, dev, a.n, a.seed, (2, 2), FR, a.hmax, wkw)
    Pr = make_pairs(D["FC"], D["CT"], D["SC"], nP); ntr = int(0.8 * a.n); tr_s = Pr["S"] < ntr; te = np.where(~tr_s)[0]
    nobjc = ((D["FC"][:, :, :-1] >= 0.5) & (D["FC"][:, :, -1:] < 0.1)).sum(-1)
    log(f"   données : {time.time() - t_:.0f}s | images se touchent {np.mean(D['CT'] == 1):.0%} / séparées {np.mean(D['CT'] == 0):.0%} / entre les deux (écartées) "
        f"{np.mean(D['CT'] == -1):.0%} | scènes même couleur {D['SC'].mean():.0%} | cases ≥ 50 % par objet {nobjc[:, :, :2].mean():.1f} | paires {len(Pr['Y'])} "
        f"(même objet {Pr['Y'].mean():.0%}) | par condition " + " / ".join(f"{COND[c]} {np.sum(Pr['C'] == c)}" for c in range(4)))
    Yt = torch.from_numpy(Pr["Y"]).float(); Ix = {k: torch.from_numpy(Pr[k]) for k in ("S", "Q", "I", "J")}

    def feats(Z, sel):
        s_, q_ = Ix["S"][sel], Ix["Q"][sel]; return Z[s_, q_, Ix["I"][sel]].to(dev), Z[s_, q_, Ix["J"][sel]].to(dev)

    def fit(Z, conds, seed):
        torch.manual_seed(seed); msk = tr_s & np.isin(Pr["C"], conds); grp = []
        for b_ in range(3):
            g0, g1 = (np.where(msk & (Pr["B"] == b_) & (Pr["Y"] == y_))[0] for y_ in (False, True))
            if min(len(g0), len(g1)) >= 32: grp += [torch.from_numpy(g0), torch.from_numpy(g1)]   # tranche équilibrée « même » / « autre »
        assert grp, "pas assez de paires pour apprendre (augmenter --n)"
        mu, sd = norm_stats(Z, torch.from_numpy(np.unique(Pr["S"][msk])))
        net = PairHead(Z.size(-1), *((mu.to(dev), sd.to(dev)) if mu is not None else ())).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
        for _ in range(a.steps):
            sel = torch.cat([g[torch.randint(0, len(g), (128,))] for g in grp])
            l = F.binary_cross_entropy_with_logits(net(*feats(Z, sel)), Yt[sel].to(dev)); opt.zero_grad(); l.backward(); opt.step(); sch.step()
        return net.eval()

    @torch.no_grad()
    def read(net, Z):                            # -> (4, 3) AUC par condition × tranche de distance (NaN : < 20 paires d'un côté)
        lg = torch.cat([net(*feats(Z, torch.from_numpy(c))).cpu() for c in np.array_split(te, max(1, len(te) // 8192))]).numpy(); y = Pr["Y"][te]
        r = np.full((4, 3), np.nan)
        for c in range(4):
            for b_ in range(3):
                s_ = (Pr["C"][te] == c) & (Pr["B"][te] == b_)
                if min(y[s_].sum(), (~y[s_]).sum()) >= 20: r[c, b_] = auc(lg[s_], y[s_])
        return r

    feat_sets = [("vrais latents", D["ZR"])] + [(f"imaginés h={h + 1}", D["ZI"][h]) for h in range(a.hmax)] + \
                [("pixels de la case (image f)", torch.from_numpy(D["PX"])), ("pixels de la case (f−2..f)", torch.from_numpy(D["PX3"]))]
    yte, cte, bte = Pr["Y"][te], Pr["C"][te], Pr["B"][te]
    head = f"   {'':>34s} | " + " | ".join(f"{c:^26s}" for c in COND) + "\n" + f"   {'lecture':>34s} | " + " | ".join(" ".join(f"{b_:>8s}" for b_ in BINS) for _ in COND)
    cnt = f"   {'paires à part (même / autre)':>34s} | " + " | ".join(" ".join(f"{int(yte[(cte == c) & (bte == b_)].sum()):>4d}/{int((~yte[(cte == c) & (bte == b_)]).sum()):<3d}"
                                                                         for b_ in range(3)) for c in range(4))
    row = lambda nm, r: f"   {nm:>34s} | " + " | ".join(" ".join(f"{x:8.2f}" if np.isfinite(x) else f"{'—':>8s}" for x in r[c]) for c in range(4))
    for reg, conds in ((f"TRANSFERT (apprise sur couleurs différentes{', séparés seulement' if a.train == 'sep' else ''})", [0] if a.train == "sep" else [0, 1]),
                       ("MÊME COULEUR (apprise sur la même couleur)", [2, 3])):
        lines = []; R_im = []
        for nm, Z in feat_sets:
            Zd = Z.to(dev) if Z.numel() * Z.element_size() < 6e9 else Z; rs = []
            for sd_ in range(a.seeds):
                net = fit(Zd, conds, sd_); rs.append(read(net, Zd))
                if nm == "vrais latents": R_im += [read(net, Zi.to(dev)) for Zi in D["ZI"]]       # la tête du vrai, lue sur l'imaginé
                del net
            lines.append(row(nm + (" (tête de l'imaginé)" if nm.startswith("imag") else ""), np.nanmean(np.stack(rs), 0))); del Zd
            if nm == "vrais latents": lines.append(row(f"imaginés h=1..{a.hmax} (tête du VRAI)", np.nanmean(np.stack(R_im), 0)))
            torch.cuda.empty_cache() if dev.startswith("cuda") else None
        log(f"\n   AFFINITÉ PAR PAIRES — {reg} : AUC « même objet » (0.50 = hasard), moyenne de {a.seeds} têtes × {a.steps} pas | {time.time() - t_:.0f}s\n"
            + head + "\n" + "\n".join(lines) + "\n" + cnt)
    log("   critère : « MÊME, se touchent, voisines » — latents nettement > pixels (≈ 0.5) ⇒ notion d'objet au-delà de la couleur ; sinon il se fie à la couleur")
    del D, feat_sets

    if not a.count: return
    nc_ = a.n_count or a.n; Dc = obj_data(m, teachN, imagine, cfg, st, dev, nc_, a.seed + 1, (1, 3), FR, a.hmax, wkw, imag=False, pool=True)
    nF = len(FR); N_ = np.repeat(Dc["NO"][:, None], nF, 1); C_ = np.where(N_ == 1, 4, 2 * Dc["SC"][:, None].astype(int) + Dc["CT"]); C_[(N_ > 1) & (Dc["CT"] < 0)] = -1
    S_ = np.repeat(np.arange(nc_)[:, None], nF, 1); ntc = int(0.8 * nc_); Yn = (N_ - 1).reshape(-1); Yc = torch.from_numpy(Yn).long().to(dev)
    PXc = torch.from_numpy(Dc["PX"]).float().mean(2)                                                  # case moyenne (48)
    log(f"\n   COMBIEN D'OBJETS (bonus) : {nc_} scènes à 1–3 objets ({np.bincount(Dc['NO'], minlength=4)[1:]}) | {time.time() - t_:.0f}s")
    trc = np.where(((S_ < ntc) & (np.isin(C_, [4, 0] if a.train == "sep" else [4, 0, 1]))).reshape(-1))[0]; tec = np.where(((S_ >= ntc) & (C_ >= 0)).reshape(-1))[0]
    Cf, Nf = C_.reshape(-1), N_.reshape(-1); trt, tet = torch.from_numpy(trc).to(dev), torch.from_numpy(tec).to(dev)
    grp = [torch.from_numpy(trc[Yn[trc] == k]).to(dev) for k in range(3)]; res = {}
    for nm, Z in (("vrais latents moyennés", Dc["ZR"].float()), ("pixels : case moyenne", PXc), ("pixels : image entière", torch.from_numpy(Dc["IMG"]).float())):
        Z = Z.reshape(-1, Z.size(-1)).to(dev); ln = nm.startswith("vrais"); mu, sd = (None, None) if ln else (Z[trt].mean(0), Z[trt].std(0) + 1.0); acc = []
        for sd_ in range(a.seeds):
            torch.manual_seed(sd_); net = nn.Sequential(nn.Linear(Z.size(-1), 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 3)).to(dev)
            f_ = (lambda z: F.layer_norm(z, (z.size(-1),))) if ln else (lambda z: (z - mu) / sd)
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
            for _ in range(a.steps):
                sel = torch.cat([g[torch.randint(0, len(g), (128,), device=dev)] for g in grp])            # classes 1 / 2 / 3 équilibrées
                l = F.cross_entropy(net(f_(Z[sel])), Yc[sel]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
            with torch.no_grad(): pr = net.eval()(f_(Z[tet])).argmax(-1).cpu().numpy() + 1
            acc.append([(pr[Cf[tec] == c] == Nf[tec][Cf[tec] == c]).mean() if (Cf[tec] == c).any() else np.nan for c in (4, 0, 1, 2, 3)]
                       + [(pr[Cf[tec] == c] < Nf[tec][Cf[tec] == c]).mean() if (Cf[tec] == c).any() else np.nan for c in (0, 1, 2, 3)])
            del net
        res[nm] = np.nanmean(np.array(acc), 0); del Z
    log(f"   exactitude (hasard ≈ 0.33) ; condition = scènes à 2–3 objets ; « en moins » = part des scènes comptées avec TROP PEU d'objets\n"
        f"   {'lecture':>26s} | {'1 objet':>7s} | " + " | ".join(f"{c:>11s}" for c in ABR) + " | " + " | ".join(f"{'en moins ' + c:>20s}" for c in ABR) + "\n"
        + "\n".join(f"   {nm:>26s} | {r[0]:7.2f} | " + " | ".join(f"{x:11.2f}" for x in r[1:5]) + " | " + " | ".join(f"{x:20.2f}" for x in r[5:]) for nm, r in res.items())
        + f"\n   images à part par condition : 1 objet {np.sum(Cf[tec] == 4)} | " + " | ".join(f"{COND[c]} {np.sum(Cf[tec] == c)}" for c in range(4)) + f" | {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
