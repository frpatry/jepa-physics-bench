"""
JUGE APPRIS SUR SON PROPRE JEU (étape 3 du plan du 2026-11-10) : la tête « combien de gestes séparent cette scène de cette image-but »
validée par av_dist_diag.py (main au but vs T au but : 0 % au lieu de 52 %), préparée ici POUR LE PLANIFICATEUR (av_plan0 --judge).
Dans le planificateur, la scène jugée est une image IMAGINÉE (t+h depuis le vrai passé ≤ t) et le but est la suite « vrai passé ≤ t puis
l'image-but pour toujours » lue à l'image t+h. Le juge apprend donc, sur le JEU LIBRE seulement (aucune démonstration, aucun état d'oracle) :
  scène = l'image f = t+h d'une séquence vécue — ses VRAIS latents (œil-cible) et/ou ses latents IMAGINÉS avec ses vrais gestes
          (leçon ARC : un juge juste sur de vraies images peut mourir sur des images imaginées) ;
  but   = l'image d'un autre instant b SANS sa main, « pour toujours » après t, lue à l'image f ;
  étiquette = gestes entre f et le premier instant (en allant vers b) où le monde sans sa main est déjà celui de b (lue dans les pixels).
t et h sont tirés comme le planificateur les rencontre (fenêtre qui glisse : t = T−1−h la plupart du temps).
"""
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_phase0 import to_tok0, to_torch, to_np, layout, H, KEYS
from av_world0 import gen_world0, VIEW


class DistHead(nn.Module):
    """(scène maintenant, image-but) (B, npf, din) -> gestes qui les séparent (B,) ; par case [but − maintenant, but, maintenant] comme GoalHead v2."""
    def __init__(s, din, npf, norm=True, h=192, nl=3):
        super().__init__()
        s.norm = norm; s.inp = nn.Linear(3 * din, h); s.pos = nn.Parameter(torch.randn(1, npf + 1, h) * 0.02); s.q = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, batch_first=True, dropout=0.0, activation="gelu"), nl); s.out = nn.Linear(h, 1)
    def forward(s, z_now, z_goal):
        n_ = (lambda z: F.layer_norm(z.float(), (z.size(-1),))) if s.norm else (lambda z: z.float()); zn, zg = n_(z_now), n_(z_goal)
        x = torch.cat([s.q.expand(len(zn), -1, -1), s.inp(torch.cat([zg - zn, zg, zn], -1))], 1) + s.pos
        return s.out(s.tr(x)[:, 0]).squeeze(-1)


def sep_labels(xnh, TS, thr):
    """xnh (T, H, H, 3) uint8, images SANS sa main -> (nt, nt) : gestes entre l'instant TS[i] et le premier instant (en allant vers TS[j]) où le monde est celui de TS[j]."""
    x = xnh.reshape(len(xnh), -1).astype(np.int16); same = np.abs(x[:, None] - x[None, TS]).max(-1) <= thr
    L = np.zeros((len(TS), len(TS)), np.float32)
    for i, a_ in enumerate(TS):
        for j, b_ in enumerate(TS):
            sg = 1 if b_ >= a_ else -1; L[i, j] = next(k for k in range(abs(b_ - a_) + 1) if same[a_ + sg * k, j])
    return L


def auc(score, pos):
    r = np.empty(len(score)); r[np.argsort(score)] = np.arange(1, len(score) + 1); n1 = int(pos.sum()); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


@torch.no_grad()
def judge_data(m, teachN, imagine, cfg, st, dev, dR, hmax, plan_c=3, n=3000, per=3, nb=8, thr=0.15, seed=4242, bs=8):
    """jeu libre -> scènes à l'image f = t+h : vrais latents ZR et latents imaginés ZI (n, per, npf, dR) ; buts ZG (n, per, nb, npf, dR) ;
    étiquettes LAB (n, per, nb) ; horizons HH (n, per). teachN / imagine : ceux du planificateur (même étage lu)."""
    T = int(cfg.get("T", 16)); P_ = cfg["P"]; npf = (H // P_) ** 2; nv = T * npf; md, fr = layout(nv, npf); v = VIEW["0e"]
    wkw = {**cfg.get("wkw", {}), "p_out": 0.0, "nohand": True}; wkw.pop("circ", None)
    w = to_np([gen_world0(n, "0e", T, H, seed=seed, **wkw)]); rng = np.random.default_rng(seed); ix = torch.arange(nv, device=dev)
    ZR, ZI, ZG, LAB, HH = [], [], [], [], []
    for i in range(0, n, bs):
        b = to_torch(w, slice(i, i + bs)); B = len(b["X"]); tok, cmd = to_tok0(b, st, v, P_, dev)
        zr_all = teachN(tok[:, :nv], ix.expand(B, -1)).view(B, T, npf, dR)                                       # la vraie séquence, toutes ses images
        L_all = np.stack([sep_labels(x, np.arange(T), round(thr * 255)) for x in w["XNH"][i:i + bs]])           # (B, T, T)
        zr, zi, zg, lab, hs = [], [], [], [], []
        for _ in range(per):
            hh = int(rng.integers(1, hmax + 1)); t0 = T - 1 - hh if rng.random() < 0.5 else int(rng.integers(plan_c, T - hh)); f = t0 + hh
            gb = rng.choice(T, nb, replace=False)                                                                # instants-buts (passés ou futurs)
            ci = torch.from_numpy(np.where(fr <= t0)[0]).to(dev).expand(B, -1); ti = torch.from_numpy(np.where((fr == f) & (md == 0))[0]).to(dev).expand(B, -1)
            c = cmd.clone(); c[:, f + 1:] = 0                                                                    # ses vrais gestes jusqu'à f (comme un plan de h gestes)
            zi.append(imagine(m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci), ti, c, ci).float().view(B, npf, dR).half().cpu())
            Xg = b["X"][:, None].repeat(1, nb, 1, 1, 1, 1); Xg[:, :, t0 + 1:] = b["XNH"][:, gb][:, :, None]     # vrai passé ≤ t0, puis le but « pour toujours »
            bg = {k: b[k].repeat_interleave(nb, 0) for k in KEYS}; bg["X"] = Xg.flatten(0, 1); tkg, _ = to_tok0(bg, st, v, P_, dev)
            zg.append(teachN(tkg[:, :nv], ix.expand(B * nb, -1)).view(B, nb, T, npf, dR)[:, :, f].half().cpu())
            zr.append(zr_all[:, f].half().cpu()); lab.append(L_all[:, f][:, gb]); hs.append(np.full(B, hh))
        ZR.append(torch.stack(zr, 1)); ZI.append(torch.stack(zi, 1)); ZG.append(torch.stack(zg, 1)); LAB.append(np.stack(lab, 1)); HH.append(np.stack(hs, 1))
    return torch.cat(ZR), torch.cat(ZI), torch.cat(ZG), torch.from_numpy(np.concatenate(LAB)), torch.from_numpy(np.concatenate(HH))


def build_judge(m, teachN, imagine, cfg, st, dev, dR, hmax, mode=1, plan_c=3, n=3000, steps=3000, seeds=3, thr=0.15, log=print):
    """entraîne `seeds` têtes sur le jeu libre et rend juge(z_scène, z_but) -> gestes (moyenne des têtes).
    mode 1 : scènes = vrais latents seulement ; mode 2 : moitié vrais latents, moitié latents IMAGINÉS.
    Journal : jeu libre à part, jugé sur les vraies scènes ET sur les scènes imaginées (le juge tient-il en imagination ?)."""
    t_ = time.time(); ZR, ZI, ZG, lab3, HH = judge_data(m, teachN, imagine, cfg, st, dev, dR, hmax, plan_c, n, thr=thr)
    n, per, nb, npf = ZG.shape[:4]; ntr = int(0.8 * n)
    si, ji, ki = (torch.from_numpy(x.reshape(-1)) for x in np.meshgrid(np.arange(n), np.arange(per), np.arange(nb), indexing="ij")); lab = lab3.reshape(-1).float()
    tr0, tr1 = torch.where((si < ntr) & (lab == 0))[0], torch.where((si < ntr) & (lab > 0))[0]; te = torch.where(si >= ntr)[0]
    log(f"JUGE APPRIS ({'vraies scènes' if mode == 1 else 'vraies scènes + scènes IMAGINÉES'}) : {n} séquences de jeu libre × {per} instants × {nb} buts, horizons 1..{hmax} | "
        f"paires où le monde a changé : {float((lab > 0).float().mean()):.0%} | {time.time() - t_:.0f}s")
    y = lab[te].numpy(); mv = y > 0; hte = HH[si[te], ji[te]].numpy(); nets = []
    def score(pr, sel):
        e = np.abs(pr - y); return (f"erreur {e[sel].mean():.2f} geste (monde changé {e[sel & mv].mean():.2f}, corrélation {np.corrcoef(pr[sel & mv], y[sel & mv])[0, 1]:+.2f}), "
                                    f"« déjà au but » AUC {auc(-pr[sel], ~mv[sel]):.2f}")
    for seed in range(seeds):
        torch.manual_seed(seed); net = DistHead(dR, npf).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
        for _ in range(steps):
            sel = torch.cat([tr0[torch.randint(0, len(tr0), (128,))], tr1[torch.randint(0, len(tr1), (128,))]])   # moitié « rien n'a bougé », moitié « le monde a changé »
            zn = ZR[si[sel], ji[sel]]
            if mode == 2: im = torch.rand(len(sel)) < 0.5; zn[im] = ZI[si[sel[im]], ji[sel[im]]]                    # une scène sur deux : imaginée
            l = F.smooth_l1_loss(net(zn.to(dev), ZG[si[sel], ji[sel], ki[sel]].to(dev)), lab[sel].to(dev)); opt.zero_grad(); l.backward(); opt.step(); sched.step()
        net.eval(); nets.append(net)
        with torch.no_grad():
            pr, pi = (torch.cat([net(Z_[si[c], ji[c]].to(dev), ZG[si[c], ji[c], ki[c]].to(dev)).cpu() for c in te.split(1024)]).numpy() for Z_ in (ZR, ZI))
        al = np.ones(len(y), bool)
        log(f"   tête {seed} | jeu libre à part (repère « toujours la médiane » {np.abs(y - np.median(y)).mean():.2f}) | {time.time() - t_:.0f}s\n"
            f"      VRAIES scènes   : {score(pr, al)}\n      scènes IMAGINÉES : {score(pi, al)}\n"
            + "\n".join(f"         imaginé à h={h_} : {score(pi, hte == h_)}  | vrai : erreur {np.abs(pr - y)[hte == h_].mean():.2f}" for h_ in sorted(set(hte.tolist()))))
    del ZR, ZI, ZG
    return lambda zn, zg: torch.stack([net(zn, zg) for net in nets]).mean(0)
