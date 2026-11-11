"""
SENS DU MOUVEMENT — TEST AVANT DE RÉENTRAÎNER (étape 4 du plan du 2026-11-10 ; accord user « Go »).
Le juge appris tient en imagination (av_judge, --judge 2 : AUC 0.80) mais le planificateur classe toujours les gestes au hasard : en imagination,
l'objet poussé ne dépend pas du geste (4.36 px avec SES gestes, 4.35 px avec ceux d'un autre). Avant de donner au bébé un sens « ça bouge ici »,
on cherche OÙ est le verrou. Bébé GELÉ ; une petite tête NEUVE regarde, case par case, (la scène maintenant, la scène à t+h) et dit dans quelles
cases le monde SANS SA MAIN aura changé entre maintenant et t+h (étiquette lue dans les pixels, jeu libre seulement). Lue de plusieurs façons :
  présent seul            (maintenant, maintenant)        : PLANCHER — ce qui se devine sans connaître la suite ;
  vrai futur              (maintenant, vrais latents t+h) : PLAFOND — l'œil voit-il seulement que l'objet a bougé ?
  imaginé, SES gestes     (maintenant, latents imaginés)  : tête apprise SUR l'imaginé (leçon ARC : les têtes meurent en imagination) ;
  imaginé, gestes d'un AUTRE : la même tête, mêmes souvenirs, autres gestes futurs (témoin).
LECTURE : plafond bas -> le verrou est l'ŒIL ; plafond haut mais SES gestes ≈ ceux d'un autre ≈ plancher -> le prédicteur ignore la poussée
(réentraîner avec le nouveau sens) ; SES gestes nettement devant -> l'information est déjà là, la donner au juge.
« maintenant » = l'œil-cible sur le vrai passé ≤ t seulement (comme le planificateur ; aucune fuite du futur).
  python av_plan0.py --ckpt .../phase0_rD17n_0e_190k.pt --task pusht --move_diag 1
"""
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_phase0 import to_tok0, to_torch, to_np, layout, H
from av_world0 import gen_world0, VIEW
from av_judge import auc


class MoveHead(nn.Module):
    """(scène maintenant, scène à t+h) (B, npf, din) -> par case : le monde sans sa main y aura-t-il changé ? (B, npf) logits ; par case [après − maintenant, après, maintenant]."""
    def __init__(s, din, npf, h=192, nl=3):
        super().__init__()
        s.inp = nn.Linear(3 * din, h); s.pos = nn.Parameter(torch.randn(1, npf, h) * 0.02)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, batch_first=True, dropout=0.0, activation="gelu"), nl); s.out = nn.Linear(h, 1)
    def forward(s, z0, zf):
        n_ = lambda z: F.layer_norm(z.float(), (z.size(-1),)); z0, zf = n_(z0), n_(zf)
        return s.out(s.tr(s.inp(torch.cat([zf - z0, zf, z0], -1)) + s.pos)).squeeze(-1)


@torch.no_grad()
def move_data(m, teachN, imagine, cfg, st, dev, dR, hmax, plan_c=3, n=3000, per=3, thr=0.15, seed=5151, bs=8):
    """jeu libre -> Z0 (maintenant, vrai passé ≤ t0), ZR (vrai t0+h), ZI (imaginé, ses gestes), ZO (imaginé, gestes d'un autre) (n·per, npf, dR) ;
    LAB (n·per, npf) cases où le monde sans sa main a changé entre t0 et t0+h ; NPX pixels changés ; STILL monde immobile juste avant t0 ;
    TOUCH contact (examen TSRC) entre t0 et t0+h ; HH horizons."""
    T = int(cfg.get("T", 16)); P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; md, fr = layout(nv, npf); v = VIEW["0e"]
    wkw = {**cfg.get("wkw", {}), "p_out": 0.0, "nohand": True}; wkw.pop("circ", None)
    w0 = gen_world0(n, "0e", T, H, seed=seed, **wkw); TS = w0["TSRC"]; w = to_np([w0]); del w0
    rng = np.random.default_rng(seed); ix = torch.arange(nv, device=dev); th = round(thr * 255)
    gat = lambda tok, i_: torch.gather(tok, 1, i_.unsqueeze(-1).expand(-1, -1, tok.size(-1)))
    Z0, ZR, ZI, ZO, LAB, NPX, STILL, TOUCH, HH = ([] for _ in range(9))
    for i in range(0, n, bs):
        b = to_torch(w, slice(i, i + bs)); B = len(b["X"]); tok, cmd = to_tok0(b, st, v, P_, dev)
        zr_all = teachN(tok[:, :nv], ix.expand(B, -1)).view(B, T, npf, dR); xnh = w["XNH"][i:i + bs].astype(np.int16)
        for _ in range(per):
            hh = int(rng.integers(1, hmax + 1)); t0 = T - 1 - hh if rng.random() < 0.5 else int(rng.integers(plan_c, T - hh)); f = t0 + hh
            ci = torch.from_numpy(np.where(fr <= t0)[0]).to(dev).expand(B, -1); ti = torch.from_numpy(np.where((fr == f) & (md == 0))[0]).to(dev).expand(B, -1)
            vi = torch.from_numpy(np.where((fr <= t0) & (md == 0))[0]).to(dev).expand(B, -1)
            c = cmd.clone(); c[:, f + 1:] = 0; co = c.clone(); co[:, t0 + 1:f + 1] = cmd.roll(1, 0)[:, t0 + 1:f + 1]   # mêmes souvenirs, AUTRES gestes futurs
            ctx = m.context(gat(tok, ci), ci)
            Z0.append(teachN(gat(tok, vi), vi)[:, -npf:].half().cpu()); ZR.append(zr_all[:, f].half().cpu())
            ZI.append(imagine(ctx, ti, c, ci).float().view(B, npf, dR).half().cpu()); ZO.append(imagine(ctx, ti, co, ci).float().view(B, npf, dR).half().cpu())
            ch = np.abs(xnh[:, f] - xnh[:, t0]).max(-1) > th                                                     # (B, H, H) pixels du monde sans main qui ont changé
            LAB.append(ch.reshape(B, nP, P_, nP, P_).any((2, 4)).reshape(B, npf)); NPX.append(ch.reshape(B, -1).sum(1))
            STILL.append(np.abs(xnh[:, t0] - xnh[:, t0 - 1]).reshape(B, -1).max(1) <= th); TOUCH.append((TS[i:i + bs, t0:f + 1] > 0).any(1)); HH.append(np.full(B, hh))
    # (lot, tirage) -> à plat, dans l'ordre des séquences : la séquence du tirage j est j // per
    tcat = lambda L: torch.stack([torch.cat(L[k::per]) for k in range(per)], 1).flatten(0, 1); ncat = lambda L: np.stack([np.concatenate(L[k::per]) for k in range(per)], 1).reshape(n * per, *L[0].shape[1:])
    return tcat(Z0), tcat(ZR), tcat(ZI), tcat(ZO), ncat(LAB), ncat(NPX), ncat(STILL), ncat(TOUCH), ncat(HH)


def move_diag(m, teachN, imagine, cfg, st, dev, dR, hmax, plan_c=3, n=3000, steps=3000, seeds=2, thr=0.15, log=print):
    t_ = time.time(); per = 3; Z0, ZR, ZI, ZO, LAB, NPX, STILL, TOUCH, HH = move_data(m, teachN, imagine, cfg, st, dev, dR, hmax, plan_c, n, per, thr)
    N, npf = LAB.shape; si = np.arange(N) // per; ntr = int(0.8 * n); any_ = LAB.any(1); lab = torch.from_numpy(LAB).float()
    tr0, tr1 = (torch.from_numpy(np.where((si < ntr) & x)[0]) for x in (~any_, any_)); te = np.where(si >= ntr)[0]; tet = torch.from_numpy(te)
    y = any_[te]; mine = y & STILL[te] & TOUCH[te]; calm = ~y & STILL[te]; hte = HH[te]; med = np.median(NPX[te][mine]) if mine.any() else 0
    pw = torch.tensor(min(10.0, 1 / max(0.5 * float(LAB[(si < ntr) & any_].mean()), 1e-3) - 1), device=dev)
    log(f"SENS DU MOUVEMENT (test, bébé gelé) : {n} séquences de jeu libre × {per} instants, horizons 1..{hmax} | le monde sans sa main a changé : {any_.mean():.0%} des tirages "
        f"(cases changées alors : {LAB[any_].sum(1).mean():.1f}/{npf}) | monde immobile juste avant : {STILL.mean():.0%} | contact pendant : {TOUCH.mean():.0%} | "
        f"à part : {int(mine.sum())} « SA poussée » (immobile avant, contact, changé ; médiane {med:.0f} px changés) contre {int(calm.sum())} « rien ne bouge » | {time.time() - t_:.0f}s")

    def fit(Zf, seed):
        torch.manual_seed(seed); net = MoveHead(dR, npf).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
        for _ in range(steps):
            sel = torch.cat([tr0[torch.randint(0, len(tr0), (128,))], tr1[torch.randint(0, len(tr1), (128,))]])   # moitié « rien n'a changé », moitié « le monde a changé »
            l = F.binary_cross_entropy_with_logits(net(Z0[sel].to(dev), Zf[sel].to(dev)), lab[sel].to(dev), pos_weight=pw); opt.zero_grad(); l.backward(); opt.step(); sched.step()
        return net.eval()

    @torch.no_grad()
    def read(net, Zf):                          # -> (8,) : a bougé ? (tout) | SA poussée vs rien | petite | grande | h 1–2 | h 3–4 | h 5–6 | où (par case, dans SES poussées)
        lg = torch.cat([net(Z0[c].to(dev), Zf[c].to(dev)).cpu() for c in tet.split(1024)]).numpy(); s = lg.max(1)
        duel = lambda pos: auc(s[pos | calm], pos[pos | calm]) if pos.any() and calm.any() else np.nan
        return np.array([auc(s, y), duel(mine), duel(mine & (NPX[te] <= med)), duel(mine & (NPX[te] > med))] + [duel(mine & (hte >= lo) & (hte <= lo + 1)) for lo in (1, 3, 5)]
                        + [auc(lg[mine].reshape(-1), LAB[te][mine].reshape(-1)) if mine.any() else np.nan])

    R = {}
    for nm, Ztr, evs in (("présent seul (PLANCHER)", Z0, (("présent seul (PLANCHER)", Z0),)),
                         ("vrai futur (PLAFOND)", ZR, (("vrai futur (PLAFOND)", ZR), ("tête du vrai futur, lue sur l'imaginé", ZI))),
                         ("imaginé, SES gestes", ZI, (("imaginé, SES gestes", ZI), ("imaginé, gestes d'un AUTRE", ZO)))):
        for seed in range(seeds):
            net = fit(Ztr, seed)
            for en, Zf in evs: R.setdefault(en, []).append(read(net, Zf))
            del net
        log(f"   tête « {nm} » : {seeds} graines × {steps} pas | {time.time() - t_:.0f}s")
    log(f"   AUC (0.50 = hasard, 1.00 = parfait), moyenne des {seeds} têtes — « SA poussée vs rien » : la tête range-t-elle les tirages où il a poussé un objet immobile devant ceux où rien ne bouge ?\n"
        f"   {'lecture':>38s} | {'a bougé ? (tout)':>16s} | {'SA poussée vs rien':>18s} | {'petite':>6s} | {'grande':>6s} | {'h 1–2':>5s} | {'h 3–4':>5s} | {'h 5–6':>5s} | {'où (par case)':>13s}")
    for en, r in R.items():
        x = np.nanmean(np.stack(r), 0); log(f"   {en:>38s} | {x[0]:16.2f} | {x[1]:18.2f} | {x[2]:6.2f} | {x[3]:6.2f} | {x[4]:5.2f} | {x[5]:5.2f} | {x[6]:5.2f} | {x[7]:13.2f}")
