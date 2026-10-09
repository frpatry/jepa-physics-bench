"""
PHASE 0 « MES MAINS » — ENTRAÎNEUR (monde : av_world0.py ; fondements : recherche_bebe_mains.md).

Architecture (règle validée avec l'user : ce qui est RESSENTI -> encodeur ; ce qui est VOULU -> prédicteur seulement) :
  - ENCODEUR des 4 sens (vue 64 patches/frame, ouïe 1 token/frame stéréo explicite, toucher 1 token/frame = peau 4 côtés,
    sens du bras 1 token/frame), chacun traité à part (sep) — la FUSION se fait dans le prédicteur ;
  - PRÉDICTEUR conditionné par la COMMANDE (copie d'efférence) DÈS LE DÉPART (style PLDM / SPR, pas après coup) :
    contexte (sens visibles) + commandes de TOUTES les frames (le futur = gestes prévus) -> latents des sens cachés ;
  - masques : tube / bloc de vision, FUTUR de tous les sens (imaginer la vue, le son, le toucher, le bras), un SENS ENTIER
    deviné par les autres (cross-modal), et parfois un sens retiré du contexte ;
  - CONTINGENCE : « deviner son geste en VOYANT l'avant et l'après » (modèle inverse sur la VISION seule — le sens du bras
    contient le déplacement, il ne doit pas servir de raccourci) ; combiné au JEPA (seul, il rend aveugle au passif) ;
  - cible EMA + SIGReg léger + lr cosinus (recette stable du run SL).
Curriculum : 0a bras -> 0b mains -> 0c contingence -> 0d contact -> 0e distance (budgets en pas), avec RAPPEL (30 % du lot
tiré des étapes déjà vécues) ; la VUE du bébé (flou uniforme, gris, contraste) mûrit pendant chaque étape vers la suivante.

EXAMENS (zéro étiquette sauf l'instrument 6) :
  1 ÉCOUTE SES GESTES : imaginer le futur avec SES commandes vs celles d'une autre séquence (+ % d'erreur si on ment) ;
  2 RUBAN COUPÉ : le mobile cesse de répondre -> surprise ? ; RETARD de 3 frames (Bahrick & Watson) -> surprise ?
  3 VU ICI, SENTI LÀ : sens du bras d'une autre séquence -> surprise ?
  4 CONTACT VU MAIS PAS SENTI : toucher effacé pendant un contact -> surprise ?
  5 SAIT CE QU'IL A FAIT : R² du geste deviné par la vision (modèle inverse) ;
  6 INSTRUMENT : sonde ridge exacte, position de la main et d'un objet (% de la largeur ; pixels bruts en repère).

  python av_phase0.py --total 60000 --ckpt /content/drive/MyDrive/jepa_runs/phase0.pt     # Colab A100
"""
import argparse, copy, math, os, time
import multiprocessing as mp
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import NB, tube_masks, _layer_masked
from av_dev import blur
from av_dev_long import stereo
from av_world0 import gen_world0, VIEW
from vjepa import _idx, _gather, sigreg

T, H = 16, 32
ORDER = ["0a", "0b", "0c", "0d", "0e"]
SENS = ("vue", "ouïe", "toucher", "bras")
KEYS = ("X", "A", "TOUCH", "PROP", "CMD")

# ---------------------------------------------------------------- modèle
class Enc4(nn.Module):
    """encodeur des 4 SENS (la commande n'y entre JAMAIS). Indices : vue [0,nv), ouïe [nv,nv+T), toucher, bras.
    sep : attention à l'intérieur d'un sens seulement (la fusion = le prédicteur)."""
    def __init__(s, din, d, nv, T, nl, nh, sep=True):
        super().__init__()
        s.din, s.nv, s.T, s.sep = list(din), nv, T, sep
        s.inp = nn.ModuleList([nn.Linear(k, d) for k in din]); s.mod = nn.Embedding(4, d); s.pos = nn.Embedding(nv + 3 * T, d)
        layer = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d)
    def modality(s, idx): return (idx >= s.nv).long() + (idx >= s.nv + s.T).long() + (idx >= s.nv + 2 * s.T).long()
    def forward(s, tok, idx):
        md = s.modality(idx); e = 0
        for k, lin in enumerate(s.inp): e = e + (md == k).unsqueeze(-1) * lin(tok[..., :s.din[k]])
        x = e + s.mod(md) + s.pos(idx)
        if not s.sep: return s.ln(s.tr(x))
        cnt = [(md == k).sum(1) for k in range(4)]
        if all(bool((c == c[0]).all()) for c in cnt):          # indices triés -> sens contigus : une passe par sens (rapide)
            out, j = [], 0
            for c in cnt:
                c = int(c[0])
                if c: out.append(s.tr(x[:, j:j + c])); j += c
            return s.ln(torch.cat(out, 1))
        allow = (md.unsqueeze(2) == md.unsqueeze(1)).unsqueeze(1)
        for L in s.tr.layers: x = _layer_masked(L, x, allow)
        return s.ln(x)

class PredAC(nn.Module):
    """prédicteur conditionné par la VOLONTÉ : [contexte | commandes des T frames | requêtes] -> latents des requêtes.
    resid (run 2) : prédiction = DERNIER latent VU de la même case (même sens, frame la plus récente ≤ cible, normalisé)
    + changement (tête zéro-init) — run 1 : l'imagination PERDAIT l'objet, même immobile (diag av_plan0 --obj_diag)."""
    def __init__(s, d, ntok, T, nl, nh, resid=False, nv=0, npf=1):
        super().__init__()
        s.resid = resid
        if resid:                                                      # case (0..npf-1 vue, npf ouïe, npf+1 toucher, npf+2 bras) et frame de chaque token
            idx = torch.arange(ntok); cell = torch.where(idx < nv, idx % npf, npf + (idx - nv) // T); fr = torch.where(idx < nv, idx // npf, (idx - nv) % T)
            s.register_buffer("cell_of", cell, persistent=False); s.register_buffer("frame_of", fr, persistent=False); s.ncell = npf + 3
        s.mask_token = nn.Parameter(torch.zeros(d)); s.pos = nn.Embedding(ntok, d); s.tpos = nn.Embedding(T, d)
        s.ec = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, d)
    def base(s, ctx, ctx_idx, tgt_idx):
        """latent du contexte à la même case, frame visible la plus récente ≤ frame cible (0 si aucune)."""
        B, kc, d = ctx.shape; Tn = s.tpos.num_embeddings; b_ = torch.arange(B, device=ctx.device)[:, None]
        pos = torch.full((B, s.ncell, Tn), -1, dtype=torch.long, device=ctx.device)
        pos[b_, s.cell_of[ctx_idx], s.frame_of[ctx_idx]] = torch.arange(kc, device=ctx.device).expand(B, -1)
        fr_ = torch.arange(Tn, device=ctx.device).expand_as(pos); last = torch.where(pos >= 0, fr_, -1).cummax(-1).values
        q = torch.gather(pos, 2, last.clamp_min(0)); q = torch.where(last >= 0, q, -1)[b_, s.cell_of[tgt_idx], s.frame_of[tgt_idx]]
        z = F.layer_norm(torch.gather(ctx.detach(), 1, q.clamp_min(0).unsqueeze(-1).expand(-1, -1, d)).float(), (d,))   # base SANS gradient (run 2a : effondrement)
        return z * (q >= 0).unsqueeze(-1)
    def forward(s, ctx, tgt_idx, cmd, ctx_idx=None):
        c = s.ec(cmd) + s.tpos.weight[:cmd.size(1)]; tg = s.mask_token + s.pos(tgt_idx)
        x = s.tr(torch.cat([ctx, c, tg], 1)); out = s.head(s.ln(x[:, -tgt_idx.size(1):]))
        if s.resid:
            assert ctx_idx is not None, "prédicteur résiduel : passer ctx_idx"
            out = s.base(ctx, ctx_idx, tgt_idx) + out
        return out

class InvHead(nn.Module):
    """CONTINGENCE : deviner son geste en VOYANT l'avant et l'après (patches de 2 frames consécutives -> commande)."""
    def __init__(s, d, npf, h=128):
        super().__init__()
        s.inp = nn.Linear(d, h); s.pos = nn.Parameter(torch.zeros(1, 2 * npf, h)); s.cls = nn.Parameter(torch.zeros(1, 1, h))
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, batch_first=True, dropout=0.0, activation="gelu"), 1)
        s.out = nn.Linear(h, 2)
    def forward(s, z0, z1):
        x = s.inp(torch.cat([z0, z1], 1)) + s.pos
        return s.out(s.tr(torch.cat([s.cls.expand(len(x), -1, -1), x], 1))[:, 0])

class DiffHead(nn.Module):
    """CONTINGENCE, version DIFFÉRENCE : par patch [z_t − z_{t−1}, z_t] -> MLP -> moyenne -> commande.
    Diag av_inv_diag (instantané 0a 6k, latents gelés) : R² 0.73–0.75 vs 0.66 pour InvHead neuve (0.07 entraînée en conjoint)."""
    def __init__(s, d, npf, h=256):
        super().__init__(); s.pos = nn.Parameter(torch.zeros(1, npf, 2 * d)); s.mlp = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.out = nn.Linear(h, 2)
    def forward(s, z0, z1): return s.out(s.mlp(torch.cat([z1 - z0, z1], -1) + s.pos).mean(1))

class Baby0(nn.Module):
    def __init__(s, din, nv, npf, d, nl, nh, pl, sep, inv_head="attn", resid=False, std_tgt=False):
        super().__init__()
        s.enc = Enc4(din, d, nv, T, nl, nh, sep); s.pred = PredAC(d, nv + 3 * T, T, pl, nh, resid, nv, npf)
        if resid: nn.init.zeros_(s.pred.head.weight); nn.init.zeros_(s.pred.head.bias)   # départ = « rien ne change »
        s.inv = DiffHead(d, npf) if inv_head == "diff" else InvHead(d, npf)
        if std_tgt: s.register_buffer("tstd", torch.ones(4, d))       # écart-type courant des latents CIBLES par sens et par dimension

def baby_from_cfg(cfg, dev):
    """reconstruit le bébé d'un instantané (toutes options) — pour les scripts d'examen / de planification.
    Fixe aussi la longueur des séquences du module (T) à celle de l'instantané."""
    global T; T = int(cfg.get("T", 16)); P = cfg["P"]; npf = (H // P) ** 2
    return Baby0(cfg["din"], T * npf, npf, cfg["d"], cfg["nl"], cfg["nh"], cfg["pred_layers"], bool(cfg["sep"]), cfg.get("inv_head", "attn"),
                 bool(cfg.get("resid", 0)), bool(cfg.get("std_tgt", 0))).to(dev)

# ---------------------------------------------------------------- données -> tokens
def layout(nv, npf):
    md = np.zeros(nv + 3 * T, int); md[nv:] = 1 + np.arange(3 * T) // T
    fr = np.concatenate([np.arange(nv) // npf, np.tile(np.arange(T), 3)]); return md, fr

def view_at(stage, f):
    """vue du bébé à l'intérieur d'une étape : mûrit de VIEW[étape] vers VIEW[étape suivante] (f = avancement 0..1)."""
    nxt = ORDER[min(ORDER.index(stage) + 1, len(ORDER) - 1)]
    return {k: (1 - f) * VIEW[stage][k] + f * VIEW[nxt][k] for k in VIEW[stage]}

def to_tok0(b, st, v, P, dev):
    """lot -> tokens (B, nv + 3T, W) [vue | ouïe | toucher | bras] et commandes (B, T, 2) — normalisés."""
    X = b["X"].to(dev, non_blocking=True).float() / 255.0; B = X.size(0); nP = H // P
    X = blur(X, v["sigma"]); X = (1 - v["gray"]) * X + v["gray"] * X.mean(-1, keepdim=True)
    mu = X.mean((2, 3, 4), keepdim=True); X = (mu + (X - mu) * v["contrast"]).clamp(0, 1)
    vis = X.reshape(B, T, nP, P, nP, P, 3).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T * nP * nP, P * P * 3); nv = vis.size(1)
    a = (stereo(b["A"].to(dev).float()).reshape(B, T, -1) - st["amu"]) / st["asd"]
    tch = b["TOUCH"].to(dev).float() / st["tsd"]; pr = b["PROP"].to(dev).float()
    pr = torch.cat([(pr[..., :2] - 0.5) * 2, pr[..., 2:] / 0.05], -1)
    tok = torch.zeros(B, nv + 3 * T, a.size(-1), device=dev)
    tok[:, :nv, :vis.size(-1)] = vis; tok[:, nv:nv + T] = a; tok[:, nv + T:nv + 2 * T, :8] = tch; tok[:, nv + 2 * T:, :4] = pr
    return tok, b["CMD"].to(dev).float() / 0.05

def to_np(ws):
    cat = lambda k: np.concatenate([w[k] for w in ws])
    return dict(X=(cat("X") * 255).round().astype(np.uint8), A=cat("A").astype(np.float16), TOUCH=cat("TOUCH"), PROP=cat("PROP"), CMD=cat("CMD"))

def to_torch(w, sl=slice(None)): return {k: torch.from_numpy(np.ascontiguousarray(w[k][sl])) for k in KEYS}

def _batch0(job):
    seed, mix, wkw = job
    return to_np([gen_world0(n, s_, T, H, seed=seed * 8 + j, **wkw) for j, (s_, n) in enumerate(mix) if n > 0])

class Stream:
    """lots NEUFS générés en parallèle (avance bornée) ; mélange = étape courante + RAPPEL des étapes déjà vécues."""
    def __init__(s, start, bs, workers, replay, prefetch=16, wkw=None):
        s.pool = mp.get_context("fork").Pool(workers); s.i, s.bs, s.replay, s.prefetch, s.stage, s.q = start, bs, replay, prefetch, "0a", deque()
        s.wkw = wkw or {}
    def mix(s):
        i = ORDER.index(s.stage); nr = int(round(s.bs * s.replay)) if i > 0 else 0; m = {s.stage: s.bs - nr}
        for j in range(nr): k = ORDER[(s.i + j) % i]; m[k] = m.get(k, 0) + 1
        return list(m.items())
    def next(s):
        while len(s.q) < s.prefetch: s.q.append(s.pool.apply_async(_batch0, ((10_000_000 + s.i, s.mix(), s.wkw),))); s.i += 1
        return {k: torch.from_numpy(v) for k, v in s.q.popleft().get().items()}

def masks0(B, nP, nv, rng, n_masks, touch=0):
    """[(contexte (B,N), cible (B,N))] — même nombre de tokens par ligne. touch > 0 : autant de tirages en plus
    « devine ce que tu SENS » (toucher entier deviné par la vue, l'ouïe, le bras et les gestes)."""
    md, fr = layout(nv, nP * nP); N = len(md); pairs = []
    for _ in range(n_masks):
        st = rng.choice(["tube", "vblock", "future", "future", "cross"] + ["touch"] * touch); tg = np.zeros((B, N), bool)
        if st == "touch": tg[:] = md == 2; pairs.append((~tg, tg)); continue
        if st == "future":                                           # IMAGINER : tous les sens après t0, avec les gestes prévus
            t0 = rng.integers(3, T - 2); pairs.append((np.broadcast_to(fr <= t0, (B, N)).copy(), np.broadcast_to(fr > t0, (B, N)).copy())); continue
        if st == "tube": tg[:, :nv] = tube_masks(B, T, nP, 0.5, 1, rng)[0].numpy()
        elif st == "vblock":
            for b in range(B): t0 = rng.integers(0, T - 2); tg[b] = (md == 0) & (fr >= t0) & (fr < t0 + 3)
        else: tg[:] = md == rng.integers(1, 4)                       # un SENS ENTIER deviné par les autres
        cm = ~tg
        if st != "cross" and rng.random() < 0.3: cm &= md != rng.integers(1, 4)   # un sens retiré du contexte
        pairs.append((cm, tg))
    return pairs

# ---------------------------------------------------------------- examens
@torch.no_grad()
def errs(m, tgt, b, st, v, P, ctx_m, tgt_m, dev, d, b_tgt=None, lie=False, bs=50):
    """erreur L1 (n, kt) du prédicteur (contexte ctx_m + commandes) face aux latents cibles (EMA, séquence complète) ;
    b_tgt : séquence dont on encode la cible (violation d'attente) ; lie : commandes d'une AUTRE séquence."""
    ci0, ti0 = torch.from_numpy(np.where(ctx_m)[0]).to(dev), torch.from_numpy(np.where(tgt_m)[0]).to(dev); out = []
    for i in range(0, len(b["X"]), bs):
        sl = slice(i, i + bs); tok, cmd = to_tok0({k: x[sl] for k, x in b.items()}, st, v, P, dev); B = len(tok)
        ttok = tok if b_tgt is None else to_tok0({k: x[sl] for k, x in b_tgt.items()}, st, v, P, dev)[0]
        if lie: cmd = cmd.roll(1, 0)
        z = F.layer_norm(tgt(ttok, torch.arange(tok.size(1), device=dev).expand(B, -1)).float(), (d,))
        ci, ti = ci0.expand(B, -1), ti0.expand(B, -1)
        out.append((m.pred(m.enc(_gather(tok, ci), ci), ti, cmd, ci).float() - _gather(z, ti)).abs().mean(-1).cpu())
    return torch.cat(out)

@torch.no_grad()
def vision_feats(enc, b, st, v, P, dev, nv, bs=50):
    out = []
    for i in range(0, len(b["X"]), bs):
        tok, _ = to_tok0({k: x[i:i + bs] for k, x in b.items()}, st, v, P, dev); ix = torch.arange(nv, device=dev).expand(len(tok), -1)
        out.append(enc(tok[:, :nv], ix).half().cpu())
    return torch.cat(out)

def ridge_err(Z, Y, dev):
    """SONDE LINÉAIRE RIDGE exacte (forme duale) 70/15/15, λ choisi sur la validation -> erreur moyenne en % de la largeur."""
    N = len(Z); k, v = int(0.7 * N), int(0.85 * N); X = Z.reshape(N, -1).to(dev).float(); X = X - X[:k].mean(0)
    t = Y.to(dev).float(); tm = t[:k].mean(0); K = X[:k] @ X[:k].T; s = K.diagonal().mean(); best = None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        al = torch.linalg.solve(K + lam * s * torch.eye(k, device=dev), t[:k] - tm); Wr = X[:k].T @ al
        ev = (X[k:v] @ Wr + tm - t[k:v]).norm(dim=-1).mean().item()
        if best is None or ev < best[0]: best = (ev, Wr)
    return float((X[v:] @ best[1] + tm - t[v:]).norm(dim=-1).mean() * 100)

def probe_frames(pr, nP, P, npf, rng):
    """(séquence, frame) où la main est VISIBLE (8 par séquence) et où il y a UN seul objet (8 frames, sonde objet)."""
    n = len(pr["X"]); hs, os_ = [], []
    for i in range(n):
        ok = np.where(pr["INVIEW"][i])[0]
        for t in rng.choice(ok, min(8, len(ok)), replace=False): hs.append((i, t))
        if pr["NOBJ"][i] == 1:
            for t in rng.choice(T, 8, replace=False): os_.append((i, t))
    return np.array(hs), np.array(os_)

def exam(m, tgt, probes, st, v, a, dev, nv, npf, tag, ref=None):
    P, d = a.P, a.d; md, fr = layout(nv, npf); res = {}; ctrl = probes["ctrl"]; tb = to_torch(ctrl)
    # 1 ÉCOUTE SES GESTES : futur imaginé avec SES gestes vs ceux d'une autre séquence
    hT = T // 2; cm, tm = fr < hT, fr >= hT; e0 = errs(m, tgt, tb, st, v, P, cm, tm, dev, d); e1 = errs(m, tgt, tb, st, v, P, cm, tm, dev, d, lie=True)   # 1re moitié vue -> 2e imaginée
    tmd, tfr = md[tm], fr[tm]; nP = H // P; msg = []
    for k, nm in enumerate(SENS):
        c = tmd == k; res[f"geste_{nm}"] = float(e1[:, c].mean() / e0[:, c].mean() - 1)
        msg.append(f"{nm} {res[f'geste_{nm}']:+.0%}")
    hc = np.clip((ctrl["HAND"] * nP).astype(int), 0, nP - 1); cell = np.where(tmd == 0)[0]   # vue PRÈS DE LA MAIN (3×3 cases)
    gy, gx = (np.arange(nv) % npf)[cell] // nP, (np.arange(nv) % npf)[cell] % nP; f_ = tfr[cell]
    near = (np.abs(gx[None] - hc[:, f_, 0]) <= 1) & (np.abs(gy[None] - hc[:, f_, 1]) <= 1) & ctrl["INVIEW"][:, f_]
    nr = torch.from_numpy(near); res["geste_main"] = float((e1[:, cell] * nr).sum() / (e0[:, cell] * nr).sum() - 1)
    msg.insert(0, f"vue PRÈS DE LA MAIN {res['geste_main']:+.0%}")
    # 2 RUBAN COUPÉ / RETARD (mobile) — mêmes graines -> mêmes séquences jusqu'à la coupure
    bn, bc, bd = to_torch(probes["mob_n"]), to_torch(probes["mob_c"]), to_torch(probes["mob_d"]); vm = tm & (md == 0)
    en = errs(m, tgt, bn, st, v, P, cm, vm, dev, d).mean(1); ec = errs(m, tgt, bn, st, v, P, cm, vm, dev, d, b_tgt=bc).mean(1)
    ed = errs(m, tgt, bd, st, v, P, cm, vm, dev, d).mean(1)
    res["ruban"], res["retard"] = float((ec > en).float().mean()), float((ed > en).float().mean())
    # 3 VU ICI, SENTI LÀ (bras d'une autre séquence) ; 4 CONTACT VU MAIS PAS SENTI (toucher effacé)
    bsw = dict(tb); bsw["PROP"] = tb["PROP"].roll(1, 0); pm = md == 3
    res["bras"] = float((errs(m, tgt, tb, st, v, P, ~pm, pm, dev, d, b_tgt=bsw).mean(1) > errs(m, tgt, tb, st, v, P, ~pm, pm, dev, d).mean(1)).float().mean())
    ct = np.where((ctrl["TSRC"] > 0).any(1))[0]; bt = to_torch(ctrl, ct); bz = dict(bt); bz["TOUCH"] = torch.zeros_like(bt["TOUCH"]); km = md == 2
    res["toucher"] = float((errs(m, tgt, bt, st, v, P, ~km, km, dev, d, b_tgt=bz).mean(1) > errs(m, tgt, bt, st, v, P, ~km, km, dev, d).mean(1)).float().mean())
    # 5 SAIT CE QU'IL A FAIT (modèle inverse, vision seule) et 6 INSTRUMENT (sonde ridge, encodeur cible)
    Zo = vision_feats(m.enc, tb, st, v, P, dev, nv).view(len(tb["X"]), T, npf, -1); hs = probes["hs"]; ok = hs[:, 1] >= 1
    with torch.no_grad():
        p = m.inv(Zo[hs[ok, 0], hs[ok, 1] - 1].to(dev).float(), Zo[hs[ok, 0], hs[ok, 1]].to(dev).float()).cpu()
    y = tb["CMD"][hs[ok, 0], hs[ok, 1]] / 0.05; res["inverse_r2"] = float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    Z = vision_feats(tgt, tb, st, v, P, dev, nv).view(len(tb["X"]), T, npf, -1)
    res["ecart"] = float(Z[:, :, :4].float().std(0).mean())
    res["main_px"] = ridge_err(Z[hs[:, 0], hs[:, 1]], torch.from_numpy(ctrl["HAND"][hs[:, 0], hs[:, 1]]), dev)
    os_ = probes["os"]; res["objet_px"] = ridge_err(Z[os_[:, 0], os_[:, 1]], torch.from_numpy(ctrl["POS"][os_[:, 0], os_[:, 1], 0]), dev)
    r0 = ref or res; sp = lambda k: f"{res[k]:.0%} (init {r0[k]:.0%})"   # les paires comparées n'ont pas la même difficulté : juger le PROGRÈS
    print(f"  EXAMEN {tag} | 1 ÉCOUTE SES GESTES (erreur si on lui ment) : {', '.join(msg)}\n"
          f"      SURPRISE (50 % = ne remarque rien) : 2 ruban coupé {sp('ruban')}, retard {sp('retard')} | 3 vu ici/senti là {sp('bras')} "
          f"| 4 contact non senti {sp('toucher')}\n"
          f"      5 devine son geste (vision) R² {res['inverse_r2']:+.2f} | 6 sonde : main {res['main_px']:.1f} %, objet {res['objet_px']:.1f} % "
          f"(pixels {probes['pix'][0]:.1f} / {probes['pix'][1]:.1f}) | écart-type latents {res['ecart']:.3f}", flush=True)
    return res

def build_probes(a, nv, npf, dev):
    n = a.n_probe; rng = np.random.default_rng(7); ms = dict(mob_size=a.mob_size)
    wd, we = gen_world0(n, "0d", T, H, seed=2001), gen_world0(n, "0e", T, H, seed=2002)
    ctrl = to_np([wd, we]); ctrl.update({k: np.concatenate([wd[k], we[k]]) for k in ("HAND", "INVIEW", "POS", "NOBJ", "TSRC")})
    pr = dict(ctrl=ctrl, mob_n=to_np([gen_world0(n, "0c", T, H, seed=2003, force_mtype=0, **ms)]),
              mob_c=to_np([gen_world0(n, "0c", T, H, seed=2003, force_mtype=2, force_tcut=T // 2, **ms)]),
              mob_d=to_np([gen_world0(n, "0c", T, H, seed=2003, force_mtype=0, mobile_delay=3, **ms)]))
    pr["hs"], pr["os"] = probe_frames(ctrl, H // a.P, a.P, npf, rng)
    Xp = torch.from_numpy(ctrl["X"]).float() / 255; nP = H // a.P
    Xp = Xp.reshape(len(Xp), T, nP, a.P, nP, a.P, 3).permute(0, 1, 2, 4, 3, 5, 6).reshape(len(Xp), T, npf, -1)
    pr["pix"] = (ridge_err(Xp[pr["hs"][:, 0], pr["hs"][:, 1]], torch.from_numpy(ctrl["HAND"][pr["hs"][:, 0], pr["hs"][:, 1]]), dev),
                 ridge_err(Xp[pr["os"][:, 0], pr["os"][:, 1]], torch.from_numpy(ctrl["POS"][pr["os"][:, 0], pr["os"][:, 1], 0]), dev))
    return pr

# ---------------------------------------------------------------- entraînement
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=60000); p.add_argument("--budgets", type=str, default="6000,8000,8000,12000",
                   help="pas des étapes 0a,0b,0c,0d ; 0e jusqu'à --total")
    p.add_argument("--bs", type=int, default=64); p.add_argument("--P", type=int, default=4); p.add_argument("--d", type=int, default=192)
    p.add_argument("--nl", type=int, default=6); p.add_argument("--nh", type=int, default=6); p.add_argument("--pred_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--ema", type=float, default=0.996); p.add_argument("--n_masks", type=int, default=3)
    p.add_argument("--sig_w", type=float, default=0.005); p.add_argument("--inv_w", type=float, default=0.1, help="poids de la CONTINGENCE (deviner son geste)")
    p.add_argument("--inv_k", type=int, default=4, help="paires de frames par séquence pour la contingence")
    p.add_argument("--inv_head", type=str, default="attn", choices=["attn", "diff"], help="diff = tête DIFFÉRENCE (run 1 : la tête attn conjointe restait à R² ≈ 0)")
    p.add_argument("--inv_clip", type=int, default=0, help="1 = la tête de contingence hors de l'écrêtage global des gradients (dominé par le JEPA)")
    p.add_argument("--resid", type=int, default=0, help="1 = prédicteur RÉSIDUEL (dernier latent vu de la même case + changement)")
    p.add_argument("--std_tgt", type=int, default=0, help="1 = erreur sur cibles STANDARDISÉES par sens et par dimension (les détails discrets comptent autant que la main)")
    p.add_argument("--touch_masks", type=int, default=0, help="tirages supplémentaires « devine ce que tu sens » dans la loterie des masques")
    p.add_argument("--contact_w", type=float, default=0.0, help="poids en plus sur les tokens des frames AVEC CONTACT (toucher non nul)")
    p.add_argument("--p_obj", type=float, default=0.45, help="part des gestes de babillage dirigés VERS un objet (contacts)")
    p.add_argument("--mob_size", type=float, default=0.10, help="taille du mobile (run 1 : 0.10 ≈ 6 px, trop petit)")
    p.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"], help="mse (I-JEPA) : avec le résiduel, L1 donne le même poids à chaque case -> les cases immobiles votent « pas de changement » (run 2A sourd à la commande)")
    p.add_argument("--act_w", type=float, default=0.0, help="poids du CONTRASTE D'ACTION : l'avenir imaginé avec SES gestes doit être plus proche du vrai qu'avec ceux d'un autre")
    p.add_argument("--act_margin", type=float, default=0.2, help="écart exigé (fraction de l'erreur avec ses vrais gestes)")
    p.add_argument("--calm", type=int, default=0, help="1 = COMPLEXITÉ PROGRESSIVE (calme 1 -> 0) ; 2 = monde LISIBLE à progression LENTE (1 -> 0.7 en 0d, 0.7 -> 0.3 en 0e)")
    p.add_argument("--outside", type=int, default=0, help="1 = la TABLE DÉBORDE du champ dans une part croissante des séquences (0d 10 % -> 30 %, 0e 30 % -> 70 %) : objets qui sortent, entendus hors champ")
    p.add_argument("--T", type=int, default=16, help="images par séquence (16 ≈ 1 s ; 32 = 2 s : il faut repartir de zéro)")
    p.add_argument("--parent", type=float, default=0.0, help="part des séquences (avec objets) où la MAIN D'UN PARENT vient pousser un objet (démonstration du contact)")
    p.add_argument("--stop_at", type=int, default=0, help="arrêter (avec sauvegarde) à ce pas — essais courts qu'on peut ensuite PROLONGER")
    p.add_argument("--replay", type=float, default=0.3, help="part du lot tirée des étapes déjà vécues")
    p.add_argument("--sep", type=int, default=1); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--exam_every", type=int, default=5000); p.add_argument("--ckpt_every", type=int, default=2500)
    p.add_argument("--n_probe", type=int, default=300); p.add_argument("--ckpt", type=str, default="/content/phase0.pt"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    global T; T = a.T                                                          # longueur des séquences (tout le module la lit ici)
    torch.backends.cuda.matmul.allow_tf32 = True; rng = np.random.default_rng(a.seed)
    nP = H // a.P; npf = nP * nP; nv = T * npf; din = (a.P * a.P * 3, 4 * NB, 8, 4)
    bud = [int(x) for x in a.budgets.split(",")]; starts = np.cumsum([0] + bud)          # début de 0a, 0b, 0c, 0d, 0e
    def stage_of(it):
        j = min(int(np.searchsorted(starts, it - 1, side="right")) - 1, 4)          # pas 1..budget -> 0a, etc.
        f = (it - 1 - starts[j]) / bud[j] if j < 4 else 0.0; return ORDER[j], min(f, 1.0)
    wkw = dict(p_obj=a.p_obj, mob_size=a.mob_size, p_parent=a.parent)
    ws = to_np([gen_world0(300, s_, T, H, seed=99 + j, **wkw) for j, s_ in enumerate(ORDER)])          # normalisation des sens
    A0 = stereo(torch.from_numpy(ws["A"]).float()).reshape(-1, T, 4 * NB)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev), tsd=torch.from_numpy(ws["TOUCH"]).std((0, 1)).clamp_min(0.01).to(dev))
    print(f"GPU {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'} | {T} frames × ({npf} patches + ouïe + toucher + bras) = {nv + 3 * T} tokens "
          f"| commande -> prédicteur seulement | étapes {dict(zip(ORDER, list(starts)))}", flush=True)
    probes = build_probes(a, nv, npf, dev)
    print(f"sonde PIXELS bruts (repère) : main {probes['pix'][0]:.1f} %, objet {probes['pix'][1]:.1f} % de la largeur ({time.time() - t0:.0f}s)", flush=True)
    torch.manual_seed(a.seed); m = Baby0(din, nv, npf, a.d, a.nl, a.nh, a.pred_layers, bool(a.sep), a.inv_head, bool(a.resid), bool(a.std_tgt)).to(dev)
    tgt = copy.deepcopy(m.enc).eval()
    for p_ in tgt.parameters(): p_.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05)
    cfg = dict(P=a.P, d=a.d, nl=a.nl, nh=a.nh, pred_layers=a.pred_layers, sep=a.sep, din=din, budgets=bud, world="av_world0", inv_head=a.inv_head,
               resid=a.resid, std_tgt=a.std_tgt, wkw=wkw, T=T)
    state = dict(it=0, hist=[], exams=[])
    if os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        m.load_state_dict(ck["m"]); tgt.load_state_dict(ck["tgt"]); state = ck["state"]
        if "opt" in ck: opt.load_state_dict(ck["opt"])                    # un INSTANTANÉ n'a pas l'optimiseur (reprise depuis une étape)
        if "norm" in ck: st = ck["norm"]                                  # garder la normalisation des sens avec laquelle il a appris
        print(f"REPRISE au pas {state['it']} (étape {stage_of(state['it'])[0]})", flush=True)
    else:
        m.eval(); state["exams"].append(("init", exam(m, tgt, probes, st, VIEW["0a"], a, dev, nv, npf, "init (aléatoire)"))); m.train()
    S = Stream(state["it"], a.bs, a.workers, a.replay, wkw=wkw); md_np, fr_np = layout(nv, npf); ma = None
    md_t, fr_t = torch.from_numpy(md_np).to(dev), torch.from_numpy(fr_np).to(dev)
    while state["it"] < a.total:
        it = state["it"] = state["it"] + 1; stage, f = stage_of(it); S.stage = stage; v = view_at(stage, f)
        fe = min(1.0, (it - starts[4]) / max(1, a.total - starts[4]))          # avancement dans 0e
        if a.outside: S.wkw["p_out"] = {"0d": 0.1 + 0.2 * f, "0e": 0.3 + (0.3 if a.calm == 2 else 0.4) * fe}.get(stage, 0.0)
        if a.calm == 1: S.wkw["calm"] = {"0d": 1 - 0.6 * f, "0e": 0.4 * (1 - fe)}.get(stage, 1.0)
        if a.calm == 2: S.wkw["calm"] = {"0d": 1 - 0.3 * f, "0e": 0.7 - 0.4 * fe}.get(stage, 1.0)   # LENT : jamais chaotique (1 -> 0.7 -> 0.3)
        lr_f = min(1.0, it / 3000) * (0.05 + 0.95 * (1 + math.cos(math.pi * it / a.total)) / 2)
        for g in opt.param_groups: g["lr"] = a.lr * lr_f
        bt_ = S.next(); tok, cmd = to_tok0(bt_, st, v, a.P, dev); B, N, _ = tok.shape
        pairs = masks0(B, nP, nv, rng, a.n_masks, a.touch_masks); full = torch.arange(N, device=dev).expand(B, -1)
        contact = (bt_["TOUCH"].to(dev).abs().sum(-1) > 0).float()                 # (B, T) frames avec contact
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            with torch.no_grad():
                z = F.layer_norm(tgt(tok, full).float(), (a.d,))
                if a.std_tgt:                                       # écart-type courant PAR SENS et PAR DIMENSION des cibles
                    for k_ in range(4): m.tstd[k_].mul_(0.99).add_(0.01 * z[:, md_t == k_].reshape(-1, a.d).std(0))
            jl = 0.0; summ = []
            for c_, g_ in pairs:
                cidx, tidx = _idx(torch.from_numpy(c_).to(dev)), _idx(torch.from_numpy(g_).to(dev))
                zc = m.enc(_gather(tok, cidx), cidx); summ.append(zc.float().mean(1))
                dif = m.pred(zc, tidx, cmd, cidx).float() - _gather(z, tidx)
                dif = dif ** 2 if a.loss == "mse" else dif.abs()        # mse : gradient ∝ erreur -> les cases qui CHANGENT dominent (L1 : vote « rien ne change »)
                if a.std_tgt:                                       # RELATIF : poids moyen 1 par sens (run 2a : 1/std absolu -> EFFONDREMENT)
                    ts_ = m.tstd.clamp_min(0.05); dif = dif * (ts_.mean(1, keepdim=True) / ts_)[md_t[tidx]]
                w_ = 1 + a.contact_w * torch.gather(contact, 1, fr_t[tidx])              # les instants de CONTACT comptent plus
                jl = jl + (dif.mean(-1) * w_).sum() / w_.sum()
            jl = jl / len(pairs); il = torch.zeros((), device=dev)
            if a.inv_w > 0:                                          # CONTINGENCE : vision SEULE (le bras serait un raccourci)
                ts = torch.from_numpy(rng.choice(np.arange(1, T), a.inv_k, replace=False)).to(dev)
                zv = m.enc(tok[:, :nv], full[:, :nv]).view(B, T, npf, -1)
                il = F.smooth_l1_loss(m.inv(zv[:, ts - 1].flatten(0, 1), zv[:, ts].flatten(0, 1)).float(), cmd[:, ts].flatten(0, 1))
            al, gap = torch.zeros((), device=dev), torch.zeros((), device=dev)
            if a.act_w > 0:                                          # CONTRASTE D'ACTION (« compare avec un autre geste »)
                t0_ = int(rng.integers(3, T - 2)); cm_ = torch.from_numpy(fr_np <= t0_).to(dev)
                cidx_ = torch.where(cm_)[0].expand(B, -1); tidx_ = torch.where(~cm_ & (md_t != 1))[0].expand(B, -1)   # futur, sans l'ouïe
                zc_ = m.enc(_gather(tok, cidx_), cidx_); zt_ = _gather(z, tidx_)
                cmd_l = cmd.clone(); cmd_l[:, t0_ + 1:] = cmd.roll(1, 0)[:, t0_ + 1:]                    # mêmes souvenirs, AUTRES gestes futurs
                def err_(c):
                    d_ = m.pred(zc_, tidx_, c, cidx_).float() - zt_
                    return (d_ ** 2 if a.loss == "mse" else d_.abs()).mean((1, 2))
                e_t, e_l = err_(cmd), err_(cmd_l)
                al = F.relu(e_t - e_l + a.act_margin * e_t.detach()).mean(); gap = ((e_l - e_t) / e_t.clamp_min(1e-6)).mean().detach()
        sr = torch.zeros((), device=dev)
        if a.sig_w > 0:
            with torch.autocast("cuda", enabled=False): sr = sigreg(torch.cat(summ).float())
        loss = jl + a.inv_w * il + a.sig_w * sr + a.act_w * al
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([q for n_, q in m.named_parameters() if not (a.inv_clip and n_.startswith('inv.'))], 1.0); opt.step()
        mom = 1 - (1 - a.ema) * (math.cos(math.pi * it / a.total) + 1) / 2
        with torch.no_grad():
            for pt, pc in zip(tgt.parameters(), m.enc.parameters()): pt.mul_(mom).add_(pc.detach(), alpha=1 - mom)
        cur = np.array([jl.item(), il.item(), sr.item(), float(z[:, :nv].std(0).mean()), gap.item()]); ma = cur if ma is None or len(ma) != len(cur) else 0.99 * ma + 0.01 * cur
        if it % 500 == 0:
            state["hist"].append((it, stage, *ma.tolist()))
            print(f"  pas {it:6d} | étape {stage} ({f:.0%})" + (f" | calme {S.wkw.get('calm', 0):.2f}" if a.calm else "") + (f" | hors champ {S.wkw.get('p_out', 0):.2f}" if a.outside else "") + f" | vue σ={v['sigma']:.1f} gris {v['gray']:.1f} | JEPA {ma[0]:.4f} | geste deviné {ma[1]:.4f} "
                  f"| SIGReg {ma[2]:.3f} | écart-type cibles {ma[3]:.3f}" + (f" | AUTRES gestes : erreur {ma[4]:+.0%}" if a.act_w > 0 else "") + f" | {time.time() - t0:.0f}s", flush=True)
        end_stage = it < a.total and stage_of(it + 1)[0] != stage
        if it % a.exam_every == 0 or end_stage or it == a.total:
            m.eval(); r = exam(m, tgt, probes, st, v, a, dev, nv, npf, f"pas {it} étape {stage}" + (" (FIN D'ÉTAPE)" if end_stage else ""), state["exams"][0][1] if state["exams"] else None)
            state["exams"].append((f"pas {it} ({stage})", r)); m.train()
            torch.save(dict(m=m.state_dict(), tgt=tgt.state_dict(), state=state, cfg=cfg, norm=st), a.ckpt.replace(".pt", f"_{stage}_{it // 1000}k.pt"))
        stop = a.stop_at and it >= a.stop_at
        if it % a.ckpt_every == 0 or it == a.total or stop:
            torch.save(dict(m=m.state_dict(), tgt=tgt.state_dict(), opt=opt.state_dict(), state=state, cfg=cfg, norm=st), a.ckpt + ".tmp")
            os.replace(a.ckpt + ".tmp", a.ckpt)
        if stop: print(f"ARRÊT demandé au pas {it} (relancer sans --stop_at pour prolonger)", flush=True); break
    print("\n========== EXAMENS AU FIL DU DÉVELOPPEMENT ==========")
    for tag, r in state["exams"]:
        print(f"  {tag:>14s} | gestes (vue main) {r['geste_main']:+.0%} bras {r['geste_bras']:+.0%} | ruban {r['ruban']:.0%} retard {r['retard']:.0%} "
              f"| vu/senti {r['bras']:.0%} | contact {r['toucher']:.0%} | geste deviné R² {r['inverse_r2']:+.2f} | main {r['main_px']:.1f} % objet {r['objet_px']:.1f} %")
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
