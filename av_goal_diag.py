"""
DIAG GESTES MANQUANTS — l'information « quels gestes mènent au but » est-elle LÀ, et la tête sait-elle la lire ?
Bébé gelé ; « maintenant » = image t0, « but » = image t0+K (K = 1, 2, 4, 8) ; une tête NEUVE par façon de présenter maintenant + but,
entraînée puis jugée sur des séquences à part (R² des K gestes ; 0 = deviner la moyenne ; « pousse » = séquences avec contact senti entre t0 et t0+K).
1er passage (2026-10-22) : isolé 0.00, entière 0.00 (!), cible 0.31, bras 0.78 -> soupçons : (a) tête v1 sur latents NON normalisés de l'œil
en ligne (constante), (b) « ensemble » mal rangé (l'œil groupe les sens par blocs contigus). Corrigé ici : goal_encode trie les indices,
tête v2 normalise et compare case à case.
  isolé         : image-but encodée SEULE (D17m 1er essai)
  ens. +main    : passé ≤ t0 + image-but (avec sa main) dans la MÊME passe de l'œil, rien entre les deux
  ens. SANS main: idem, image-but SANS la main (but « objet seulement », comme Push-T)
  entière       : toute la séquence vue par l'œil (fuite de la suite : repère haut) — tête v2 ET tête v1
  cible         : œil-CIBLE (EMA) sur toute la séquence (fuite)
  bras          : 4 nombres du bras senti à t0 et t0+K (repère « corps »)
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_world0 import gen_world0
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_torch, layout, VIEW, H, GoalHead, goal_encode


class GoalHeadV1(nn.Module):                                              # la tête du 1er essai D17m (sans normalisation ni différence)
    def __init__(s, d, npf, kmax, h=192, nl=3):
        super().__init__()
        s.kmax = kmax; s.inp = nn.Linear(d, h); s.pos = nn.Parameter(torch.randn(1, 2 * npf + 1, h) * 0.02)
        s.q = nn.Parameter(torch.randn(1, kmax, h) * 0.02); s.kemb = nn.Embedding(kmax + 1, h)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, batch_first=True, dropout=0.0, activation="gelu"), nl); s.out = nn.Linear(h, 2)
    def forward(s, zn, zg, za, K):
        B = len(zn); x = s.inp(torch.cat([zn, zg, za], 1)) + s.pos; q = s.q.expand(B, -1, -1) + s.kemb.weight[K][None, None]
        return s.out(s.tr(torch.cat([x, q], 1))[:, -s.kmax:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--n", type=int, default=3000); p.add_argument("--t0", type=int, default=5)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--p_out", type=float, default=0.0, help="part des séquences main hors champ (entraînement 0e ≈ 0.56)")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t_ = time.time()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    T = int(cfg.get("T", 16)); av_phase0.T = T
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"], strict=False); m.eval()
    tgt = baby_from_cfg(cfg, dev).enc; tgt.load_state_dict(ck["tgt"]); tgt.eval()
    P_ = cfg["P"]; npf = (H // P_) ** 2; nv = T * npf; d = cfg["d"]; md, fr = layout(nv, npf); v = VIEW["0e"]
    wkw = {**cfg.get("wkw", {}), "p_out": a.p_out, "nohand": True}; wkw.pop("circ", None)
    w = gen_world0(a.n, "0e", T, H, seed=4242, **wkw)
    Ks = (1, 2, 4, 8); t0 = a.t0; assert t0 + max(Ks) < T
    print(f"{a.n} séquences (main hors champ {a.p_out:.0%}), t0 = {t0}, K = {Ks} | {time.time() - t_:.0f}s", flush=True)
    LN = lambda z: F.layer_norm(z.float(), (d,))
    names = ("isolé", "ens. +main", "ens. SANS main", "entière", "cible")
    feats = {k: {"now": {K: [] for K in Ks}, "arm": {K: [] for K in Ks}, "goal": {K: [] for K in Ks}} for k in names}   # « maintenant » vu AVEC le but : dépend de K
    raw_arm, cmds, touch = [], [], []
    scale = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for i in range(0, a.n, 50):
            b = to_torch(w, slice(i, i + 50)); tok, cmd = to_tok0(b, st, v, P_, dev); B = len(tok)
            bn = dict(b); bn["X"] = b["XNH"]; tokn, _ = to_tok0(bn, st, v, P_, dev)
            g = lambda tk, idx: torch.gather(tk, 1, idx.unsqueeze(-1).expand(-1, -1, tk.size(-1)))
            iv = torch.arange(nv, device=dev).expand(B, -1); zf = m.enc(tok[:, :nv], iv).view(B, T, npf, -1)
            zt = LN(tgt(tok, torch.arange(tok.size(1), device=dev).expand(B, -1)))
            ztv = zt[:, :nv].view(B, T, npf, -1); zta = zt[:, nv + 2 * T + t0:nv + 2 * T + t0 + 1]
            cg_np = np.where(fr <= t0)[0]; cg = torch.from_numpy(cg_np).to(dev).expand(B, -1); zc = m.enc(g(tok, cg), cg)   # passé seul (isolé)
            vpos = np.searchsorted(cg_np, np.arange(t0 * npf, (t0 + 1) * npf)); apos = int(np.searchsorted(cg_np, nv + 2 * T + t0))
            if i == 0: scale = {"œil en ligne (vue)": float(zf.float().abs().mean()), "œil-cible normalisé": float(ztv.float().abs().mean())}
            for K in Ks:
                zn1, zg1, za1 = goal_encode(m.enc, tok, tok, t0, K, nv, npf, T, fr)
                zn2, zg2, za2 = goal_encode(m.enc, tok, tokn, t0, K, nv, npf, T, fr)
                gi = torch.arange((t0 + K) * npf, (t0 + K + 1) * npf, device=dev).expand(B, -1)
                feats["isolé"]["goal"][K].append(m.enc(g(tok, gi), gi).half().cpu())
                feats["ens. +main"]["goal"][K].append(zg1.half().cpu()); feats["ens. SANS main"]["goal"][K].append(zg2.half().cpu())
                feats["entière"]["goal"][K].append(zf[:, t0 + K].half().cpu()); feats["cible"]["goal"][K].append(ztv[:, t0 + K].half().cpu())
                for k, zn_, za_ in (("isolé", zc[:, vpos], zc[:, apos:apos + 1]), ("ens. +main", zn1, za1), ("ens. SANS main", zn2, za2),
                                    ("entière", zf[:, t0], zta), ("cible", ztv[:, t0], zta)):
                    feats[k]["now"][K].append(zn_.half().cpu()); feats[k]["arm"][K].append(za_.half().cpu())
            raw_arm.append(tok[:, nv + 2 * T:, :4].float().cpu()); cmds.append(cmd.float().cpu()); touch.append((b["TOUCH"].abs().sum(-1) > 0).cpu())
    cat = lambda L: torch.cat(L)
    raw_arm, cmds, touch = cat(raw_arm), cat(cmds), cat(touch); ntr = int(0.8 * a.n)
    print(f"latents prêts | échelle moyenne |z| : " + ", ".join(f"{k} {x:.2f}" for k, x in scale.items()) + f" | {time.time() - t_:.0f}s", flush=True)
    def r2(p_, y_): return float(1 - ((p_ - y_) ** 2).sum() / ((y_ - y_.mean((0, 1), keepdim=True)) ** 2).sum().clamp_min(1e-8))
    def fit(name, getx, mk):
        torch.manual_seed(0); net = mk().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01)
        for s in range(a.steps):
            K = Ks[s % len(Ks)]; bi = torch.randint(0, ntr, (256,))
            pr = net(*getx(K, bi), K)[:, :K]; l = F.smooth_l1_loss(pr, cmds[bi, t0 + 1:t0 + K + 1].to(dev)); opt.zero_grad(); l.backward(); opt.step()
        net.eval(); out = []
        with torch.no_grad():
            for K in Ks:
                bi = torch.arange(ntr, a.n); pr = net(*getx(K, bi), K)[:, :K].cpu(); y = cmds[bi, t0 + 1:t0 + K + 1]
                hc = touch[bi, t0 + 1:t0 + K + 1].any(1)
                out.append(f"K={K} {r2(pr, y):+.2f} (pousse {r2(pr[hc], y[hc]) if int(hc.sum()) > 10 else float('nan'):+.2f}, n={int(hc.sum())})")
        print(f"   {name:>18s} | " + " | ".join(out) + f"  ({time.time() - t_:.0f}s)", flush=True)
    for k in names:
        f_ = feats[k]; now, arm, goal = ({K: cat(f_[x][K]) for K in Ks} for x in ("now", "arm", "goal"))
        gx = lambda K, bi, now=now, arm=arm, goal=goal: (now[K][bi].to(dev).float(), goal[K][bi].to(dev).float(), arm[K][bi].to(dev).float())
        fit(k + " (v2)", gx, lambda: GoalHead(d, npf, max(Ks)))
        if k == "entière": fit(k + " (v1)", gx, lambda: GoalHeadV1(d, npf, max(Ks)))
    class ArmNet(nn.Module):                                              # repère « corps » : bras à t0 et t0+K -> K gestes
        def __init__(s): super().__init__(); s.k = nn.Embedding(9, 64); s.mlp = nn.Sequential(nn.Linear(8 + 64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2 * max(Ks)))
        def forward(s, x, K): return s.mlp(torch.cat([x, s.k.weight[K].expand(len(x), -1)], -1)).view(len(x), max(Ks), 2)
    fit("bras", lambda K, bi: (torch.cat([raw_arm[bi, t0], raw_arm[bi, t0 + K]], -1).to(dev),), ArmNet)


if __name__ == "__main__":
    main()
