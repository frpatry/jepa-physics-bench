"""
DIAG GESTES MANQUANTS (D17m : la tête n'apprend rien, R² ≈ 0 même pour 1 geste) — l'information est-elle LÀ ?
Bébé gelé ; « maintenant » = image t0, « but » = image t0+K (K = 1, 2, 4, 8) ; une tête GoalHead NEUVE par façon de présenter
maintenant + but, entraînée puis jugée sur des séquences à part (R² des K gestes ; 0 = deviner la moyenne) :
  isolé    : comme l'entraînement D17m — passé seul (tous les sens ≤ t0) + image-but encodée SEULE
  ensemble : passé ≤ t0 + image-but dans la MÊME passe de l'œil (un trou entre les deux)
  entière  : toute la séquence vue par l'œil (fuite de la suite : repère haut, comme la contingence)
  cible    : latents de l'œil-CIBLE (EMA) sur toute la séquence (ce que le JEPA prédit)
  bras     : les 4 nombres du bras senti à t0 et à t0+K (sans la vue ; repère du « corps »)
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_world0 import gen_world0
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_torch, layout, VIEW, H, GoalHead


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
    wkw = {**cfg.get("wkw", {}), "p_out": a.p_out}; wkw.pop("circ", None)
    w = gen_world0(a.n, "0e", T, H, seed=4242, **wkw)
    Ks = (1, 2, 4, 8); t0 = a.t0; assert t0 + max(Ks) < T
    print(f"{a.n} séquences (main hors champ {a.p_out:.0%}), t0 = {t0}, K = {Ks} | {time.time() - t_:.0f}s", flush=True)
    LN = lambda z: F.layer_norm(z.float(), (d,))
    cg_np = np.where(fr <= t0)[0]; vpos = np.searchsorted(cg_np, np.arange(t0 * npf, (t0 + 1) * npf)); apos = int(np.searchsorted(cg_np, nv + 2 * T + t0))
    feats = {k: {"now": [], "arm": [], "goal": {K: [] for K in Ks}} for k in ("isolé", "ensemble", "entière", "cible")}
    raw_arm, cmds = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for i in range(0, a.n, 50):
            b = to_torch(w, slice(i, i + 50)); tok, cmd = to_tok0(b, st, v, P_, dev); B = len(tok)
            g = lambda idx: torch.gather(tok, 1, idx.unsqueeze(-1).expand(-1, -1, tok.size(-1)))
            cg = torch.from_numpy(cg_np).to(dev).expand(B, -1); zc = m.enc(g(cg), cg)
            for k in ("isolé", "ensemble"): feats[k]["now"].append(zc[:, vpos].half().cpu()); feats[k]["arm"].append(zc[:, apos:apos + 1].half().cpu())
            iv = torch.arange(nv, device=dev).expand(B, -1); zf = m.enc(tok[:, :nv], iv).view(B, T, npf, -1)
            zt = LN(tgt(tok, torch.arange(tok.size(1), device=dev).expand(B, -1)))
            ztv = zt[:, :nv].view(B, T, npf, -1); zta = zt[:, nv + 2 * T + t0:nv + 2 * T + t0 + 1]
            feats["entière"]["now"].append(zf[:, t0].half().cpu()); feats["entière"]["arm"].append(zta.half().cpu())
            feats["cible"]["now"].append(ztv[:, t0].half().cpu()); feats["cible"]["arm"].append(zta.half().cpu())
            for K in Ks:
                gi = torch.arange((t0 + K) * npf, (t0 + K + 1) * npf, device=dev).expand(B, -1)
                feats["isolé"]["goal"][K].append(m.enc(g(gi), gi).half().cpu())
                cj = torch.cat([cg, gi], 1); zj = m.enc(g(cj), cj)                              # passé + but dans la même passe
                feats["ensemble"]["goal"][K].append(zj[:, -npf:].half().cpu())
                feats["entière"]["goal"][K].append(zf[:, t0 + K].half().cpu()); feats["cible"]["goal"][K].append(ztv[:, t0 + K].half().cpu())
            raw_arm.append(tok[:, nv + 2 * T:, :4].float().cpu()); cmds.append(cmd.float().cpu())
    cat = lambda L: torch.cat(L)
    raw_arm, cmds = cat(raw_arm), cat(cmds); ntr = int(0.8 * a.n)
    print(f"latents prêts | {time.time() - t_:.0f}s", flush=True)
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
                out.append(f"K={K} R² {r2(pr, y):+.2f} (1er geste {r2(pr[:, :1], y[:, :1]):+.2f})")
        print(f"   {name:>9s} | " + " | ".join(out) + f"  ({time.time() - t_:.0f}s)", flush=True)
    for k, f_ in feats.items():
        now, arm = cat(f_["now"]), cat(f_["arm"]); goal = {K: cat(f_["goal"][K]) for K in Ks}
        fit(k, lambda K, bi, now=now, arm=arm, goal=goal: (now[bi].to(dev).float(), goal[K][bi].to(dev).float(), arm[bi].to(dev).float()),
            lambda: GoalHead(d, npf, max(Ks)))
    class ArmNet(nn.Module):                                              # repère « corps » : bras à t0 et t0+K -> K gestes
        def __init__(s): super().__init__(); s.k = nn.Embedding(9, 64); s.mlp = nn.Sequential(nn.Linear(8 + 64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2 * max(Ks)))
        def forward(s, x, K): return s.mlp(torch.cat([x, s.k.weight[K].expand(len(x), -1)], -1)).view(len(x), max(Ks), 2)
    fit("bras", lambda K, bi: (torch.cat([raw_arm[bi, t0], raw_arm[bi, t0 + K]], -1).to(dev),), ArmNet)


if __name__ == "__main__":
    main()
