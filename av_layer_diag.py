"""
DIAG B — OÙ l'encodeur perd-il la précision de position ? (instrument)
Les pixels bruts situent la main à ~1 px (3.4 % de la largeur) ; nos encodeurs la rendent à ~5 px.
Pour chaque instantané et CHAQUE COUCHE (0 = plongement d'entrée, 1..L = après chaque couche, L+ = sortie normalisée),
un lecteur (même architecture que les examens) lit la position de la MAIN et des 2 disques dans les tokens visuels d'UNE frame.
Disques : perte invariante à l'ordre (min sur l'échange) -> pas d'artefact de tri.

Lecture : précision qui se dégrade AU FIL DES COUCHES -> l'architecture/l'objectif efface le détail en profondeur
(remède : lire une couche intermédiaire, ou contraindre) ; déjà mauvaise à la couche 0-1 -> c'est le plongement des patches ;
qui se dégrade AU FIL DE L'ENTRAÎNEMENT (10k -> 70k) -> c'est l'objectif JEPA qui jette le détail.

  python av_layer_diag.py --ckpts .../av_dev_v7_10k.pt,.../av_dev_v7_70k.pt,.../av_dev_v5_60k.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
from av_jepa import NB, enc_config, _layer_masked
from av_act import gen_world_act
from av_dev import DevJEPA
from av_dev_long import _PosReader, T, H

@torch.no_grad()
def layer_feats(enc, X, P, dev, bs=256):
    """X (N, H, H, 3) uint8 (frames) -> liste [couche 0..L, sortie] de (N, npf, d) — tokens VISUELS d'une frame seule
    (encodeurs séparés / locaux : la vision n'écoute pas le son, on ne donne que les patches, position de la frame 0)."""
    g = H // P; npf = g * g; outs = None
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(dev).float() / 255.0; B = len(x)
        tok = x.reshape(B, g, P, g, P, 3).permute(0, 1, 3, 2, 4, 5).reshape(B, npf, P * P * 3)
        tok = F.pad(tok, (0, enc.ev.in_features - tok.size(-1)))                 # largeur commune image/son (comme to_tokens)
        idx = torch.arange(npf, device=dev).expand(B, -1); isa = torch.zeros_like(idx, dtype=torch.bool)
        h = enc.ev(tok) + enc.mod(isa.long()) + enc.pos(idx); hs = [h]
        if enc.local > 0:
            cell = idx % npf; ry, rx = cell // enc.nP, cell % enc.nP
            dyx = torch.maximum((ry.unsqueeze(2) - ry.unsqueeze(1)).abs(), (rx.unsqueeze(2) - rx.unsqueeze(1)).abs())
            for j, L in enumerate(enc.tr.layers):
                R = enc.local if j < enc.local_layers else 0
                h = _layer_masked(L, h, (dyx <= R).unsqueeze(1)); hs.append(h)
        else:
            for L in enc.tr.layers: h = L(h); hs.append(h)
        hs.append(enc.ln(h))
        hs = [z.half().cpu() for z in hs]
        outs = [[z] for z in hs] if outs is None else [o + [z] for o, z in zip(outs, hs)]
    return [torch.cat(o) for o in outs]

def ridge(Z, y, dev):
    """SONDE LINÉAIRE RIDGE résolue EXACTEMENT (aucun entraînement qui peut échouer) : tokens d'une frame aplatis -> position de la
    MAIN (x, y) et du MILIEU des 2 disques (pas d'ambiguïté « quel disque »). λ choisi sur une validation. -> erreurs en %."""
    N = len(Z); k = int(0.7 * N); v = int(0.85 * N)
    X = Z.reshape(N, -1).to(dev).float(); X = X - X[:k].mean(0)
    t = torch.cat([y[:, 4:6], (y[:, 0:2] + y[:, 2:4]) / 2], 1).to(dev); tm = t[:k].mean(0)
    G = X[:k].T @ X[:k]; b = X[:k].T @ (t[:k] - tm); s = G.diagonal().mean(); best = None
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        Wr = torch.linalg.solve(G + lam * s * torch.eye(len(G), device=dev), b)
        ev = ((X[k:v] @ Wr + tm - t[k:v]).view(-1, 2, 2).norm(dim=-1)).mean().item()
        if best is None or ev < best[0]: best = (ev, Wr)
    e = (X[v:] @ best[1] + tm - t[v:]).view(-1, 2, 2).norm(dim=-1).mean(0) * 100
    return float(e[0]), float(e[1])

def read(Z, y, dev, steps):
    """-> erreur (% de la largeur) main, disques (invariant à l'ordre)."""
    k = int(0.8 * len(Z)); mu, sd = y[:k].mean(0), y[:k].std(0) + 1e-6
    torch.manual_seed(0); r = _PosReader(Z.size(-1), Z.size(1), nout=6).to(dev)
    opt = torch.optim.AdamW(r.parameters(), 3e-4, weight_decay=0.05); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    sw = [2, 3, 0, 1, 4, 5]
    for _ in range(steps):
        bi = torch.randint(0, k, (128,)); p = r(Z[bi].to(dev).float()); t = ((y[bi] - mu) / sd).to(dev)
        l = torch.minimum(((p - t) ** 2).mean(-1), ((p[:, sw] - t) ** 2).mean(-1)).mean()
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
    r.eval()
    with torch.no_grad(): p = torch.cat([r(Z[i:i + 512].to(dev).float()).cpu() for i in range(k, len(Z), 512)]) * sd + mu
    t = y[k:]; ps = p[:, sw]
    better = ((ps - t)[:, :4] ** 2).sum(-1) < ((p - t)[:, :4] ** 2).sum(-1); p = torch.where(better[:, None], ps, p)
    e = (p - t).view(-1, 3, 2).norm(dim=-1) * 100
    return float(e[:, 2].mean()), float(e[:, :2].mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True); ap.add_argument("--n", type=int, default=800); ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    P = 4; g = H // P; npf = g * g; nv = T * npf; da = 2 * 2 * NB; W = max(P * P * 3, da)
    w = gen_world_act(a.n, T, H, seed=11, a_sub=2, fric=0.02, babble=1)
    X = torch.from_numpy((w["X"] * 255).round().astype(np.uint8))[:, ::2].flatten(0, 1)          # 1 frame sur 2
    y = torch.from_numpy(np.concatenate([w["POS"].reshape(a.n, T, 4), w["HAND"]], -1))[:, ::2].flatten(0, 1).float()
    tokpix = X.float().div(255).reshape(len(X), g, P, g, P, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(X), npf, P * P * 3)
    hm, dm = ridge(tokpix, y, dev)
    print(f"{'encodeur':>22s} | {'couche':>8s} | {'MAIN':>7s} | {'milieu disques':>14s}   (SONDE LINÉAIRE ridge ; erreur en % de la largeur ; 1 px = 3.1 %)")
    print(f"{'PIXELS bruts':>22s} | {'-':>8s} | {hm:6.1f}% | {dm:13.1f}%   ({time.time() - t0:.0f}s)", flush=True)
    m0 = DevJEPA(W, da, nv, T, 192, 6, 6, 3).to(dev)
    for ck in a.ckpts.split(","):
        enc = copy.deepcopy(m0.enc); c_ = torch.load(ck, map_location=dev, weights_only=False); enc.load_state_dict(c_["tgt"]); enc.eval(); enc_config(enc, c_, 1)
        Fs = layer_feats(enc, X, P, dev); name = ck.split("/")[-1].replace("av_dev_", "").replace(".pt", "")
        for j, Z in enumerate(Fs):
            hm, dm = ridge(Z, y, dev); lab = "entrée" if j == 0 else ("sortie" if j == len(Fs) - 1 else f"{j}")
            print(f"{name:>22s} | {lab:>8s} | {hm:6.1f}% | {dm:13.1f}%   ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
