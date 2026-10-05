"""
DIAGNOSTIC — la VISION s'effondre-t-elle seule ? (hypothèse : l'encodeur cible rend ses tokens visuels
quasi constants d'une scène à l'autre — seule la position du patch compte — ce qui rend la cible visuelle
trivialement prévisible, pendant que l'AUDIO garde de la variance ; SIGReg sur le résumé de scène ne voit rien.)
Pour chaque instantané : écart-type ENTRE SCÈNES des tokens, par position, moyenné — vision vs audio.

  python av_vis_std.py --hum 0.15 --ckpts a.pt b.pt ...
"""
import argparse, copy
import numpy as np, torch
from av_jepa import gen_world, NB
from av_dev import DevJEPA, represent
from av_dev_long import to_tokens, stereo, T, H

def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpts", nargs="+"); p.add_argument("--hum", type=float, default=0.15)
    p.add_argument("--n", type=int, default=600); a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = 4; nP = H // P; npf = nP * nP; nv = T * npf; da = 2 * 2 * NB; W = max(P * P * 3, da)
    w0 = gen_world(2000, T, H, seed=0, a_sub=2, hum=a.hum); A0 = stereo(torch.from_numpy(w0["A"])).reshape(2000, T, -1)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev))
    w = gen_world(a.n, T, H, seed=1000, a_sub=2, hum=a.hum)
    X, A = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)), torch.from_numpy(w["A"])
    tok = torch.cat([to_tokens(X[i:i + 200].to(dev), A[i:i + 200].to(dev), P, st).half().cpu() for i in range(0, a.n, 200)])
    m = DevJEPA(W, da, nv, T, 192, 6, 6, 3).to(dev)
    for c in [None] + a.ckpts:
        enc = copy.deepcopy(m.enc)
        if c: enc.load_state_dict(torch.load(c, map_location=dev, weights_only=False)["tgt"])
        enc.eval(); Z = represent(type("W", (), {"enc": enc})(), tok, np.ones(nv + T, bool), dev).float()   # (n, N, d)
        zv, za = Z[:, :nv], Z[:, nv:]
        sv, sa = zv.std(0).mean().item(), za.std(0).mean().item()      # entre scènes, par position et dimension
        nvn, nan_ = zv.norm(dim=-1).mean().item(), za.norm(dim=-1).mean().item()
        print(f"{(c or 'aléatoire').split('/')[-1]:>28s} | écart-type entre scènes : VISION {sv:.3f} | AUDIO {sa:.3f} | ratio V/A {sv / sa:.2f} "
              f"| normes V {nvn:.1f} A {nan_:.1f}", flush=True)

if __name__ == "__main__":
    main()
