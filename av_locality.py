"""
DIAG LOCALITÉ — les tokens de l'encodeur codent-ils LEUR COIN ou TOUTE LA SCÈNE ?
(contrôle « état exact » : le prédicteur apprend l'effet des gestes et planifie 27 % quand l'état est explicite,
mais reste sourd sur nos latents alors que la main y est lisible -> l'info serait ÉTALÉE sur tous les tokens.)

Deux mesures, monde avec main (babillage), frames où la main bouge sans rien pousser :
  - CONCENTRATION : part du changement latent |z(t+1) - z(t)| tombant sur les patches autour de la main
    (voisinage 3×3 de ses positions t et t+1 ; ≈ 14-25 % si le changement est uniforme, ~100 % pour des pixels) ;
  - PART COMMUNE : part de la variance d'un token partagée par tous les patches de la frame (0 % = local).

  python av_locality.py --ckpts /content/drive/MyDrive/jepa_runs/av_dev_v5_60k.pt
"""
import argparse, copy
import numpy as np, torch
from av_jepa import gen_world, NB, enc_config
from av_act import gen_world_act
from av_dev import DevJEPA
from av_dev_long import to_tokens, stereo, T, H
from av_hand_diag import encode

def stats(Z, w, nP):
    """Z (n, T, npf, d) -> concentration du changement autour de la main, part commune."""
    n = len(Z); dz = (Z[:, 1:].float() - Z[:, :-1].float()).abs().sum(-1)            # (n, T-1, npf)
    hp = np.clip((w["HAND"] * nP).astype(int), 0, nP - 1)                              # (n, T, 2) cellule (x, y)
    mv = np.linalg.norm(np.diff(w["HAND"], axis=1), axis=-1) * 32 > 1.0                 # la main a bougé > 1 px
    still = (w["WHO"][:, 1:] < 0)                                                      # sans pousser de disque
    gy, gx = np.divmod(np.arange(nP * nP), nP)
    conc, unif = [], []
    for i in range(n):
        for t in range(T - 1):
            if not (mv[i, t] and still[i, t]): continue
            near = np.zeros(nP * nP, bool)
            for (cx, cy) in (hp[i, t], hp[i, t + 1]): near |= (abs(gx - cx) <= 1) & (abs(gy - cy) <= 1)
            d = dz[i, t].numpy(); conc.append(d[near].sum() / (d.sum() + 1e-9)); unif.append(near.mean())
    zc = Z.float() - Z.float().mean(0, keepdim=True)
    shared = zc.mean(2).pow(2).sum(-1).mean().item() / zc.pow(2).sum(-1).mean().item()
    return float(np.mean(conc)), float(np.mean(unif)), shared, len(conc)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=str, required=True); p.add_argument("--modes", type=str, default="frame,full")
    p.add_argument("--sep_enc", type=int, default=1); p.add_argument("--hum", type=float, default=0.15); p.add_argument("--n", type=int, default=400)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = 4; nP = H // P; npf = nP * nP; nv = T * npf; da = 2 * 2 * NB; W = max(P * P * 3, da)
    w0 = gen_world(2000, T, H, seed=0, a_sub=2, hum=a.hum); A0 = stereo(torch.from_numpy(w0["A"])).reshape(2000, T, -1)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev)); del w0
    w = gen_world_act(a.n, T, H, seed=1004, a_sub=2, hum=a.hum, fric=0.02, babble=1)
    X, A = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)), torch.from_numpy(w["A"])
    tok = torch.cat([to_tokens(X[i:i + 200].to(dev), A[i:i + 200].to(dev), P, st).half().cpu() for i in range(0, a.n, 200)])
    print(f"{'encodeur':>30s} | {'changement près de la main':>26s} | {'(si uniforme)':>13s} | {'part COMMUNE aux patches':>24s}")
    c, u, s, k = stats(tok[:, :nv].reshape(a.n, T, npf, -1), w, nP)
    print(f"{'PIXELS bruts':>30s} | {c:26.0%} | {u:13.0%} | {s:24.0%}   ({k} transitions)", flush=True)
    m0 = DevJEPA(W, da, nv, T, 192, 6, 6, 3).to(dev)
    for ck in a.ckpts.split(","):
        enc = copy.deepcopy(m0.enc); c_ = torch.load(ck, map_location=dev, weights_only=False); enc.load_state_dict(c_["tgt"]); enc.eval(); enc_config(enc, c_, a.sep_enc)
        for md in a.modes.split(","):
            c, u, s, _ = stats(encode(enc, tok, md, nv, npf, dev), w, nP)
            print(f"{ck.split('/')[-1][-16:] + ' [' + md + ']':>30s} | {c:26.0%} | {u:13.0%} | {s:24.0%}", flush=True)

if __name__ == "__main__":
    main()
