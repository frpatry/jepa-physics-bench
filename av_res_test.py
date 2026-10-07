"""
TEST RÉSOLUTION (instrument, ~10 min GPU) — quelle précision de position la VUE permet-elle, selon la taille
d'image et la grille de patches ? Lecteur supervisé (même architecture que les examens) sur les PIXELS BRUTS
d'une frame (patches), monde avec main (babillage). Plafond de ce qu'un encodeur pourrait transmettre.

Erreur = distance moyenne prédite/vraie, en % de la largeur de l'image (disques triés par x).
Repère : rayon d'un disque = 12 %, demi-côté de la main ≈ 6 %.

  python av_res_test.py
"""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from av_act import gen_world_act
from av_dev_long import _PosReader

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, default="32:4,64:8,64:4,128:8")
    p.add_argument("--n", type=int, default=800); p.add_argument("--steps", type=int, default=4000)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time(); T = 16
    print(f"{'image':>7s} | {'patch':>5s} | {'grille':>6s} | {'disques : erreur':>16s} | {'main : erreur':>13s} | (en % de la largeur ; rayon disque 12 %)")
    for cfg in a.configs.split(","):
        H, P = map(int, cfg.split(":")); g = H // P
        w = gen_world_act(a.n, T, H, seed=7, a_sub=2, fric=0.02, babble=1)
        X = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)).flatten(0, 1)          # (n*T, H, H, 3)
        tok = X.reshape(-1, g, P, g, P, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(X), g * g, P * P * 3)
        P2 = w["POS"]; o_ = np.argsort(P2[..., 0], axis=-1)
        y = torch.from_numpy(np.concatenate([np.take_along_axis(P2, o_[..., None], axis=2).reshape(a.n, T, 4), w["HAND"]], -1)).float().flatten(0, 1)
        k = int(0.8 * len(tok)); mu, sd = y[:k].mean(0), y[:k].std(0) + 1e-6
        torch.manual_seed(0); r = _PosReader(tok.size(-1), g * g, nout=6).to(dev)
        opt = torch.optim.AdamW(r.parameters(), 3e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        for _ in range(a.steps):
            bi = torch.randint(0, k, (128,))
            l = F.mse_loss(r(tok[bi].to(dev).float() / 255.0), ((y[bi] - mu) / sd).to(dev)); opt.zero_grad(); l.backward(); opt.step(); sched.step()
        r.eval()
        with torch.no_grad():
            pr = torch.cat([r(tok[i:i + 256].to(dev).float() / 255.0).cpu() for i in range(k, len(tok), 256)]) * sd + mu
        yt = y[k:]; e = (pr - yt).view(-1, 3, 2).norm(dim=-1) * 100                       # (N, 3) en % de la largeur
        print(f"{H:>5d}px | {P:>3d}px | {g:>3d}×{g:<2d} | {e[:, :2].mean():15.1f}% | {e[:, 2].mean():12.1f}% | ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
