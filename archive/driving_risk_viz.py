"""
VISUEL ARTICLE : superposition du RISQUE D'IMPACT sur des images dashcam.

On entraîne un world model LeJEPA sur dashcam (Nexar) + une sonde danger, puis pour des
clips de collision on calcule :
  - OÙ : saillance par gradient (d risque / d patch) -> heatmap spatiale sur l'image.
  - QUAND : risque(t) = P(collision | frames 0..t) -> jauge qui monte vers l'impact.
On dump frames (JPEG b64) + heatmap + courbe de risque -> rendu en superposition.

  python driving_risk_viz.py --n_nexar 400 --k_viz 4
"""
import argparse, json, base64
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2
from lejepa_video import patchify, WM
from driving_transfer import load_nexar, train_wm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16); p.add_argument("--H", type=int, default=48)
    p.add_argument("--patch", type=int, default=8); p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4); p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--reg_w", type=float, default=1.0); p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--n_nexar", type=int, default=400); p.add_argument("--k_viz", type=int, default=4)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    nP = a.H // a.patch
    Xf, Y = load_nexar(a.n_nexar, a.T, a.H)                   # (n,T,H,H,3) float [0,1] RGB
    Xp = patchify(Xf, a.patch); ntok, obs = Xp.shape[1], Xp.shape[2]
    Xpt = torch.tensor(Xp); Yt = torch.tensor(Y)
    g = torch.Generator().manual_seed(1); pm = torch.randperm(len(Xp), generator=g)
    nt = int(0.7 * len(Xp)); tr, te = pm[:nt].numpy(), pm[nt:].numpy()
    enc = train_wm(Xp[tr], ntok, obs, a, dev, "dashcam")      # world model sur dashcam
    # ---- sonde danger (MLP) ----
    @torch.no_grad()
    def feats(idx): return torch.cat([enc.feat(Xpt[torch.tensor(idx[i:i+32])].to(dev)) for i in range(0, len(idx), 32)])
    clf = nn.Sequential(nn.Linear(a.d_model, 256), nn.GELU(), nn.Linear(256, 2)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), 1e-3); Ftr = feats(tr); ytr = Yt[tr].to(dev)
    for _ in range(600):
        opt.zero_grad(); F.cross_entropy(clf(Ftr), ytr).backward(); opt.step()
    with torch.no_grad():
        acc = (clf(feats(te)).argmax(-1) == Yt[te].to(dev)).float().mean().item()
    print(f"sonde danger (dashcam) test acc = {acc:.2f}", flush=True)
    # ---- clips de collision pour la visu ----
    import imageio
    pos_te = [int(i) for i in te if Y[i] == 1][:a.k_viz]
    for k, i in enumerate(pos_te):
        o = Xpt[i:i+1].to(dev).requires_grad_(True)
        logit = clf(enc.enc(o).mean(1))[0, 1]                 # risque (avec gradient vers l'entrée)
        enc.zero_grad(); clf.zero_grad()
        if o.grad is not None: o.grad.zero_()
        logit.backward()
        sal = o.grad.abs().sum(-1)[0].detach().cpu().numpy().reshape(a.T, nP, nP)  # OÙ
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        risk = []                                             # QUAND : risque vu frames 0..t
        with torch.no_grad():
            od = o.detach()
            for t in range(a.T):
                m = torch.zeros(1, ntok, dtype=torch.bool, device=dev); m[:, (t + 1) * nP * nP:] = True
                risk.append(round(float(clf(enc.enc(od, m).mean(1)).softmax(-1)[0, 1]), 2))
        S = 256                                               # taille d'affichage
        gif = []
        for f in range(a.T):
            base = cv2.resize((Xf[i, f] * 255).astype(np.uint8), (S, S), interpolation=cv2.INTER_LINEAR)
            hb = cv2.resize((sal[f] * 255).astype(np.uint8), (S, S), interpolation=cv2.INTER_CUBIC)
            hb = cv2.GaussianBlur(hb, (0, 0), 12)             # lissage -> heatmap propre
            hc = cv2.cvtColor(cv2.applyColorMap(hb, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            ov = (0.55 * base + 0.45 * hc).clip(0, 255).astype(np.uint8)
            # jauge de risque (barre verticale a droite) + texte
            bh = int(risk[f] * (S - 20)); col = (255, 40, 40) if risk[f] > 0.5 else (255, 200, 0)
            cv2.rectangle(ov, (S - 18, S - 10 - bh), (S - 6, S - 10), col, -1)
            cv2.rectangle(ov, (S - 18, 10), (S - 6, S - 10), (255, 255, 255), 1)
            cv2.putText(ov, f"risque {risk[f]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            gif.append(ov)
        path = f"/content/risk_clip{k}.gif"; imageio.mimsave(path, gif, duration=0.3)
        verdict = "DANGER DETECTE" if risk[-1] > 0.5 else "rate (risque bas)"
        print(f"  clip {k}: {path}  | risque final {risk[-1]:.2f}  -> {verdict}", flush=True)
    print("\nAfficher les GIF dans Colab :", flush=True)
    print("from IPython.display import Image, display", flush=True)
    print("for k in range(%d): display(Image('/content/risk_clip%%d.gif'%%k))" % len(pos_te), flush=True)

if __name__ == "__main__":
    main()
