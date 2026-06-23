"""
Conduite — PREDICTION DE COLLISION par ROLLOUT latent (option B).

MEME architecture JEPA spatio-temporelle CONJOINTE que d'habitude (SIGReg, sans EMA) :
on monte juste la resolution et on ajoute le masquage TEMPOREL pour imaginer le futur.

  - SSL JEPA : masquage de tokens, melange masquage spatial aleatoire + masquage TEMPOREL
    (toutes les frames > t) -> le predicteur apprend a IMAGINER la suite.
  - Risque (B) : on masque les frames > t, le predicteur imagine les latents futurs, une tete
    collision les classe. K futurs ECHANTILLONNES (bruit gaussien, legitime car SIGReg garde
    les latents isotropes) -> risque = P moyenne de collision. Monte AVANT l'impact -> freinage.
  - Heatmap 12x12 : gradient du risque vers les patches visibles (= ou est le danger).

  python driving_rollout.py --n_nexar 800 --H 96 --patch 8 --T 16 --k_viz 3
  (si OOM : baisser --bs)
"""
import argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2, imageio
from lejepa_video import Enc, patchify, sigreg
from driving_transfer import load_nexar

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16); p.add_argument("--H", type=int, default=96)
    p.add_argument("--patch", type=int, default=8); p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4); p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--reg_w", type=float, default=1.0)
    p.add_argument("--ssl_steps", type=int, default=2500); p.add_argument("--head_steps", type=int, default=1200)
    p.add_argument("--bs", type=int, default=8); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--n_nexar", type=int, default=800); p.add_argument("--k_viz", type=int, default=3)
    p.add_argument("--K", type=int, default=32, help="nb de futurs echantillonnes pour le risque")
    p.add_argument("--noise", type=float, default=1.0, help="echelle du bruit latent (x residu)")
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = a.d_model; nP = a.H // a.patch; npf = nP * nP; ntok = a.T * npf
    X, Y = load_nexar(a.n_nexar, a.T, a.H)                    # X:(n,T,H,H,3) raw RGB [0,1]
    Xp = patchify(X, a.patch); obs = Xp.shape[2]
    Xt = torch.tensor(Xp); Yt = torch.tensor(Y)
    g = torch.Generator().manual_seed(1); pm = torch.randperm(len(Xp), generator=g)
    nt = int(0.7 * len(Xp)); tr, te = pm[:nt].numpy(), pm[nt:].numpy()
    frame_of = torch.arange(ntok, device=dev) // npf          # frame index de chaque token
    print(f"device={dev}  {len(Xp)} clips  H={a.H} patch={a.patch} -> grille {nP}x{nP}, {ntok} tokens/clip", flush=True)

    # ---- 1) SSL JEPA conjoint (spatial aleatoire + temporel) ----
    torch.manual_seed(0)
    enc = Enc(obs, d, ntok, a.n_layer, a.n_head).to(dev)
    pred = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), a.lr)
    for st in range(a.ssl_steps):
        bi = tr[np.random.randint(0, len(tr), a.bs)]; o = Xt[bi].to(dev)
        if np.random.rand() < 0.5:
            m = torch.rand(a.bs, ntok, device=dev) < a.mask_ratio            # masquage spatial
        else:
            t0 = np.random.randint(2, a.T - 1)
            m = (frame_of > t0).unsqueeze(0).expand(a.bs, -1).clone()         # masquage TEMPOREL
        z = enc(o, None); p = pred(enc(o, m))
        loss = (F.smooth_l1_loss(p[m], z[m]) if m.any() else (p - z).pow(2).mean()) + a.reg_w * sigreg(z.reshape(-1, d))
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0: print(f"  [SSL] step {st} loss {loss.item():.3f}", flush=True)
    for prm in list(enc.parameters()) + list(pred.parameters()): prm.requires_grad = False

    # ---- 2) repr causale : contexte OBSERVE (0..t) + futur IMAGINE (>t) ----
    def ctx_fut(o, t0):                                       # encodeur ne voit QUE 0..t0
        m = (frame_of > t0).unsqueeze(0).expand(o.size(0), -1)
        h = enc(o, m)
        ctx = h[:, frame_of <= t0].mean(1)                    # evidence observee (causale)
        fut = pred(h)[:, frame_of > t0]                       # futur imagine (B,nfut,d)
        return ctx, fut

    # residu du predicteur (echelle du bruit pour echantillonner K futurs)
    with torch.no_grad():
        o = Xt[tr[np.random.randint(0, len(tr), min(16, len(tr)))]].to(dev)
        t0h = a.T // 2; z = enc(o, None); _, fu = ctx_fut(o, t0h)
        sigma = (z[:, frame_of > t0h] - fu).std().item()
    print(f"  residu predicteur sigma={sigma:.3f}", flush=True)

    # ---- 3) tete collision sur le futur imagine ----
    head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Linear(128, 2)).to(dev)
    oph = torch.optim.Adam(head.parameters(), 1e-3)
    for st in range(a.head_steps):
        bi = tr[np.random.randint(0, len(tr), a.bs)]; o = Xt[bi].to(dev)
        t0 = np.random.randint(2, a.T - 1)
        with torch.no_grad():
            ctx, fut = ctx_fut(o, t0); gf = torch.cat([ctx, fut.mean(1)], -1)  # contexte + futur
        lab = Yt[torch.tensor(bi)].to(dev)
        oph.zero_grad(); F.cross_entropy(head(gf), lab).backward(); oph.step()

    @torch.no_grad()
    def risk(i, t0, K):                                       # P(collision) par K futurs bruites
        o = Xt[i:i+1].to(dev); ctx, fut = ctx_fut(o, t0)
        nf = fut.size(1)
        futK = fut.expand(K, -1, -1) + sigma * a.noise * torch.randn(K, nf, d, device=dev)
        gf = torch.cat([ctx.expand(K, -1), futK.mean(1)], -1)
        return round(float(head(gf).softmax(-1)[:, 1].mean()), 3)

    # ---- 4) eval : accuracy + anticipation ----
    @torch.no_grad()
    def acc_at(t0):
        ok = 0; tot = 0
        for j in range(0, len(te), 32):
            idx = te[j:j+32]; o = Xt[torch.tensor(idx)].to(dev)
            ctx, fut = ctx_fut(o, t0); pr = head(torch.cat([ctx, fut.mean(1)], -1)).argmax(-1)
            ok += int((pr == Yt[torch.tensor(idx)].to(dev)).sum()); tot += len(idx)
        return ok / tot
    half = a.T // 2
    print(f"\nsonde collision (futur imagine, contexte=demi-clip) acc = {acc_at(half):.2f}", flush=True)
    rc = [risk(int(i), half, a.K) for i in te if Y[i] == 1][:40]
    rn = [risk(int(i), half, a.K) for i in te if Y[i] == 0][:40]
    print(f"anticipation a mi-clip : risque moyen collisions={np.mean(rc):.2f}  vs normales={np.mean(rn):.2f}", flush=True)

    # ---- 5) heatmap (gradient du risque -> patches visibles) ----
    def saliency(i, t0, fv):
        o = Xt[i:i+1].to(dev).detach().requires_grad_(True)
        ctx, fut = ctx_fut(o, t0); gf = torch.cat([ctx, fut.mean(1)], -1); logit = head(gf)[0, 1]
        enc.zero_grad(); pred.zero_grad(); head.zero_grad()
        if o.grad is not None: o.grad.zero_()
        logit.backward()
        s = o.grad.abs().sum(-1)[0][frame_of == fv].reshape(nP, nP).detach().cpu().numpy()
        return (s - s.min()) / (s.max() - s.min() + 1e-6)

    # ---- 6) GIF : image reelle + heatmap + jauge de risque qui monte ----
    S = 288
    pos_te = [int(i) for i in te if Y[i] == 1][:a.k_viz]
    for k, i in enumerate(pos_te):
        rk = [risk(i, min(f, a.T - 2), a.K) for f in range(a.T)]
        gif = []
        for f in range(a.T):
            base = cv2.resize((X[i, f] * 255).astype(np.uint8), (S, S), interpolation=cv2.INTER_LINEAR)
            hm = saliency(i, min(f, a.T - 2), min(f, a.T - 2))
            hb = cv2.GaussianBlur(cv2.resize((hm * 255).astype(np.uint8), (S, S), interpolation=cv2.INTER_CUBIC), (0, 0), 10)
            hc = cv2.cvtColor(cv2.applyColorMap(hb, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            ov = (0.55 * base + 0.45 * hc).clip(0, 255).astype(np.uint8)
            bh = int(rk[f] * (S - 20)); col = (255, 40, 40) if rk[f] > 0.5 else (255, 200, 0)
            cv2.rectangle(ov, (S - 18, S - 10 - bh), (S - 6, S - 10), col, -1)
            cv2.rectangle(ov, (S - 18, 10), (S - 6, S - 10), (255, 255, 255), 1)
            cv2.putText(ov, f"risque {rk[f]:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            gif.append(ov)
        path = f"/content/rollout_clip{k}.gif"; imageio.mimsave(path, gif, duration=0.3)
        print(f"  clip {k}: {path}  | risque final {rk[-1]:.2f}  (max {max(rk):.2f})", flush=True)
    print("\nAfficher : from IPython.display import Image, display", flush=True)
    print("for k in range(%d): display(Image('/content/rollout_clip%%d.gif'%%k))" % len(pos_te), flush=True)

if __name__ == "__main__":
    main()
