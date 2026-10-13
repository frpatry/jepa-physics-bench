"""
CE QUE L'ŒIL DISTINGUE — GIF animé de diagnostic (instrument, aucun entraînement du bébé).
Pour chaque case de vue (4×4 pixels), une petite sonde linéaire lit dans le latent de l'œil-cible : MAIN, OBJET ou FOND.
Étiquettes des cases (instrument seulement) : couleur des pixels — la main est blanche, les objets colorés, le fond noir.
GIF : à gauche la vraie image ; puis, pour chaque œil, l'image en gris avec les cases vues comme MAIN (rouge) et OBJET (bleu),
et une boîte autour de chaque détection. Pour chaque œil, on affiche aussi la justesse équilibrée de la sonde (hasard 33 %).

  python av_seegif.py --ckpts phase0_rD0_0d_34k.pt,phase0_rD1_0d_34k.pt --out /content/drive/MyDrive/jepa_runs/oeil.gif
"""
import argparse, copy, os, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from av_world0 import gen_world0
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, layout, VIEW, H
from scipy import ndimage

COLS = {1: (255, 60, 60), 2: (60, 140, 255)}                      # main rouge, objet bleu


def labels(X, P):
    """X (n, T, H, H, 3) dans [0,1] -> étiquette par case (n, T, nP*nP) : 0 fond, 1 main (blanc), 2 objet (coloré)."""
    n, T = X.shape[:2]; nP = H // P
    q = X.reshape(n, T, nP, P, nP, P, 3).transpose(0, 1, 2, 4, 3, 5, 6).reshape(n, T, nP * nP, P * P, 3)
    white = (q.min(-1) > 0.85).mean(-1); colored = ((q.max(-1) > 0.15) & (q.min(-1) <= 0.85)).mean(-1)
    lab = np.zeros((n, T, nP * nP), np.int64); lab[colored > 0.3] = 2; lab[white > 0.3] = 1
    return lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True); p.add_argument("--dir", default="/content/drive/MyDrive/jepa_runs")
    p.add_argument("--out", default="/content/drive/MyDrive/jepa_runs/oeil.gif"); p.add_argument("--n", type=int, default=1500)
    p.add_argument("--show", type=int, default=4, help="séquences dans le GIF"); p.add_argument("--scale", type=int, default=8)
    p.add_argument("--imagine", type=int, default=1, help="1 = à partir de la moitié de la séquence, les cases montrées sont IMAGINÉES (contexte = la 1re moitié, "
                   "avec ses vrais gestes) : on voit si l'imagination garde la main et l'objet")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    names = [c.strip() for c in a.ckpts.split(",")]
    cks = [torch.load(os.path.join(a.dir, c), map_location=dev, weights_only=False) for c in names]
    cfg0 = cks[0]["cfg"]; T = int(cfg0.get("T", 16)); av_phase0.T = T; P = cfg0["P"]; nP = H // P; npf = nP * nP; nv = T * npf; v = VIEW["0e"]
    wkw = {**cfg0.get("wkw", {}), "p_parent": 0.0}                    # sans la main du parent (elle serait « colorée »)
    w = gen_world0(a.n, "0e", T, H, seed=2026, **wkw)
    keep = np.where(w["NOBJ"] >= 1)[0]; b = to_torch(to_np([{k: w[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}])); n = len(keep)
    X = w["X"][keep]; lab = labels(X, P); ntr = int(0.8 * n)
    print(f"{n} séquences, cases : fond {np.mean(lab == 0):.0%} / main {np.mean(lab == 1):.1%} / objet {np.mean(lab == 2):.1%} | {time.time() - t0:.0f}s", flush=True)
    show = np.arange(ntr, n)[:a.show]; maps = []
    for c, ck in zip(names, cks):
        cfg = ck["cfg"]; d = cfg["d"]; m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"])
        enc = copy.deepcopy(m.enc); enc.load_state_dict(ck["tgt"]); enc.eval(); Z = []
        with torch.no_grad():
            for i0 in range(0, n, 50):
                tk, _ = to_tok0({k: x[i0:i0 + 50] for k, x in b.items()}, ck["norm"], v, P, dev); ix = torch.arange(nv, device=dev).expand(len(tk), -1)
                Z.append(F.layer_norm(enc(tk[:, :nv], ix).float(), (d,)).view(len(tk), T, npf, d).half().cpu())
        Z = torch.cat(Z); h = T // 2
        Zs = Z[torch.from_numpy(show)].clone()
        if a.imagine:                                                   # IMAGINATION : frames h.. prédites depuis le contexte 0..h-1 + vrais gestes
            md, fr = layout(nv, npf); ci = torch.from_numpy(np.where(fr < h)[0]).to(dev)[None]
            ti = torch.from_numpy(np.where((fr >= h) & (md == 0))[0]).to(dev)[None]
            with torch.no_grad():
                tk, cmd = to_tok0({k: x[torch.from_numpy(show)] for k, x in b.items()}, ck["norm"], v, P, dev); B_ = len(tk)
                ctx = m.context(torch.gather(tk, 1, ci.expand(B_, -1).unsqueeze(-1).expand(-1, -1, tk.size(-1))), ci.expand(B_, -1))
                zi = F.layer_norm(m.pred(ctx, ti.expand(B_, -1), cmd, ci.expand(B_, -1)).float(), (d,)).view(B_, T - h, npf, d)
            Zs[:, h:] = zi.half().cpu()
        # sonde linéaire 3 classes, équilibrée (les cases main/objet sont rares)
        Xtr = Z[:ntr].reshape(-1, d).float().to(dev); Ytr = torch.from_numpy(lab[:ntr].reshape(-1)).to(dev)
        cnt = torch.bincount(Ytr, minlength=3).float(); wcls = (cnt.sum() / (3 * cnt.clamp_min(1)))
        torch.manual_seed(0); lin = torch.nn.Linear(d, 3).to(dev); opt = torch.optim.AdamW(lin.parameters(), 3e-3, weight_decay=1e-3)
        for _ in range(1500):
            bi = torch.randint(0, len(Xtr), (4096,), device=dev); l = F.cross_entropy(lin(Xtr[bi]), Ytr[bi], weight=wcls)
            opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            pr = torch.cat([lin(Z[i].float().to(dev)).softmax(-1).cpu() for i in range(ntr, n)])    # ((n-ntr)*T, npf, 3) aplati par séquence
            pr = pr.view(n - ntr, T, npf, 3)
            prs = torch.stack([lin(Zs[j].float().to(dev)).softmax(-1).cpu() for j in range(len(show))])        # séquences du GIF (imaginées après h)
        yt = lab[ntr:]; yh = pr.argmax(-1).numpy(); bacc = np.mean([np.mean(yh[yt == k] == k) for k in range(3)])
        rec = [np.mean(yh[yt == k] == k) for k in range(3)]
        print(f"  {c}: justesse équilibrée {bacc:.0%} (fond {rec[0]:.0%}, main {rec[1]:.0%}, objet {rec[2]:.0%})", flush=True)
        maps.append((c.replace(".pt", "").replace("phase0_", ""), bacc, prs.numpy()))
        del Z, Xtr, m, enc; torch.cuda.empty_cache() if dev == "cuda" else None
    # ---------- GIF
    s = a.scale; W1 = H * s; pad = 6; head = 34; frames = []
    for j, i in enumerate(show):
        for t in range(T):
            img = Image.new("RGB", ((W1 + pad) * (1 + len(maps)) + pad, W1 + head + pad), (20, 20, 20)); dr = ImageDraw.Draw(img)
            real = Image.fromarray((X[i, t] * 255).astype(np.uint8)).resize((W1, W1), Image.NEAREST)
            img.paste(real, (pad, head)); dr.text((pad, 4), f"vraie image  séq {j + 1}  t={t}", fill=(230, 230, 230))
            if a.imagine and t >= T // 2: dr.text((pad, 18), "cadre jaune = IMAGINÉ", fill=(255, 210, 0))
            gray = np.repeat((X[i, t].mean(-1, keepdims=True) * 0.6 * 255).astype(np.uint8), 3, -1)
            for k, (nm, bacc, pr) in enumerate(maps):
                pm = pr[j, t]; base = Image.fromarray(gray).resize((W1, W1), Image.NEAREST); ov = np.array(base).astype(np.float32)
                for cl in (1, 2):
                    a_ = pm[:, cl].reshape(nP, nP); a_ = np.kron(a_, np.ones((P * s, P * s)))[..., None]
                    ov = ov * (1 - 0.65 * a_) + np.array(COLS[cl]) * 0.65 * a_
                tile = Image.fromarray(ov.clip(0, 255).astype(np.uint8)); dt = ImageDraw.Draw(tile)
                cls = pm.argmax(-1).reshape(nP, nP)
                for cl in (1, 2):                                      # une boîte par groupe de cases détectées (composantes connexes)
                    lab_, nlab = ndimage.label(cls == cl)
                    for q in range(1, nlab + 1):
                        rr, cc = np.where(lab_ == q)
                        dt.rectangle([cc.min() * P * s, rr.min() * P * s, (cc.max() + 1) * P * s - 1, (rr.max() + 1) * P * s - 1], outline=COLS[cl], width=2)
                if a.imagine and t >= T // 2: dt.rectangle([0, 0, W1 - 1, W1 - 1], outline=(255, 210, 0), width=3)
                x0 = pad + (k + 1) * (W1 + pad); img.paste(tile, (x0, head))
                dr.text((x0, 4), f"{nm}", fill=(230, 230, 230)); dr.text((x0, 18), f"justesse {bacc:.0%}  rouge=main bleu=objet", fill=(170, 170, 170))
            frames.append(img)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=350, loop=0)
    print(f"GIF : {a.out} ({len(frames)} images) | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
