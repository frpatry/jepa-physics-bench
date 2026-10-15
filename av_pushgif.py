"""
GIF « POUSSÉE PAR POUSSÉE » (diagnostic de l'IMAGINATION, instrument seulement — le bébé n'apprend rien ici).
Séquences du monde 0e à UN objet où la main POUSSE l'objet pendant les images 8–15. Le bébé voit les images 0–7 (tous ses sens)
et reçoit ses vrais gestes ; il IMAGINE les images 8–15. Un lecteur (instrument supervisé) entraîné sur des latents IMAGINÉS
(d'autres séquences) lit où le bébé imagine l'objet et sa main.
Dessin, pour chaque modèle : la vraie image ; vrai objet = cercle VERT, vraie main = carré CYAN ; à partir de l'image 8 :
objet imaginé = croix ROUGE, main imaginée = croix JAUNE, objet imaginé avec les gestes d'un AUTRE = croix MAGENTA,
« copie » (objet resté où il était à l'image 7) = point GRIS. Cadre jaune = imaginé.

  python av_pushgif.py --ckpts phase0_rD9_0e_60k.pt,phase0_r5_0e_60k.pt --out /content/drive/MyDrive/jepa_runs/pousse.gif
"""
import argparse, os, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from av_world0 import gen_world0
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, layout, VIEW, H
from av_plan0 import LocalReader

GREEN, CYAN, RED, YEL, MAG, GRAY = (60, 220, 60), (60, 220, 230), (255, 60, 60), (255, 220, 0), (230, 60, 230), (150, 150, 150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True); p.add_argument("--dir", default="/content/drive/MyDrive/jepa_runs")
    p.add_argument("--out", default="/content/drive/MyDrive/jepa_runs/pousse.gif"); p.add_argument("--n", type=int, default=4000)
    p.add_argument("--show", type=int, default=6); p.add_argument("--scale", type=int, default=8); p.add_argument("--blocks", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    names = [c.strip() for c in a.ckpts.split(",")]
    cks = [torch.load(os.path.join(a.dir, c), map_location=dev, weights_only=False) for c in names]
    cfg0 = cks[0]["cfg"]; T = int(cfg0.get("T", 16)); av_phase0.T = T; P = cfg0["P"]; nP = H // P; npf = nP * nP; nv = T * npf; v = VIEW["0e"]
    h = T // 2; nf = T - h
    wkw = {**cfg0.get("wkw", {}), "p_parent": 0.0, **({"blocks": 1.0} if a.blocks else {})}
    w = gen_world0(a.n, "0e", T, H, seed=4242, **wkw)
    keep = np.where(w["NOBJ"] == 1)[0]; n = len(keep)
    b = to_torch(to_np([{k: w[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}]))
    Y = torch.from_numpy(np.concatenate([w["POS"][keep, :, 0], w["HAND"][keep]], -1)).float()      # (n, T, 4) objet x,y, main x,y
    po_ = w["POS"][keep, :, 0]; inview = ((po_ > 0.06) & (po_ < 0.94)).all((1, 2))                      # l'objet reste DANS le champ (sinon on ne voit rien)
    moved = (np.linalg.norm(po_[:, T - 1] - po_[:, h - 1], axis=-1) * H > 3.0) & inview                 # l'objet a bougé de > 3 px pendant 8–15
    ntr = int(0.8 * n); show = np.where(moved[ntr:])[0][:a.show] + ntr
    print(f"{n} séquences à 1 objet, {int(moved.sum())} où l'objet est poussé (et reste dans le champ) ; montrées : {len(show)} | {time.time() - t0:.0f}s", flush=True)
    md, fr = layout(nv, npf); ci0 = torch.from_numpy(np.where(fr < h)[0]).to(dev); ti0 = torch.from_numpy(np.where((fr >= h) & (md == 0))[0]).to(dev)
    res = []
    for c, ck in zip(names, cks):
        cfg = ck["cfg"]; d = cfg["d"]; m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval()

        @torch.no_grad()
        def imag(idx, lie=False):
            out = []
            for i in range(0, len(idx), 50):
                sl = torch.from_numpy(idx[i:i + 50]); tok, cmd = to_tok0({k: x[sl] for k, x in b.items()}, ck["norm"], v, P, dev); B = len(tok)
                if lie: cmd = cmd.roll(1, 0)
                ci, ti = ci0.expand(B, -1), ti0.expand(B, -1)
                ctx = m.context(torch.gather(tok, 1, ci.unsqueeze(-1).expand(-1, -1, tok.size(-1))), ci)
                out.append(m.pred(ctx, ti, cmd, ci).float().view(B, nf, npf, d).cpu())
            return torch.cat(out)
        Ztr = imag(np.arange(ntr)).flatten(0, 1); Ytr = Y[:ntr, h:].flatten(0, 1); mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
        torch.manual_seed(0); rd = LocalReader(d, npf, 4, mu=mu, sd=sd).to(dev); opt = torch.optim.AdamW(rd.parameters(), 3e-4, weight_decay=0.05)
        for _ in range(a.steps):
            bi = torch.randint(0, len(Ztr), (256,)); l = F.mse_loss(rd(Ztr[bi].to(dev)), ((Ytr[bi] - mu) / sd).to(dev)); opt.zero_grad(); l.backward(); opt.step()
        rd.eval(); te = np.arange(ntr, n)
        with torch.no_grad():
            rp = lambda Z: (rd(Z.flatten(0, 1).to(dev)).cpu() * sd + mu).view(-1, nf, 4)
            pi, pl = rp(imag(te)), rp(imag(te, lie=True))
        yt = Y[ntr:, h:]; mv = torch.from_numpy(moved[ntr:])
        px = lambda pr, k: float(((pr - yt)[mv].view(-1, nf, 2, 2).norm(dim=-1)[..., k] * H).mean())
        cp = Y[ntr:, h - 1:h].expand(-1, nf, -1)
        print(f"  {c} : objet POUSSÉ imaginé {px(pi, 0):.2f} px (copie {px(cp, 0):.2f}, gestes d'un AUTRE {px(pl, 0):.2f}) | main {px(pi, 1):.2f} px "
              f"(copie {px(cp, 1):.2f}) | {time.time() - t0:.0f}s", flush=True)
        res.append((c.replace(".pt", "").replace("phase0_", ""), pi[show - ntr].numpy(), pl[show - ntr].numpy()))
        del m; torch.cuda.empty_cache() if dev == "cuda" else None
    # ---------- GIF : une colonne par modèle, même séquence
    s = a.scale; W1 = H * s; pad = 6; head = 30; frames = []; X = w["X"][keep]
    to_px = lambda q: (float(q[0]) * W1, float(q[1]) * W1)
    def cross(dr, q, col, r=6, wd=3):
        x, y = to_px(q); dr.line([x - r, y - r, x + r, y + r], fill=col, width=wd); dr.line([x - r, y + r, x + r, y - r], fill=col, width=wd)
    for j, i in enumerate(show):
        for t in range(T):
            img = Image.new("RGB", ((W1 + pad) * len(res) + pad, W1 + head + 22), (20, 20, 20)); dr = ImageDraw.Draw(img)
            for k, (nm, pi, pl) in enumerate(res):
                tile = Image.fromarray((X[i, t] * 255).astype(np.uint8)).resize((W1, W1), Image.NEAREST); dt = ImageDraw.Draw(tile)
                po, ph = Y[i, t, :2].numpy(), Y[i, t, 2:].numpy(); x, y = to_px(po); dt.ellipse([x - 9, y - 9, x + 9, y + 9], outline=GREEN, width=2)
                x, y = to_px(ph); dt.rectangle([x - 8, y - 8, x + 8, y + 8], outline=CYAN, width=2)
                if t >= h:
                    x, y = to_px(Y[i, h - 1, :2].numpy()); dt.ellipse([x - 3, y - 3, x + 3, y + 3], fill=GRAY)
                    cross(dt, pl[j, t - h, :2], MAG, 5, 2); cross(dt, pi[j, t - h, 2:], YEL); cross(dt, pi[j, t - h, :2], RED)
                    dt.rectangle([0, 0, W1 - 1, W1 - 1], outline=YEL, width=3)
                x0 = pad + k * (W1 + pad); img.paste(tile, (x0, head)); dr.text((x0, 4), nm, fill=(230, 230, 230))
                dr.text((x0, 16), f"séq {j + 1}  image {t}" + ("  IMAGINÉ" if t >= h else "  vu"), fill=(255, 210, 0) if t >= h else (170, 170, 170))
            dr.text((pad, W1 + head + 4), "vert=vrai objet  cyan=vraie main  rouge=objet imaginé  jaune=main imaginée  magenta=objet (gestes d'un autre)  gris=copie",
                    fill=(200, 200, 200))
            frames.append(img)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=400, loop=0)
    print(f"GIF : {a.out} ({len(frames)} images) | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
