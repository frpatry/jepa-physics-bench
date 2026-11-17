"""
OÙ L'ŒIL SITUE-T-IL LES CHOSES ? — GIF de diagnostic (17 nov. 2026 ; bébé GELÉ, aucun entraînement du bébé)
Question user : visuellement, sa représentation est-elle bonne ? av_seegif.py ne montre que « dans quelles cases il y a un objet » (pâtés de cases).
Ici un lecteur (instrument, le même que celui du planificateur : LocalReader, appris sur 80 % des scènes à un objet) lit dans les latents de l'œil-cible le CENTRE de
l'objet et celui de la main ; sur des scènes jamais vues par le lecteur, on dessine sur la vraie image :
  croix VERTE = vrai centre de l'objet, rond JAUNE = là où l'œil le situe | croix BLEUE = vrai centre de la main, rond ROUGE = là où l'œil la situe.
  python av_posgif.py --ckpt bebe3c.pt --out pos.gif
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, H
from av_world0 import gen_world0, VIEW
from av_plan0 import LocalReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--out", default="pos.gif"); p.add_argument("--n", type=int, default=3000); p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--show", type=int, default=6); p.add_argument("--scale", type=int, default=10)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]; av_phase0.T = T
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; ix = torch.arange(nv, device=dev)
    w = gen_world0(a.n, "0e", T, H, seed=778, **{**cfg.get("wkw", {}), "p_parent": 0.0}); keep = np.where(w["NOBJ"] == 1)[0]; n = len(keep)
    b = to_torch(to_np([{k: w[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}])); X = w["X"][keep]
    Y = torch.from_numpy(np.concatenate([w["POS"][keep, :, 0], w["HAND"][keep]], -1)).float()                    # (n, T, 4) : objet x y, main x y (dans [0, 1])
    Z = []
    with torch.no_grad():
        for i in range(0, n, 50):
            tok, _ = to_tok0({k: x[i:i + 50] for k, x in b.items()}, st, v, P_, dev); Z.append(F.layer_norm(tgt(tok[:, :nv], ix.expand(len(tok), -1)).float(), (d,)).view(-1, T, npf, d).half().cpu())
    Z = torch.cat(Z); ntr = int(0.8 * n); Ztr, Ytr = Z[:ntr].flatten(0, 1), Y[:ntr].flatten(0, 1); mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6; torch.manual_seed(0)
    ro = LocalReader(d, npf, 4, mu=mu, sd=sd).to(dev); opt = torch.optim.AdamW(ro.parameters(), 3e-4, weight_decay=0.05); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for _ in range(a.steps):
        bi = torch.randint(0, len(Ztr), (256,)); l = F.mse_loss(ro(Ztr[bi].to(dev).float()), ((Ytr[bi] - mu) / sd).to(dev)); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    ro.eval()
    with torch.no_grad(): Pt = torch.cat([(ro(Z[ntr:].flatten(0, 1)[i:i + 512].to(dev).float()).cpu() * sd + mu) for i in range(0, (n - ntr) * T, 512)]).view(n - ntr, T, 4)
    Yt = Y[ntr:]; E = (Pt - Yt).view(n - ntr, T, 2, 2).norm(dim=-1) * H                                           # (scènes, T, [objet, main]) en px
    q = lambda e: f"moyenne {float(e.mean()):.2f} px, médiane {float(e.median()):.2f}, 9 fois sur 10 sous {float(e.flatten().quantile(0.9)):.2f}, pire {float(e.max()):.1f}"
    mv = (Yt[:, 1:, :2] - Yt[:, :-1, :2]).norm(dim=-1) * H > 0.3; mv = torch.cat([mv[:, :1], mv], 1)              # l'objet bouge à cette image
    hd = (Yt[:, :, :2] - Yt[:, :, 2:]).norm(dim=-1) * H                                                           # distance main–objet (centres)
    log(f"{ck['state']['it']} pas | {n} scènes à un objet, lecteur appris sur {ntr}, lu sur {n - ntr} jamais vues | {time.time() - t0:.0f}s")
    log(f"  OBJET : {q(E[..., 0])}\n  MAIN  : {q(E[..., 1])}")
    log(f"  objet immobile {float(E[..., 0][~mv].mean()):.2f} px | objet qui bouge {float(E[..., 0][mv].mean()):.2f} px (n={int(mv.sum())}) | main à moins de 6 px de l'objet {float(E[..., 0][hd < 6].mean()):.2f} px | main loin {float(E[..., 0][hd >= 10].mean()):.2f} px")
    dl = ((Pt[:, 1:, :2] - Pt[:, :-1, :2]) * H); dt = ((Yt[:, 1:, :2] - Yt[:, :-1, :2]) * H); big = dt.norm(dim=-1) > 0.5
    log(f"  DÉPLACEMENT de l'objet d'une image à la suivante (quand il bouge de > 0.5 px, n={int(big.sum())}) : vrai {float(dt[big].norm(dim=-1).mean()):.2f} px, lu {float(dl[big].norm(dim=-1).mean()):.2f} px, "
        f"direction (cos) {float(F.cosine_similarity(dl[big], dt[big], dim=-1).mean()):+.2f} | quand il est immobile : déplacement lu {float(dl[~big].norm(dim=-1).mean()):.2f} px (tremblement de la lecture)")
    # ---------- GIF : des scènes où l'objet bouge d'abord (plus parlantes), puis quelconques
    order = torch.argsort(-(dt.norm(dim=-1).sum(1))); show = [int(x) for x in order[:a.show // 2]] + [int(x) for x in range(n - ntr) if x not in set(order[:a.show // 2].tolist())][:a.show - a.show // 2]
    S = a.scale; W = H * S; cols = min(3, len(show)); rows = (len(show) + cols - 1) // cols; frames = []
    def cross(dr, x, y, c, r=6): dr.line([x - r, y, x + r, y], fill=c, width=2); dr.line([x, y - r, x, y + r], fill=c, width=2)
    def ring(dr, x, y, c, r=5): dr.ellipse([x - r, y - r, x + r, y + r], outline=c, width=2)
    for t in range(T):
        im = Image.new("RGB", (cols * (W + 6), rows * (W + 22) + 16), (20, 20, 20)); dr = ImageDraw.Draw(im)
        dr.text((4, 2), f"image {t} | croix = VRAI centre (vert objet, bleu main) | rond = où l'oeil le situe (jaune objet, rouge main)", fill=(230, 230, 230))
        for k, s_ in enumerate(show):
            x0, y0 = (k % cols) * (W + 6), 16 + (k // cols) * (W + 22)
            im.paste(Image.fromarray((X[ntr + s_, t] * 255).astype(np.uint8)).resize((W, W), Image.NEAREST), (x0, y0 + 16))
            dr.text((x0 + 2, y0 + 2), f"objet {float(E[s_, t, 0]):.1f} px | main {float(E[s_, t, 1]):.1f} px", fill=(200, 200, 200))
            yt, pt = Yt[s_, t] * W, Pt[s_, t] * W
            cross(dr, x0 + float(yt[0]), y0 + 16 + float(yt[1]), (60, 255, 60)); ring(dr, x0 + float(pt[0]), y0 + 16 + float(pt[1]), (255, 230, 40))
            cross(dr, x0 + float(yt[2]), y0 + 16 + float(yt[3]), (60, 160, 255)); ring(dr, x0 + float(pt[2]), y0 + 16 + float(pt[3]), (255, 60, 60))
        frames.append(im)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=350, loop=0); log(f"GIF : {a.out} ({T} images, {len(show)} scènes) | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
