"""
SUIVRE LES BONS GESTES, SANS DÉCISION (idée user, 17 nov. 2026 ; bébé GELÉ, aucun entraînement)
« En sachant exactement quel geste doit être utilisé, interroge les latents pour ce geste, fais le geste, et ainsi de suite. Aucune décision du meilleur geste. »
Un oracle (instrument : il connaît l'état et la physique) joue Push-T : il contourne le T, se place derrière, pousse. À CHAQUE geste de sa partie on interroge
l'imagination par objet EXACTEMENT comme le planificateur (fenêtre glissante, image de départ = l'œil sur le passé seulement, mêmes entrées) avec LES gestes qui vont
vraiment être faits (1 pas, et les 8 suivants), puis on compare à ce qui se passe vraiment :
  - déplacement imaginé de l'objet et de la main (champ de déplacement du module sur leurs cases, sans lecteur) contre le vrai ;
  - erreur dans les latents sur les cases de l'objet : recopier / recopier + horloge / imagination, contre le vrai latent futur (vu de la même façon) ;
  - deux façons de l'interroger : sens du 1er pas donnés (comme au jeu) ou non (comme à l'entraînement avant l'étape des sens).
GIF : la vraie image 8 gestes plus tard, croix = vrai centre du T, ronds = centre de départ + déplacement imaginé (jaune : sens donnés, orange : sans).
  python av_follow_diag.py --ckpt bebe3c.pt --out suivre.gif
"""
import argparse, copy, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
import av_phase0, av_plan0
from av_phase0 import baby_from_cfg, to_tok0, H
from av_world0 import VIEW
from av_plan0 import Push0, oracle, VMAX


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--out", default="suivre.gif"); p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=150); p.add_argument("--h", type=int, default=8); p.add_argument("--show", type=int, default=3)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time(); log = lambda s_: print(s_, flush=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]; av_phase0.T = T; av_plan0.T = T
    m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval(); tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); om = m.oi.eval()
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; wk = cfg.get("wkw", {}); Hh = a.h; LN = lambda u: F.layer_norm(u.float(), (d,)); ic, ia = nv + T, nv + 2 * T
    cellfrac = lambda M: M.reshape(nP, P_, nP, P_).mean((1, 3)).reshape(npf)                       # part de chaque case (instrument : couleurs des pixels)
    R = []; films = []; log(f"SUIVRE LES BONS GESTES : bébé au pas {ck['state']['it']} | {a.episodes} parties de l'oracle, {a.steps} gestes, horizon {Hh}")
    for ep in range(a.episodes):
        env = Push0(ep, task="pusht", damp=wk.get("damp", 0.0), echo=wk.get("echo", 0.0), shape_phys=bool(wk.get("shape_phys", False)), L=a.steps + 2 + Hh + T, obj_scale=wk.get("obj_scale", 1.0))
        c0 = env.coverage(); cov = [c0]
        for _ in range(a.steps): env.step(oracle(env)); cov.append(env.coverage())
        te = env.t; mvd = float(np.mean(np.linalg.norm(np.diff(env.POS[:te + 1], axis=0), axis=1) * H > 0.05)); ev_all = list(env.ev); film = []
        log(f"  partie {ep} : couverture {c0:.0%} -> {cov[-1]:.0%} (max {max(cov):.0%}) | gestes où le T bouge {mvd:.0%} | {time.time() - t0:.0f}s")
        for tau in range(T - 1 - Hh, te - Hh):
            s0 = max(0, tau + Hh - (T - 1)); t = tau - s0
            env.t = tau; env.ev = [e for e in ev_all if e[0] <= tau]; b = env.batch(s0); env.t = te; env.ev = ev_all            # ce qu'il a vécu jusqu'à tau (comme au jeu)
            for k_ in ("TOUCH", "PROP", "CMD"): b[k_] = b[k_].clone(); b[k_][:, t + 1:] = 0
            with torch.no_grad():
                tok, cmd = to_tok0(b, st, v, P_, dev); cmd = cmd.clone(); cmd[:, t + 1:] = 0; cmd[0, t + 1:t + 1 + Hh] = torch.from_numpy(env.CMD[tau + 1:tau + 1 + Hh]).to(dev) / 0.05
                seen = lambda q: LN(tgt(tok[:, :(q + 1) * npf], torch.arange((q + 1) * npf, device=dev)[None]))[:, -npf:]          # l'image q vue avec le passé seulement
                zt = seen(t); zf = LN(tgt(tok, torch.arange(tok.size(1), device=dev)[None]))[0]; sn = torch.stack([zf[nv + t], zf[ic + t], zf[ia + t]])[None]
                X0 = env.X[tau]; wh0 = cellfrac((X0.min(-1) > 0.85).astype(np.float32)); co0 = cellfrac(((X0.max(-1) > 0.15) & (X0.min(-1) <= 0.85)).astype(np.float32))
                row = dict(ep=ep, tau=tau)
                for k in (1, Hh):
                    Xk = env.X[tau + k]; whk = cellfrac((Xk.min(-1) > 0.85).astype(np.float32)); cok = cellfrac(((Xk.max(-1) > 0.15) & (Xk.min(-1) <= 0.85)).astype(np.float32))
                    wo = torch.from_numpy(co0 * (wh0 < 0.1)); whd = torch.from_numpy(wh0 * (wh0 > 0.15)); evm = torch.from_numpy((((co0 >= 0.1) | (cok >= 0.1)) & (wh0 < 0.02) & (whk < 0.02)).astype(np.float32))
                    zr = seen(t + k)[0].cpu(); row[f"do{k}"] = (env.POS[tau + k] - env.POS[tau]) * H; row[f"dh{k}"] = (env.HAND[tau + k] - env.HAND[tau]) * H
                    avg = lambda x, w_: (x * w_.unsqueeze(-1)).sum(0) / w_.sum().clamp_min(1e-6) if x.dim() == 2 else float((x * w_).sum() / w_.sum().clamp_min(1e-6))
                    row[f"okc{k}"] = float(evm.sum()) > 0; row[f"oko{k}"] = float(wo.sum()) > 0.3; row[f"okh{k}"] = float(whd.sum()) > 0.15
                    zc = zt[0].cpu(); zk = (zt[0] - om.clock[t] + om.clock[t + k]).cpu(); row[f"ec{k}"] = avg((zc - zr).abs().mean(-1), evm); row[f"es{k}"] = avg((zk - zr).abs().mean(-1), evm)
                    for nm, s_ in (("S", sn), ("N", None)):                  # S : sens du 1er pas donnés (jeu) | N : non (entraînement avant l'étape des sens)
                        pl, f, o, _ = om.roll(zt, cmd, tok[:, ia:, :4], tok[:, ic:ia, :8], t, k, sens=s_); f = f[0].cpu() * P_
                        row[f"io{k}{nm}"] = avg(f, wo).numpy(); row[f"ih{k}{nm}"] = avg(f, whd).numpy(); row[f"em{k}{nm}"] = avg((pl[-1][0].cpu() - zr).abs().mean(-1), evm)
                R.append(row)
                if ep < a.show and (tau - (T - 1 - Hh)) % 2 == 0: film.append((env.X[tau + Hh].copy(), env.POS[tau].copy(), env.POS[tau + Hh].copy(), row[f"io{Hh}S"].copy(), row[f"io{Hh}N"].copy(), tau, cov[tau + Hh]))
        films.append(film)
    # ---------- mesures
    st_ = lambda key: np.stack([r[key] for r in R]); n_ = lambda x: np.linalg.norm(x, axis=-1)
    def cosm(a_, b_): return float(np.mean((a_ * b_).sum(-1) / (n_(a_) * n_(b_) + 1e-9)))
    for k in (1, Hh):
        do, dh = st_(f"do{k}"), st_(f"dh{k}"); oko, okh, okc = st_(f"oko{k}"), st_(f"okh{k}"), st_(f"okc{k}"); dn, hn = n_(do), n_(dh)
        log(f"\n=== {k} geste(s) plus tard | {len(R)} interrogations | le T bouge vraiment (> 1 px) dans {np.mean(dn > 1):.0%} des cas, reste immobile (< 0.1 px) dans {np.mean(dn < 0.1):.0%}")
        for nm, lab in (("S", "sens du 1er pas DONNÉS (comme au jeu)"), ("N", "sens du 1er pas NON donnés (comme à l'entraînement)")):
            io, ih, em = st_(f"io{k}{nm}"), st_(f"ih{k}{nm}"), st_(f"em{k}{nm}"); ec, es = st_(f"ec{k}"), st_(f"es{k}")
            mv = (dn > 1) & oko; sl = (dn < 0.1) & oko; hm = (hn > 0.5) & okh
            f3 = lambda q: (f"recopier {ec[q & okc].mean():.3f} | recopier + horloge {es[q & okc].mean():.3f} | IMAGINATION {em[q & okc].mean():.3f} ({(em[q & okc].mean() / es[q & okc].mean() - 1) * 100:+.0f} %)" if (q & okc).sum() >= 5 else "trop peu")
            log(f"  -- {lab}")
            if mv.sum() >= 5: log(f"     T qui BOUGE (n={int(mv.sum())}) : vrai {dn[mv].mean():.2f} px, imaginé {n_(io[mv]).mean():.2f} px ({n_(io[mv]).mean() / dn[mv].mean():.0%}), direction (cos) {cosm(io[mv], do[mv]):+.2f} | latents : {f3(mv)}")
            log(f"     T IMMOBILE (n={int(sl.sum())}) : déplacement imaginé {n_(io[sl]).mean():.2f} px | latents : {f3(sl)}")
            log(f"     MAIN (n={int(hm.sum())}) : vrai {hn[hm].mean():.2f} px, imaginé {n_(ih[hm]).mean():.2f} px ({n_(ih[hm]).mean() / hn[hm].mean():.0%}), direction (cos) {cosm(ih[hm], dh[hm]):+.2f}")
    # ---------- GIF
    S = 8; W = H * S; nfr = max(len(f_) for f_ in films[:a.show]); frames = []
    def cross(dr, x, y, c, r=6): dr.line([x - r, y, x + r, y], fill=c, width=2); dr.line([x, y - r, x, y + r], fill=c, width=2)
    def ring(dr, x, y, c, r=5): dr.ellipse([x - r, y - r, x + r, y + r], outline=c, width=2)
    for i in range(nfr):
        im = Image.new("RGB", (a.show * (W + 6), W + 48), (20, 20, 20)); dr = ImageDraw.Draw(im)
        dr.text((4, 2), f"image vraie {Hh} gestes plus tard | croix verte = vrai centre du T | point gris = son centre {Hh} gestes avant", fill=(230, 230, 230))
        dr.text((4, 14), "rond = centre de depart + deplacement IMAGINE pour ces gestes (jaune : sens donnes, orange : sans)", fill=(230, 230, 230))
        for e, film in enumerate(films[:a.show]):
            X8, p0, p8, fS, fN, tau, cv = film[min(i, len(film) - 1)]; x0 = e * (W + 6)
            im.paste(Image.fromarray((X8 * 255).astype(np.uint8)).resize((W, W), Image.NEAREST), (x0, 48)); dr.text((x0 + 2, 32), f"partie {e} | geste {tau} | couverture {cv:.0%}", fill=(200, 200, 200))
            dr.ellipse([x0 + p0[0] * W - 2, 48 + p0[1] * W - 2, x0 + p0[0] * W + 2, 48 + p0[1] * W + 2], fill=(150, 150, 150)); cross(dr, x0 + p8[0] * W, 48 + p8[1] * W, (60, 255, 60))
            ring(dr, x0 + p0[0] * W + fS[0] * S, 48 + p0[1] * W + fS[1] * S, (255, 230, 40)); ring(dr, x0 + p0[0] * W + fN[0] * S, 48 + p0[1] * W + fN[1] * S, (255, 140, 30), r=7)
        frames.append(im)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=250, loop=0); log(f"\nGIF : {a.out} ({nfr} images) | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
