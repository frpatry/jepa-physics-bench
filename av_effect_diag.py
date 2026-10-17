"""
DIAGNOSTIC DU VERROU « l'effet de MES gestes sur l'objet » (instrument seulement — le bébé n'apprend rien ici).
Question : l'information nécessaire est-elle DANS sa représentation ? Et son imagination s'en sert-elle ?
Cas étudiés : images t (monde à 1 objet, gestes AVEUGLES = contrôle sans fuite) où SA MAIN est PRÈS de l'objet ; à t+1 il le touche
ou le rate selon où va le geste. Cible = déplacement de l'objet entre t et t+1 (px).
Mêmes têtes apprises (80 % des séquences / 20 % test) qui devinent ce déplacement à partir de :
  A geste seul (repère bas)            B inertie : vitesse réelle de l'objet + geste
  C VRAI ÉTAT (main, objet, vitesse, forme) + geste (repère haut : la physique est apprenable)
  D PIXELS des images t-1, t + geste   E SA REPRÉSENTATION de l'image t (encodeur de contexte, images ≤ t) + geste
et son IMAGINATION : G la même tête, SANS le geste, sur (image t, image t+1) — t+1 réelle (repère) ou IMAGINÉE (ses gestes / ceux d'un autre) :
l'effet est-il DANS l'image imaginée ? (une tête apprise sur l'imaginé : les latents imaginés sont flous, un lecteur des latents réels s'y perd).
Lecture : E ≈ C ≫ A, B -> l'info est là, c'est le prédicteur qui ne s'en sert pas ; E ≈ B -> la représentation ne la porte pas.

  python av_effect_diag.py --ckpts phase0_rD12_0e_60k.pt,phase0_rD9_0e_60k.pt
"""
import argparse, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_world0 import gen_world0
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, layout, VIEW, H


class Head(nn.Module):
    """tête commune : par case (latent ou pixels) -> 16 nombres, mis bout à bout avec le geste -> déplacement (2)."""
    def __init__(s, din, ncell, ncmd=2, h=256):
        super().__init__(); s.cell = nn.Linear(din, 16) if ncell else None
        s.mlp = nn.Sequential(nn.Linear(16 * ncell + din * (ncell == 0) + ncmd, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2))
    def forward(s, x, c):
        x = s.cell(x.float()).flatten(1) if s.cell is not None else x.float()
        return s.mlp(torch.cat([x, c.float()], -1))


def fit(x, c, y, tr, te, dev, ncell, steps=6000):
    """apprend une tête sur tr (10 % gardés pour ARRÊTER AU BON MOMENT : pas de sur-apprentissage qui avantagerait une source),
    rend la prédiction sur te."""
    torch.manual_seed(0); din = x.shape[-1]; hd = Head(din, ncell).to(dev); opt = torch.optim.AdamW(hd.parameters(), 1e-3, weight_decay=0.05)
    sd = y[tr].std(0) + 1e-3; va, tr = tr[: len(tr) // 10], tr[len(tr) // 10:]
    pr = lambda ix: torch.cat([hd(x[ix[i:i + 512]].to(dev), c[ix[i:i + 512]].to(dev)).cpu() for i in range(0, len(ix), 512)]) * sd
    best, state = 1e9, None
    for it in range(steps):
        hd.train(); b = tr[torch.randint(0, len(tr), (256,))]
        l = F.mse_loss(hd(x[b].to(dev), c[b].to(dev)), (y[b] / sd).to(dev)); opt.zero_grad(); l.backward(); opt.step()
        if (it + 1) % 250 == 0:
            hd.eval()
            with torch.no_grad(): e = float(((pr(va) - y[va]) ** 2).mean())
            if e < best: best, state = e, {k: v.clone() for k, v in hd.state_dict().items()}
    hd.load_state_dict(state); hd.eval()
    with torch.no_grad(): return pr(te)


def score(p, y, hit):
    """R² global ; sur les CONTACTS (objet bougé > 1 px) : bonne direction (cos > 0.5) et amplitude ; sur les RATÉS : mouvement inventé."""
    r2 = float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    cs = F.cosine_similarity(p[hit], y[hit], dim=-1); amp = (p[hit].norm(dim=-1) / y[hit].norm(dim=-1)).median()
    miss = ~hit & (y.norm(dim=-1) < 0.3)
    return f"R² {r2:+.2f} | contacts : bonne direction {float((cs > 0.5).float().mean()) * 100:3.0f} %, cos {float(cs.mean()):+.2f}, amplitude {float(amp):.2f} " \
           f"| ratés : mouvement inventé {float(p[miss].norm(dim=-1).mean()):.2f} px"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True); p.add_argument("--dir", default="/content/drive/MyDrive/jepa_runs")
    p.add_argument("--n", type=int, default=12000); p.add_argument("--near", type=float, default=0.25)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    names = [c.strip() for c in a.ckpts.split(",")]
    cks = [torch.load(os.path.join(a.dir, c), map_location=dev, weights_only=False) for c in names]
    cfg0 = cks[0]["cfg"]; T = int(cfg0.get("T", 16)); av_phase0.T = T; P = cfg0["P"]; nP = H // P; npf = nP * nP; nv = T * npf; v = VIEW["0e"]
    wkw = {**cfg0.get("wkw", {}), "p_parent": 0.0, "blind": True, "calm": 0.3}; wkw.pop("circ", None); wkw.pop("pause", None)
    w = gen_world0(a.n, "0e", T, H, seed=5151, **wkw)
    keep = np.where(w["NOBJ"] == 1)[0]; n = len(keep)
    Po, Hd = w["POS"][keep, :, 0], w["HAND"][keep]; TS = w["TSRC"][keep]; CMD = w["CMD"][keep]; RH = w["RH"][keep]; SH = w["SHAPE"][keep, 0]
    inv = ((Po > 0.06) & (Po < 0.94)).all(-1)
    ev = [(i, t) for i in range(n) for t in range(2, T - 1)
          if inv[i, t] and inv[i, t + 1] and np.linalg.norm(Po[i, t] - Hd[i, t]) < a.near]
    ev = np.array(ev); I, Tt = ev[:, 0], ev[:, 1]
    y = torch.from_numpy((Po[I, Tt + 1] - Po[I, Tt]) * H).float()
    hit = torch.from_numpy(TS[I, Tt + 1] == 1) & (y.norm(dim=-1) > 1.0)          # il le TOUCHE et l'objet bouge > 1 px
    c = torch.from_numpy(CMD[I, Tt + 1] / 0.05).float()                      # le geste qui mène de t à t+1
    tr_seq = I < int(0.8 * n); tr, te = torch.from_numpy(np.where(tr_seq)[0]), torch.from_numpy(np.where(~tr_seq)[0])
    tr = tr[np.argsort(I[tr.numpy()] % 10, kind="stable")]                       # 10 % des SÉQUENCES d'entraînement en tête = validation
    print(f"{n} séquences à 1 objet, {len(ev)} cas « main près de l'objet » (test {len(te)}, dont contacts qui le font bouger : "
          f"{int(hit[te].sum())}) ; déplacement moyen au contact {float(y[hit].norm(dim=-1).mean()):.2f} px | {time.time() - t0:.0f}s", flush=True)
    yt, ht = y[te], hit[te]
    vel = torch.from_numpy((Po[I, Tt] - Po[I, Tt - 1]) * H).float()
    sh = F.one_hot(torch.from_numpy(SH[I]).clamp_min(0), 4).float()
    truth = torch.cat([torch.from_numpy(Hd[I, Tt]).float(), torch.from_numpy(Po[I, Tt]).float(), torch.from_numpy(Po[I, Tt] - Hd[I, Tt]).float() * 4,
                       vel, torch.from_numpy(RH[I])[:, None].float() * 10, sh], -1)
    zero = torch.zeros(len(ev), 1)
    print(f"  A geste seul               : {score(fit(zero, c, y, tr, te, dev, 0), yt, ht)}", flush=True)
    print(f"  B inertie + geste          : {score(fit(vel, c, y, tr, te, dev, 0), yt, ht)}", flush=True)
    print(f"  C VRAI ÉTAT + geste        : {score(fit(truth, c, y, tr, te, dev, 0), yt, ht)}", flush=True)
    b = to_torch(to_np([{k: w[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}]))
    md, fr = layout(nv, npf)
    with torch.no_grad():                                                    # pixels des cases (images t-1 et t), vue du bébé
        pix = []
        for s_ in range(0, len(ev), 200):
            sl = torch.from_numpy(I[s_:s_ + 200]); tok, _ = to_tok0({k: x[sl] for k, x in b.items()}, cks[0]["norm"], v, P, dev)
            vis = tok[:, :nv, :P * P * 3].view(len(sl), T, npf, -1); tt = torch.from_numpy(Tt[s_:s_ + 200]).to(dev)
            pix.append(torch.cat([vis[torch.arange(len(sl)), tt - 1], vis[torch.arange(len(sl)), tt]], -1).half().cpu())
        pix = torch.cat(pix)
    print(f"  D PIXELS (t-1, t) + geste  : {score(fit(pix, c, y, tr, te, dev, npf), yt, ht)} | {time.time() - t0:.0f}s", flush=True)
    rel = torch.from_numpy((Po[I, Tt] - Hd[I, Tt]) * H).float()
    r2 = lambda p_, y_: float(1 - ((p_ - y_) ** 2).sum() / ((y_ - y_.mean(0)) ** 2).sum())
    z2 = torch.zeros(len(ev), 2)
    probe = lambda X_, nc: f"élan de l'objet R² {r2(fit(X_, z2, vel, tr, te, dev, nc), vel[te]):+.2f} | objet − main R² {r2(fit(X_, z2, rel, tr, te, dev, nc), rel[te]):+.2f}"
    print(f"    sondes PIXELS              : {probe(pix, npf)}", flush=True)
    del pix
    for nm, ck in zip(names, cks):
        cfg = ck["cfg"]; d = cfg["d"]; m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"]); m.eval()
        # (avec cases résumé, le contexte rend [cases | résumés] : les indices pt des cases restent valables)
        Zp = torch.zeros(len(ev), npf, d, dtype=torch.half); Zt = torch.zeros_like(Zp); Zn = torch.zeros_like(Zp); Zi = torch.zeros_like(Zp); Zl = torch.zeros_like(Zp); Z0 = torch.zeros_like(Zp)
        with torch.no_grad():
            for t in range(2, T - 1):
                k_ = np.where(Tt == t)[0]
                if not len(k_): continue
                ci0 = torch.from_numpy(np.where(fr <= t)[0]).to(dev); cn0 = torch.from_numpy(np.where(fr <= t + 1)[0]).to(dev)
                ti0 = torch.from_numpy(np.where((fr == t + 1) & (md == 0))[0]).to(dev)
                at = lambda c0, f_: torch.from_numpy(np.searchsorted(c0.cpu().numpy(), np.where((fr == f_) & (md == 0))[0])).to(dev)
                pp, pt, pn = at(ci0, t - 1), at(ci0, t), at(cn0, t + 1)
                for s_ in range(0, len(k_), 100):
                    kk = k_[s_:s_ + 100]; sl = torch.from_numpy(I[kk]); tok, cmd = to_tok0({k: x[sl] for k, x in b.items()}, ck["norm"], v, P, dev); B = len(tok)
                    ci, cn, ti = ci0.expand(B, -1), cn0.expand(B, -1), ti0.expand(B, -1)
                    g = lambda ix: torch.gather(tok, 1, ix.unsqueeze(-1).expand(-1, -1, tok.size(-1)))
                    ctx = m.context(g(ci), ci); Zp[kk] = ctx[:, pp].half().cpu(); Zt[kk] = ctx[:, pt].half().cpu()
                    Zn[kk] = m.context(g(cn), cn)[:, pn].half().cpu()                                    # image t+1 RÉELLE (contexte ≤ t+1)
                    Zi[kk] = m.pred(ctx, ti, cmd, ci).float().view(B, npf, d).half().cpu()                # image t+1 IMAGINÉE, ses gestes
                    cl = cmd.clone(); cl[:, t + 1] = cmd.roll(1, 0)[:, t + 1]                            # remarque user : le témoin ne doit pas être
                    c0 = cmd.clone(); c0[:, t + 1] = 0                                                     # biaisé -> on ne change QUE le geste t -> t+1
                    Zl[kk] = m.pred(ctx, ti, cl, ci).float().view(B, npf, d).half().cpu()                 # ... geste t->t+1 d'un AUTRE (passé intact)
                    Z0[kk] = m.pred(ctx, ti, c0, ci).float().view(B, npf, d).half().cpu()                 # ... « j'arrête » (aucun geste t->t+1)
        cat = lambda *Z: torch.cat(Z, -1)
        print(f"  {nm}", flush=True)
        print(f"    E SA REPRÉSENTATION (t-1, t) + geste : {score(fit(cat(Zp, Zt), c, y, tr, te, dev, npf), yt, ht)} | {time.time() - t0:.0f}s", flush=True)
        print(f"    sondes SA REPRÉSENTATION (t-1, t)    : {probe(cat(Zp, Zt), npf)}", flush=True)
        # G : l'EFFET est-il dans l'image t+1 ? même tête, sans le geste, sur (t, t+1 réelle) / (t, t+1 imaginée avec ses gestes / ceux d'un autre)
        for lab, Z in (("t+1 RÉELLE", Zn), ("t+1 IMAGINÉE, ses gestes", Zi), ("t+1 IMAGINÉE, geste d'un AUTRE", Zl), ("t+1 IMAGINÉE, « j'arrête »", Z0)):
            print(f"    G (t, {lab:31s}) : {score(fit(cat(Zt, Z), z2, y, tr, te, dev, npf), yt, ht)}", flush=True)
        del m; torch.cuda.empty_cache() if dev == "cuda" else None
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
