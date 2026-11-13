"""
IMAGINATION PAR OBJET, PUREMENT LATENTE (13 nov. 2026 ; direction choisie par le user) — petit module posé sur l'ŒIL GELÉ d'un bébé (D17s).
Constats : son imagination ne déplace jamais un objet (av_plan0 --obj_diag 2) ; déplacer ses VRAIS latents du vrai déplacement fait mieux que recopier
dans 100 % des cas (av_transport_diag) ; ses latents portent une HORLOGE (même changement d'une scène à l'autre) et un objet au repos tremble de 0.15
alors qu'une poussée d'un geste n'ajoute que +0.03 (av_floor_diag) : le déplacement ne se lit pas dans une différence de latents, il faut le TRANSPORT.
Le module (aucune cible tirée des pixels, aucune étiquette — seule perte : les latents imaginés doivent ressembler aux vrais latents de t+k) :
  1 regroupe : une projection (partie de son sens « bougent ensemble », co_proj) dit quelles cases vont ensemble ; un petit transformeur lit la scène
    (latents de la dernière image vue + ses gestes à venir + sa main sentie + son toucher) et propose un déplacement par case, MOYENNÉ PAR GROUPE ;
  2 dit ce qui est « chose » ou « fond » (une chose qui arrive sur du fond le cache) ;
  3 TRANSPORTE les latents (dé-datés) de chaque case de son déplacement, remplit les cases quittées avec du fond, puis remet la date (horloge apprise).
Sans déplacement il ne peut que recopier + dater : la SEULE façon de faire baisser l'erreur sur un objet poussé est de trouver son déplacement.
INSTRUMENTS (mesure seulement) : positions et masques du monde — déplacement imaginé de l'objet contre le vrai, objet jamais touché (doit valoir 0),
main ; « chose » sur objet / fond / main ; erreur latente : recopier | recopier + horloge | module.
  python av_objimag.py --ckpt /content/drive/MyDrive/jepa_runs/phase0_rD17s_0e_270k.pt --out /content/drive/MyDrive/jepa_runs/objimag_D17s.pt
"""
import argparse, copy, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, Stream, H
from av_world0 import gen_world0, VIEW
from av_judge import auc


class ObjImag(nn.Module):
    def __init__(s, d, nP, T, h=192, nl=2, beta=6.0, co=None):
        super().__init__(); s.nP, s.npf, s.beta = nP, nP * nP, beta
        s.inp = nn.Linear(d, h); s.pos = nn.Parameter(torch.randn(1, s.npf, h) * 0.02); s.cond = nn.Sequential(nn.Linear(16, h), nn.GELU(), nn.Linear(h, h)); s.kemb = nn.Embedding(16, h)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, batch_first=True, dropout=0.0, activation="gelu"), nl)
        s.flow = nn.Linear(h, 2); nn.init.zeros_(s.flow.weight); nn.init.zeros_(s.flow.bias); s.objh = nn.Linear(h, 1)
        s.proj = nn.Linear(d, 64); s.ps = nn.Parameter(torch.tensor(math.log(5.0)))
        if co is not None: s.proj.load_state_dict(co)                    # part de son sens du groupe (mouvement commun)
        s.clock = nn.Parameter(torch.zeros(T, s.npf, d)); s.fill = nn.Parameter(torch.zeros(s.npf, d))
        k0 = torch.zeros(3, 3); k0[1, 1] = 1.0; s.smooth = nn.Parameter(k0); s.gain = nn.Parameter(torch.ones(d))   # LISSAGE appris SANS déplacement (essai court : sans lui il décalait
        #   toute l'image d'une demi-case pour « flouter » ses latents et gagner en erreur — le flou doit être gratuit, pas acheté par un faux déplacement)
        ys, xs = torch.meshgrid(torch.arange(nP), torch.arange(nP), indexing="ij"); s.register_buffer("xy", torch.stack([xs, ys], -1).reshape(s.npf, 2).float())

    def splat(s, C, f, w):
        """C (B, npf, d) contenu, f (B, npf, 2) déplacement en cases, w (B, npf) priorité -> contenu transporté (B, npf, d), masse reçue (B, npf)."""
        B, n, d = C.shape; p = s.xy[None] + f; p0 = p.floor(); fr = p - p0; num = C.new_zeros(B, n, d); den = C.new_zeros(B, n)
        for dx in (0, 1):
            for dy in (0, 1):
                q = p0 + torch.tensor([dx, dy], device=C.device); k = (fr[..., 0] if dx else 1 - fr[..., 0]) * (fr[..., 1] if dy else 1 - fr[..., 1])
                ok = ((q >= 0) & (q < s.nP)).all(-1); idx = (q[..., 1].clamp(0, s.nP - 1) * s.nP + q[..., 0].clamp(0, s.nP - 1)).long(); kw = k * w * ok
                den = den.scatter_add(1, idx, kw); num = num.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, d), kw.unsqueeze(-1) * C)
        return num, den

    def forward(s, Zt, cond, t, k, zero_flow=False):
        """Zt (B, npf, d) vrais latents de la dernière image vue ; cond (B, 16) gestes à venir + main sentie + toucher ; -> latents imaginés de t+k, déplacement, « chose »."""
        hf = s.tr(s.inp(Zt) + s.pos + s.cond(cond).unsqueeze(1) + s.kemb(torch.full((len(Zt),), k, device=Zt.device)).unsqueeze(1))
        r = s.flow(hf); o = torch.sigmoid(s.objh(hf)).squeeze(-1)
        u = F.normalize(s.proj(Zt), dim=-1); A = torch.softmax(s.ps.exp() * (u @ u.transpose(1, 2)), -1); f = A @ r   # un déplacement PAR GROUPE (moyenne sur les cases qui vont ensemble)
        if zero_flow: f = torch.zeros_like(f)
        C = Zt - s.clock[t]; B_, n_, d_ = C.shape
        C = F.conv2d(C.transpose(1, 2).reshape(B_ * d_, 1, s.nP, s.nP), s.smooth[None, None], padding=1).reshape(B_, d_, n_).transpose(1, 2) * s.gain
        num, den = s.splat(C, f, torch.exp(s.beta * o))                                              # contenu dé-daté, transporté ; une chose cache le fond
        bgw = (1 - o).unsqueeze(-1); fill = (bgw * C).sum(1, keepdim=True) / bgw.sum(1, keepdim=True).clamp_min(1e-3) + s.fill    # case quittée -> du fond
        return (num + 0.05 * fill) / (den.unsqueeze(-1) + 0.05) + s.clock[t + k], f, o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--out", type=str, default="/content/objimag.pt"); p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--kmax", type=int, default=4); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--contact", type=float, default=0.75, help="part du lot prise parmi les séquences où il a senti un contact"); p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--still_w", type=float, default=0.02, help="a priori « les choses restent en place » : petit coût sur le déplacement moyen (en cases)")
    p.add_argument("--eval_every", type=int, default=2000); p.add_argument("--seed", type=int, default=0); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    baby = baby_from_cfg(cfg, dev); baby.load_state_dict(ck["m"]); tgt = copy.deepcopy(baby.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); av_phase0.T = T
    for q in tgt.parameters(): q.requires_grad_(False)
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; ix = torch.arange(nv, device=dev)
    co = {"weight": ck["m"]["co_proj.weight"], "bias": ck["m"]["co_proj.bias"]} if "co_proj.weight" in ck["m"] else None
    m = ObjImag(d, nP, T, co=co).to(dev); opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    wkw = {k: x for k, x in cfg.get("wkw", {}).items() if k not in ("nohand", "same_col", "circ")}; sc = float(wkw.get("obj_scale", 1.0))
    log(f"IMAGINATION PAR OBJET, purement latente (œil GELÉ, pas {ck['state']['it']}) : {sum(q.numel() for q in m.parameters()) / 1e6:.2f} M de paramètres, {a.steps} pas, lots de {a.bs}, horizons 1..{a.kmax}")

    @torch.no_grad()
    def latents(b):                              # lot -> vrais latents (B, T, npf, d), condition par instant
        tok, cmd = to_tok0(b, st, v, P_, dev); B = len(tok)
        return F.layer_norm(tgt(tok[:, :nv], ix.expand(B, -1)).float(), (d,)).view(B, T, npf, d), cmd, tok[:, nv + 2 * T:, :4], tok[:, nv + T:nv + 2 * T, :8]
    def cond_of(cmd, prop, tch, t, k): return torch.cat([cmd[:, t + 1:t + k + 1].sum(1), cmd[:, t + 1], prop[:, t], tch[:, t]], -1)

    # ---------- données d'examen (instruments : masques et positions du monde)
    we = {**wkw, "p_out": 0.0, "p_parent": 0.0, "kick": 0.0}; EV = []
    for c0 in range(0, a.n_eval, 250):
        w0 = gen_world0(min(250, a.n_eval - c0), "0e", T, H, seed=555_000 + c0 // 250, nobj=(1, 1), masks=True, **{**we, "obj_scale": 1.0 if (c0 // 250) % 2 else sc}); MK = w0["MASK"]; nc = len(MK)
        fc = (MK.reshape(nc, T, MK.shape[2], nP, P_, nP, P_).mean((4, 6)) / 255).reshape(nc, T, MK.shape[2], npf)
        EV.append((to_torch(to_np([w0])), torch.from_numpy(fc[:, :, 0].astype(np.float32)), torch.from_numpy(fc[:, :, -1].astype(np.float32)), torch.from_numpy(w0["POS"][:, :, 0] * H).float(), torch.from_numpy(w0["HAND"] * H).float(),
                   torch.from_numpy(np.abs(w0["TOUCH"]).sum(-1) > 0)))
    log(f"   examen : {a.n_eval} scènes à un objet (tailles 1 et {sc}) | {time.time() - t_:.0f}s")

    @torch.no_grad()
    def exam(tag):
        m.eval(); t = 7; log(f"\n=== EXAMEN {tag} (image {t} -> {t}+k ; déplacements en px ; instruments = positions du monde) | {time.time() - t_:.0f}s")
        for k in (1, 2, 4):
            R = {x: [] for x in ("dt", "di", "ht", "hi", "ec", "es", "em", "oo", "ob", "oh", "tc")}
            for b, FO, FH, POS, HND, TCH in EV:
                for i in range(0, len(FO), 125):
                    sl = slice(i, i + 125); Z, cmd, prop, tch = latents({k_: x[sl] for k_, x in b.items()}); cd = cond_of(cmd, prop, tch, t, k)
                    pr, f, o = m(Z[:, t], cd, t, k); ps_, _, _ = m(Z[:, t], cd, t, k, zero_flow=True); f = f.cpu() * P_; o = o.cpu()
                    fo, fh = FO[sl, t], FH[sl, t]; core = ((fo >= 0.5) & (fh < 0.1)).float(); hc = (fh >= 0.5).float(); bg = ((fo < 0.02) & (fh < 0.02)).float()
                    avg = lambda x, M: (x * M.unsqueeze(-1)).sum(1) / M.sum(1, keepdim=True).clamp_min(1) if x.dim() == 3 else (x * M).sum(1) / M.sum(1).clamp_min(1)
                    R["dt"].append(POS[sl, t + k] - POS[sl, t]); R["di"].append(avg(f, core)); R["ht"].append(HND[sl, t + k] - HND[sl, t]); R["hi"].append(avg(f, hc))
                    ev = (((fo >= 0.1) | (FO[sl, t + k] >= 0.1)) & (fh < 0.02) & (FH[sl, t + k] < 0.02)).float()
                    for nm, q in (("ec", Z[:, t]), ("es", ps_), ("em", pr)): R[nm].append(avg((q - Z[:, t + k]).abs().mean(-1).cpu(), ev))
                    R["oo"].append(avg(o, core)); R["ob"].append(avg(o, bg)); R["oh"].append(avg(o, hc)); R["tc"].append(TCH[sl, t + 1:t + k + 1].any(1) & (core.sum(1) > 0) & (ev.sum(1) > 0))
            R = {x: torch.cat(y) for x, y in R.items()}; dn = R["dt"].norm(dim=-1); okc = R["oo"] > -1
            log(f"   k={k} | « chose » dite sur : objet {float(R['oo'].mean()):.2f}, main {float(R['oh'][R['ht'].abs().sum(-1) >= 0].mean()):.2f}, fond {float(R['ob'].mean()):.2f} | "
                f"main : vrai {float(R['ht'].norm(dim=-1).mean()):.2f} px, imaginé {float(R['hi'].norm(dim=-1).mean()):.2f}, cos {float(F.cosine_similarity(R['hi'], R['ht'], dim=-1)[R['ht'].norm(dim=-1) > 0.5].mean()):+.2f}")
            for nm, s_ in (("objet jamais touché (immobile)", (dn < 0.3)), ("déplacé de 0.5–2 px", (dn >= 0.5) & (dn < 2)), ("déplacé de 2–4 px", (dn >= 2) & (dn < 4)), ("déplacé de > 4 px", dn >= 4)):
                if int(s_.sum()) < 10: log(f"      {nm:>32s} | n={int(s_.sum()):4d} | trop peu"); continue
                cs = float(F.cosine_similarity(R["di"][s_], R["dt"][s_], dim=-1).mean()) if float(dn[s_].mean()) > 0.4 else float("nan")
                log(f"      {nm:>32s} | n={int(s_.sum()):4d} | vrai {float(dn[s_].mean()):5.2f} px | imaginé {float(R['di'][s_].norm(dim=-1).mean()):5.2f} | cos {cs:+.2f} | erreur de déplacement {float((R['di'][s_] - R['dt'][s_]).norm(dim=-1).mean()):5.2f} "
                    f"| erreur latente sur l'objet : recopier {float(R['ec'][s_].mean()):.3f} | recopier + horloge {float(R['es'][s_].mean()):.3f} | MODULE {float(R['em'][s_].mean()):.3f}")
        m.train()

    S = Stream(0, a.bs, a.workers, 0.0, wkw=dict(wkw)); S.stage = "0e"; ma = None
    for it in range(1, a.steps + 1):
        S.wkw["obj_scale"] = 1.0 if rng.random() < 0.5 else sc
        b1, b2 = S.next(), S.next(); b = {k_: torch.cat([b1[k_], b2[k_]]) for k_ in b1}; hc = b["TOUCH"].abs().sum((1, 2)) > 0
        ic, in_ = torch.where(hc)[0], torch.where(~hc)[0]; nc = min(len(ic), int(round(a.contact * a.bs))); pick = torch.cat([ic[torch.randperm(len(ic))[:nc]], in_[torch.randperm(len(in_))[:a.bs - nc]]])
        Z, cmd, prop, tch = latents({k_: x[pick] for k_, x in b.items()}); k = int(rng.integers(1, a.kmax + 1)); t = int(rng.integers(3, T - k))
        cd = cond_of(cmd, prop, tch, t, k); pr, f, o = m(Z[:, t], cd, t, k); loss = F.smooth_l1_loss(pr, Z[:, t + k], beta=0.1) + a.still_w * f.abs().mean()   # + léger a priori « les choses restent en place »
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        with torch.no_grad():
            ls = F.smooth_l1_loss(m(Z[:, t], cd, t, k, zero_flow=True)[0], Z[:, t + k], beta=0.1); lc = F.smooth_l1_loss(Z[:, t], Z[:, t + k], beta=0.1)
            cur = np.array([loss.item(), ls.item(), lc.item(), float(f.norm(dim=-1).mean()) * P_, float(f.norm(dim=-1).amax(1).mean()) * P_, float(o.mean())])
        ma = cur if ma is None else 0.98 * ma + 0.02 * cur
        if it % 250 == 0:
            log(f"  pas {it:5d} | perte {ma[0]:.4f} | sans déplacement (recopier + horloge) {ma[1]:.4f} | recopier {ma[2]:.4f} | déplacement moyen {ma[3]:.2f} px, plus grand de l'image {ma[4]:.2f} px | « chose » moyenne {ma[5]:.2f} | {time.time() - t_:.0f}s")
        if it % a.eval_every == 0 or it == a.steps:
            exam(f"au pas {it}"); torch.save(dict(m=m.state_dict(), cfg=dict(d=d, nP=nP, T=T), it=it, base=a.ckpt), a.out)
    log(f"total {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
