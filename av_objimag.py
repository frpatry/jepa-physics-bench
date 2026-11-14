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
    def __init__(s, d, nP, T, h=192, nl=2, beta=6.0, co=None, gate=0, audio=0):
        super().__init__(); s.nP, s.npf, s.beta, s.gate, s.audio = nP, nP * nP, beta, gate, audio
        if audio:                                                       # OUÏE (14 nov. 2026, demande user) : il ENTEND (latent d'ouïe de son œil) et IMAGINE le son de chaque pas
            s.aud_in = nn.Linear(d, h); nn.init.zeros_(s.aud_in.weight); nn.init.zeros_(s.aud_in.bias)
            s.aud_full = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))      # le son du pas, d'après la SCÈNE et le geste
            s.aud_body = nn.Sequential(nn.Linear(6, h), nn.GELU(), nn.Linear(h, d))      # le son que ferait SON CORPS seul (geste + main sentie) : l'écart = « j'ai fait du bruit avec quelque chose »
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
        if s.gate: f = f * o.unsqueeze(-1)   # --gate 1 (v3, échec) : seules les « choses » bougent — contourné : « chose » ≈ 0 partout et déplacement brut énorme
        if zero_flow: f = torch.zeros_like(f)
        C = Zt - s.clock[t]; B_, n_, d_ = C.shape
        C = F.conv2d(C.transpose(1, 2).reshape(B_ * d_, 1, s.nP, s.nP), s.smooth[None, None], padding=1).reshape(B_, d_, n_).transpose(1, 2) * s.gain
        num, den = s.splat(C, f, torch.exp(s.beta * o))                                              # contenu dé-daté, transporté ; une chose cache le fond
        bgw = (1 - o).unsqueeze(-1); fill = (bgw * C).sum(1, keepdim=True) / bgw.sum(1, keepdim=True).clamp_min(1e-3) + s.fill    # case quittée -> du fond
        return (num + 0.05 * fill) / (den.unsqueeze(-1) + 0.05) + s.clock[t + k], f, o


def _roll(s, Zt, cmd, prop, tch, t, k, zero_flow=False, aud=None):
    """PAS À PAS (principe du user : un geste, une image) : k pas d'UNE image, chacun avec SON geste. Les cases de la dernière image vue sont des « grains » qui gardent
    leur contenu (dé-daté, lissé une fois) et dont la POSITION avance à chaque pas (pas de flou qui s'accumule) ; à chaque pas le module regarde la scène qu'il vient
    d'imaginer, propose un déplacement (moyenné par groupe), et la main sentie est mise à jour avec le geste. -> [latents imaginés de t+1 .. t+k], déplacement total, « chose »."""
    B, n, d = Zt.shape; dev = Zt.device; C = Zt - s.clock[t]
    C = F.conv2d(C.transpose(1, 2).reshape(B * d, 1, s.nP, s.nP), s.smooth[None, None], padding=1).reshape(B, d, n).transpose(1, 2) * s.gain
    u = F.normalize(s.proj(Zt), dim=-1); A = torch.softmax(s.ps.exp() * (u @ u.transpose(1, 2)), -1); k1 = torch.ones(B, dtype=torch.long, device=dev)
    P = s.xy[None].expand(B, -1, -1).clone(); pr = prop[:, t].clone(); Zc = Zt; preds = []; o = None; AF, AB = [], []
    for j in range(1, k + 1):
        g = cmd[:, t + j]
        if j > 1: gp = cmd[:, t + j - 1]; pr = torch.cat([pr[:, :2] + 0.1 * gp, gp], -1)                # main SENTIE mise à jour par le geste déjà fait (pas de proprioception du futur)
        cd = torch.cat([g, g, pr, tch[:, t] if j == 1 else torch.zeros_like(tch[:, t])], -1)
        x = s.inp(Zc) + s.pos + s.cond(cd).unsqueeze(1) + s.kemb(k1).unsqueeze(1)
        if s.audio and aud is not None: x = x + s.aud_in(aud).unsqueeze(1)                              # ce qu'il vient d'ENTENDRE (vrai au 1er pas, imaginé ensuite)
        hf = s.tr(x)
        if s.audio: AF.append(s.aud_full(hf.mean(1))); AB.append(s.aud_body(torch.cat([g, pr], -1))); aud = AF[-1]
        if o is None: o = torch.sigmoid(s.objh(hf)).squeeze(-1); w = torch.exp(s.beta * o); bgw = (1 - o).unsqueeze(-1); fill = (bgw * C).sum(1, keepdim=True) / bgw.sum(1, keepdim=True).clamp_min(1e-3) + s.fill
        if not zero_flow:
            rg = s.flow(hf).transpose(1, 2).reshape(B, 2, s.nP, s.nP); gq = ((P + 0.5) / s.nP * 2 - 1).unsqueeze(2)       # déplacement proposé, lu LÀ OÙ SONT les grains
            P = P + A @ F.grid_sample(rg, gq, mode="bilinear", padding_mode="border", align_corners=False).squeeze(-1).transpose(1, 2)
        num, den = s.splat(C, P - s.xy[None], w); Zc = (num + 0.05 * fill) / (den.unsqueeze(-1) + 0.05) + s.clock[t + j]; preds.append(Zc)
    return preds, P - s.xy[None], o, (AF, AB)
ObjImag.roll = _roll


def _things(s, Z, prop):
    """OÙ SONT LES CHOSES ? Z (B, npf, d) latents (vus ou imaginés), prop (B, 4) main sentie -> « chose » par case (B, npf), sans geste ni toucher."""
    cd = torch.cat([torch.zeros(len(Z), 4, device=Z.device), prop, torch.zeros(len(Z), 8, device=Z.device)], -1)
    return torch.sigmoid(s.objh(s.tr(s.inp(Z) + s.pos + s.cond(cd).unsqueeze(1) + s.kemb(torch.ones(len(Z), dtype=torch.long, device=Z.device)).unsqueeze(1)))).squeeze(-1)
ObjImag.things = _things


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--out", type=str, default="/content/objimag.pt"); p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--kmax", type=int, default=4); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--contact", type=float, default=0.75, help="part du lot prise parmi les séquences où il a senti un contact"); p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--warm", type=int, default=6000, help="pas SANS déplacement au début : l'horloge, le lissage et le gain s'apprennent d'abord (sinon un faux décalage global sert de flou)")
    p.add_argument("--audio", type=int, default=0, help="1 = OUÏE : le module entend (latent d'ouïe de son œil) et imagine le son de chaque pas (cible = latent d'ouïe, aucune étiquette) ; --chain 1")
    p.add_argument("--aud_w", type=float, default=1.0)
    p.add_argument("--chain", type=int, default=0, help="1 = PAS À PAS : le module d'un pas enchaîné k fois avec le geste de chaque pas (perte latente à chaque pas)")
    p.add_argument("--init", type=str, default="", help="module de départ (ex. objimag_D17s_v5.pt : horloge, lissage, groupes déjà appris)")
    p.add_argument("--long_pow", type=float, default=0.0, help="tirage des horizons ∝ k^long_pow (0 = uniforme ; 1 = les longs plus souvent)")
    p.add_argument("--gate", type=int, default=0, help="1 = v3 : déplacement × « chose » (échec, gardé pour mémoire)")
    p.add_argument("--thing_w", type=float, default=0.0, help="a priori « peu de choses, beaucoup de fond » : petit coût sur la part de cases dites « chose »")
    p.add_argument("--still_w", type=float, default=0.005, help="a priori « les choses restent en place » : petit coût sur le déplacement moyen (en cases)")
    p.add_argument("--eval_every", type=int, default=2000); p.add_argument("--seed", type=int, default=0); p.add_argument("--dev", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.dev; t_ = time.time(); log = lambda s_: print(s_, flush=True); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]; T = int(cfg.get("T", 16)); d = cfg["d"]
    baby = baby_from_cfg(cfg, dev); baby.load_state_dict(ck["m"]); tgt = copy.deepcopy(baby.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval(); av_phase0.T = T
    for q in tgt.parameters(): q.requires_grad_(False)
    P_ = cfg["P"]; nP = H // P_; npf = nP * nP; nv = T * npf; v = VIEW["0e"]; ix = torch.arange(nv, device=dev)
    co = {"weight": ck["m"]["co_proj.weight"], "bias": ck["m"]["co_proj.bias"]} if "co_proj.weight" in ck["m"] else None
    assert a.chain or not a.audio, "--audio : avec --chain 1"
    m = ObjImag(d, nP, T, co=co, gate=a.gate, audio=a.audio).to(dev)
    if a.init: r_ = m.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["m"], strict=False); print(f"module de départ : {a.init} (neuf : {sorted({k.split('.')[0] for k in r_.missing_keys})})", flush=True)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    wkw = {k: x for k, x in cfg.get("wkw", {}).items() if k not in ("nohand", "same_col", "circ")}; sc = float(wkw.get("obj_scale", 1.0))
    log(f"IMAGINATION PAR OBJET, purement latente (œil GELÉ, pas {ck['state']['it']}) : {sum(q.numel() for q in m.parameters()) / 1e6:.2f} M de paramètres, {a.steps} pas, lots de {a.bs}, horizons 1..{a.kmax}")

    @torch.no_grad()
    def latents(b):                              # lot -> vrais latents (B, T, npf, d), condition par instant
        tok, cmd = to_tok0(b, st, v, P_, dev); B = len(tok); za = F.layer_norm(tgt(tok, torch.arange(tok.size(1), device=dev).expand(B, -1)).float(), (d,))   # tous ses sens (la vue en sort identique)
        return za[:, :nv].view(B, T, npf, d), cmd, tok[:, nv + 2 * T:, :4], tok[:, nv + T:nv + 2 * T, :8], za[:, nv:nv + T]
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
        m.eval(); log(f"\n=== EXAMEN {tag} (image t -> t+k ; déplacements en px ; instruments = positions du monde) | {time.time() - t_:.0f}s")
        for t, k in [(7, k_) for k_ in (1, 2, 4, 8) if k_ <= a.kmax] + ([(3, 12)] if a.kmax >= 12 else []):
            R = {x: [] for x in ("dt", "di", "ht", "hi", "ec", "es", "em", "oo", "ob", "oh", "tc")}
            for b, FO, FH, POS, HND, TCH in EV:
                for i in range(0, len(FO), 125):
                    sl = slice(i, i + 125); Z, cmd, prop, tch, Za = latents({k_: x[sl] for k_, x in b.items()}); cd = cond_of(cmd, prop, tch, t, k); au = Za[:, t] if a.audio else None
                    if a.chain:
                        (pl, f, o, (AF, AB)), (psl, _, _, _) = m.roll(Z[:, t], cmd, prop, tch, t, k, aud=au), m.roll(Z[:, t], cmd, prop, tch, t, k, zero_flow=True, aud=au); pr, ps_ = pl[-1], psl[-1]
                        if a.audio:
                            R.setdefault("se", []).append(torch.stack([(x - y).abs().mean(-1) for x, y in zip(AF, AB)]).amax(0).cpu()); R.setdefault("sc", []).append(TCH[sl, t + 1:t + k + 1].any(1))
                            for nm_, q_ in (("af", AF[-1]), ("ab", AB[-1]), ("ac", Za[:, t])): R.setdefault(nm_, []).append((q_ - Za[:, t + k]).abs().mean(-1).cpu())
                    else: pr, f, o = m(Z[:, t], cd, t, k); ps_, _, _ = m(Z[:, t], cd, t, k, zero_flow=True)
                    f = f.cpu() * P_; o = o.cpu()
                    fo, fh = FO[sl, t], FH[sl, t]; core = ((fo >= 0.5) & (fh < 0.1)).float(); hc = (fh >= 0.5).float(); bg = ((fo < 0.02) & (fh < 0.02)).float()
                    avg = lambda x, M: (x * M.unsqueeze(-1)).sum(1) / M.sum(1, keepdim=True).clamp_min(1) if x.dim() == 3 else (x * M).sum(1) / M.sum(1).clamp_min(1)
                    R["dt"].append(POS[sl, t + k] - POS[sl, t]); R["di"].append(avg(f, core)); R["ht"].append(HND[sl, t + k] - HND[sl, t]); R["hi"].append(avg(f, hc))
                    ev = (((fo >= 0.1) | (FO[sl, t + k] >= 0.1)) & (fh < 0.02) & (FH[sl, t + k] < 0.02)).float()
                    for nm, q in (("ec", Z[:, t]), ("es", ps_), ("em", pr)): R[nm].append(avg((q - Z[:, t + k]).abs().mean(-1).cpu(), ev))
                    R["oo"].append(avg(o, core)); R["ob"].append(avg(o, bg)); R["oh"].append(avg(o, hc)); R["tc"].append(TCH[sl, t + 1:t + k + 1].any(1) & (core.sum(1) > 0) & (ev.sum(1) > 0))
            R = {x: torch.cat(y) for x, y in R.items()}; dn = R["dt"].norm(dim=-1); okc = R["oo"] > -1
            if "se" in R: log(f"   t={t} k={k} | OUÏE imaginée : erreur avec la scène {float(R['af'].mean()):.3f} | corps seul {float(R['ab'].mean()):.3f} | recopier {float(R['ac'].mean()):.3f} || "
                              f"« bruit fait avec quelque chose » (scène − corps seul) : dit-il qu'un CONTACT a lieu ? AUC {auc(R['se'].numpy(), R['sc'].numpy()):.2f} (contact dans {float(R['sc'].float().mean()):.0%} des cas)")
            log(f"   t={t} k={k} | « chose » dite sur : objet {float(R['oo'].mean()):.2f}, main {float(R['oh'][R['ht'].abs().sum(-1) >= 0].mean()):.2f}, fond {float(R['ob'].mean()):.2f} | "
                f"main : vrai {float(R['ht'].norm(dim=-1).mean()):.2f} px, imaginé {float(R['hi'].norm(dim=-1).mean()):.2f}, cos {float(F.cosine_similarity(R['hi'], R['ht'], dim=-1)[R['ht'].norm(dim=-1) > 0.5].mean()):+.2f}")
            for nm, s_ in (("objet jamais touché (immobile)", (dn < 0.3)), ("déplacé de 0.5–2 px", (dn >= 0.5) & (dn < 2)), ("déplacé de 2–4 px", (dn >= 2) & (dn < 4)), ("déplacé de 4–8 px", (dn >= 4) & (dn < 8)), ("déplacé de > 8 px", dn >= 8)):
                if int(s_.sum()) < 10: log(f"      {nm:>32s} | n={int(s_.sum()):4d} | trop peu"); continue
                cs = float(F.cosine_similarity(R["di"][s_], R["dt"][s_], dim=-1).mean()) if float(dn[s_].mean()) > 0.4 else float("nan")
                log(f"      {nm:>32s} | n={int(s_.sum()):4d} | vrai {float(dn[s_].mean()):5.2f} px | imaginé {float(R['di'][s_].norm(dim=-1).mean()):5.2f} | cos {cs:+.2f} | erreur de déplacement {float((R['di'][s_] - R['dt'][s_]).norm(dim=-1).mean()):5.2f} "
                    f"| erreur latente sur l'objet : recopier {float(R['ec'][s_].mean()):.3f} | recopier + horloge {float(R['es'][s_].mean()):.3f} | MODULE {float(R['em'][s_].mean()):.3f}")
        m.train()

    S = Stream(0, a.bs, a.workers, 0.0, wkw=dict(wkw)); S.stage = "0e"; ma = None
    kp_ = np.arange(1, a.kmax + 1, dtype=float) ** a.long_pow; kp_ /= kp_.sum()                       # horizons LONGS tirés plus souvent (plus gros déplacements = signal plus fort)
    for it in range(1, a.steps + 1):
        S.wkw["obj_scale"] = 1.0 if rng.random() < 0.5 else sc
        b1, b2 = S.next(), S.next(); b = {k_: torch.cat([b1[k_], b2[k_]]) for k_ in b1}; hc = b["TOUCH"].abs().sum((1, 2)) > 0
        ic, in_ = torch.where(hc)[0], torch.where(~hc)[0]; nc = min(len(ic), int(round(a.contact * a.bs))); pick = torch.cat([ic[torch.randperm(len(ic))[:nc]], in_[torch.randperm(len(in_))[:a.bs - nc]]])
        Z, cmd, prop, tch, Za = latents({k_: x[pick] for k_, x in b.items()}); au = Za[:, :] if a.audio else None; k = int(rng.choice(a.kmax, p=kp_)) + 1; t = int(rng.integers(3, T - k))
        cd = cond_of(cmd, prop, tch, t, k)
        if a.chain:
            pl, f, o, (AF, AB) = m.roll(Z[:, t], cmd, prop, tch, t, k, zero_flow=it <= a.warm, aud=au[:, t] if a.audio else None); pr = pl[-1]
            loss = sum(F.smooth_l1_loss(q, Z[:, t + j + 1], beta=0.1) for j, q in enumerate(pl)) / k + a.still_w * f.abs().mean() / k + a.thing_w * o.mean()
            if a.audio: loss = loss + a.aud_w * sum(F.smooth_l1_loss(x, Za[:, t + j + 1], beta=0.1) + F.smooth_l1_loss(y, Za[:, t + j + 1], beta=0.1) for j, (x, y) in enumerate(zip(AF, AB))) / k
        else:
            pr, f, o = m(Z[:, t], cd, t, k, zero_flow=it <= a.warm); loss = F.smooth_l1_loss(pr, Z[:, t + k], beta=0.1) + a.still_w * f.abs().mean() + a.thing_w * o.mean()   # + légers a priori
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        with torch.no_grad():
            ls = F.smooth_l1_loss(m.roll(Z[:, t], cmd, prop, tch, t, k, zero_flow=True)[0][-1] if a.chain else m(Z[:, t], cd, t, k, zero_flow=True)[0], Z[:, t + k], beta=0.1); lc = F.smooth_l1_loss(Z[:, t], Z[:, t + k], beta=0.1)
            cur = np.array([F.smooth_l1_loss(pr, Z[:, t + k], beta=0.1).item(), ls.item(), lc.item(), float(f.norm(dim=-1).mean()) * P_, float(f.norm(dim=-1).amax(1).mean()) * P_, float(o.mean())])
        ma = cur if ma is None else 0.98 * ma + 0.02 * cur
        if it % 250 == 0:
            log(f"  pas {it:5d} | perte {ma[0]:.4f} | sans déplacement (recopier + horloge) {ma[1]:.4f} | recopier {ma[2]:.4f} | déplacement moyen {ma[3]:.2f} px, plus grand de l'image {ma[4]:.2f} px | « chose » moyenne {ma[5]:.2f} | {time.time() - t_:.0f}s")
        if it % a.eval_every == 0 or it == a.steps:
            exam(f"au pas {it}"); torch.save(dict(m=m.state_dict(), cfg=dict(d=d, nP=nP, T=T, gate=a.gate, audio=a.audio), it=it, base=a.ckpt), a.out)
    log(f"total {time.time() - t_:.0f}s")


if __name__ == "__main__":
    main()
