"""
BOÎTES-OBJETS — PREMIER ESSAI (conception : conception_boites_objets.md ; fondements : recherche_bebe_objets.md).

L'œil du bébé de phase 0 (run 4, 16 images) est FIGÉ. Par-dessus :
  - 4 BOÎTES apprises, sans étiquette : les 64 carrés de chaque image se les DISPUTENT (Slot Attention) ; les boîtes
    de l'instant t-1, propagées par l'imagination, servent de point de départ à celles de t (suivi « prédire puis
    corriger », façon SAVi — aucun appariement par oracle) ;
  - ÉMERGENCE : mouvement d'abord (les boîtes prédisent le CHANGEMENT de chaque carré entre t et t+1 : ce qui bouge
    ensemble va ensemble), apparence ensuite (elles reconstruisent le résumé de l'œil) — dosage qui glisse au fil
    de l'entraînement, comme 4 mois -> 8 mois ;
  - IMAGINATION PAR PAIRES : boîte_k(t+1) = boîte_k(t) + élan(k) + Σ interactions(k, j) + interaction(k, CORPS)
    + effet(k, toucher/ouïe) ; le CORPS (commande + sens du bras) est offert à TOUTES les boîtes (aucune réservée) ;
    entraînée en JEPA : les boîtes imaginées sur 8 pas doivent rejoindre celles que l'œil trouvera vraiment.
Interdits respectés : aucun masque/position d'objet, aucun nombre d'objets, aucune boîte-main, aucune loi physique.

EXAMENS (instruments — jamais donnés au modèle) :
  A ÉMERGENCE : centre de chaque boîte (là où elle « regarde ») vs vrais objets + main (meilleur appariement) ;
  B TEST DE LA MAIN : hors contact, mentir sur la commande ne doit changer QUE la boîte qui suit la main ;
  C LE VRAI JUGE : objet IMMOBILE — reste-t-il en place dans l'imagination ? objet poussé vs « rien ne bouge ».

  python av_slots0.py --eye /content/drive/MyDrive/jepa_runs/phase0_r4_0e_60k.pt --ckpt /content/drive/MyDrive/jepa_runs/slots0.pt
"""
import argparse, copy, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import av_phase0 as P0
from av_world0 import gen_world0, VIEW
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, Stream

H = 32

def grid_xy(nP, dev):
    g = (torch.arange(nP, device=dev).float() + 0.5) / nP; yy, xx = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], -1)                     # (npf, 2) centre de chaque carré (x, y)

class SlotAttn(nn.Module):
    """les boîtes se DISPUTENT les carrés (softmax sur les boîtes) ; init fournie (suivi) ou apprise (1re image)."""
    def __init__(s, din, ds, K):
        super().__init__(); s.K, s.ds = K, ds
        s.init = nn.Parameter(torch.randn(1, K, ds) * 0.5)
        s.k, s.v, s.q = nn.Linear(din, ds, bias=False), nn.Linear(din, ds, bias=False), nn.Linear(ds, ds, bias=False)
        s.gru = nn.GRUCell(ds, ds); s.mlp = nn.Sequential(nn.Linear(ds, 2 * ds), nn.ReLU(), nn.Linear(2 * ds, ds))
        s.ni, s.ns, s.nm = nn.LayerNorm(din), nn.LayerNorm(ds), nn.LayerNorm(ds)
    def forward(s, x, slots=None, iters=2):
        B = len(x); slots = s.init.expand(B, -1, -1) if slots is None else slots
        x = s.ni(x); k, v = s.k(x), s.v(x)
        for _ in range(iters):
            att = torch.softmax((k @ s.q(s.ns(slots)).transpose(1, 2)) * s.ds ** -0.5, dim=-1) + 1e-8     # (B, N, K)
            w = att / att.sum(1, keepdim=True)
            slots = s.gru((w.transpose(1, 2) @ v).reshape(-1, s.ds), slots.reshape(-1, s.ds)).reshape(B, s.K, s.ds)
            slots = slots + s.mlp(s.nm(slots))
        return slots, att

class Decoder(nn.Module):
    """FAIBLE exprès (leçon slots.py : un décodeur puissant laisse UNE boîte tout peindre) : boîte + position du carré
    -> [apparence (d), changement vers t+1 (d), poids α] ; les boîtes se partagent chaque carré (softmax de α)."""
    def __init__(s, ds, d, npf, h=128):
        super().__init__(); s.pos = nn.Parameter(torch.randn(1, 1, npf, ds) * 0.1); s.d = d
        s.mlp = nn.Sequential(nn.Linear(ds, h), nn.ReLU(), nn.Linear(h, 2 * d + 1))
    def forward(s, slots):
        o = s.mlp(slots.unsqueeze(2) + s.pos)                                    # (B, K, npf, 2d+1)
        al = torch.softmax(o[..., -1], dim=1)                                    # (B, K, npf) : qui explique ce carré
        app = (al.unsqueeze(-1) * o[..., :s.d]).sum(1); mot = (al.unsqueeze(-1) * o[..., s.d:2 * s.d]).sum(1)
        return app, mot, al

class PairDyn(nn.Module):
    """IMAGINATION PAR PAIRES : élan propre + interactions boîte–boîte + boîte–CORPS + boîte–sens (toucher, ouïe)."""
    def __init__(s, ds, h=128):
        super().__init__()
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))
        s.self_ = mlp(ds + 2, ds); s.pair = mlp(2 * ds + 2, ds); s.body = mlp(ds + 2 + 2 + 4 + 1, ds); s.sens = mlp(ds + 2 + 8 + 128 + 1, ds)
        s.upd = mlp(4 * ds, ds)
        nn.init.zeros_(s.upd[-1].weight); nn.init.zeros_(s.upd[-1].bias)       # départ : « rien ne change »
    def forward(s, sl, pos, cmd, prop, pf, touch, audio, sf):
        B, K, ds = sl.shape
        e_self = s.self_(torch.cat([sl, pos], -1))
        si, sj = sl.unsqueeze(2).expand(B, K, K, ds), sl.unsqueeze(1).expand(B, K, K, ds)
        rel = pos.unsqueeze(2) - pos.unsqueeze(1)
        e_pair = s.pair(torch.cat([si, sj, rel], -1)) * (1 - torch.eye(K, device=sl.device)).view(1, K, K, 1)
        e_pair = e_pair.sum(2) / max(1, K - 1)
        bt = torch.cat([cmd, prop * pf, pf], -1).unsqueeze(1).expand(B, K, -1)                # CORPS : offert à TOUTES les boîtes
        e_body = s.body(torch.cat([sl, pos, bt], -1))
        st = torch.cat([touch * sf, audio * sf, sf], -1).unsqueeze(1).expand(B, K, -1)
        e_sens = s.sens(torch.cat([sl, pos, st], -1))
        return sl + s.upd(torch.cat([e_self, e_pair, e_body, e_sens], -1))

class Boxes(nn.Module):
    def __init__(s, d, npf, K=4, ds=64):
        super().__init__(); s.sa = SlotAttn(d, ds, K); s.dec = Decoder(ds, d, npf); s.dyn = PairDyn(ds)
    def pos(s, al, gxy): return (al @ gxy) / al.sum(-1, keepdim=True).clamp_min(1e-6)     # (B, K, 2) centre de ce que la boîte « prend »
    def slot_pos(s, sl, gxy): return s.pos(s.dec(sl)[2], gxy)                       # position d'une boîte = centre de ce qu'elle explique (même règle vue / imaginée)

def senses(tok, cmd, nv, T):
    """commande (B,T,2) ; sens du bras (B,T,4) ; toucher (B,T,8) ; ouïe (B,T,128) — tokens normalisés de la phase 0."""
    return cmd, tok[:, nv + 2 * T:, :4], tok[:, nv + T:nv + 2 * T, :8], tok[:, nv:nv + T, :128]

def track(bx, Z, cmd, prop, touch, audio, gxy, drop=0.0):
    """suivi « prédire puis corriger » sur T images -> boîtes (B,T,K,ds), attention (B,T,npf,K), prédictions (B,T-1,K,ds)."""
    B, T = Z.shape[:2]; sl, att = bx.sa(Z[:, 0], None, iters=3); S, A, Pr = [sl], [att], []
    for t in range(1, T):
        pos = bx.slot_pos(sl, gxy)
        pf = (torch.rand(B, 1, device=Z.device) >= drop).float(); sf = (torch.rand(B, 1, device=Z.device) >= drop).float()
        pred = bx.dyn(sl, pos, cmd[:, t], prop[:, t - 1], pf, touch[:, t - 1], audio[:, t - 1], sf); Pr.append(pred)
        sl, att = bx.sa(Z[:, t], pred, iters=2); S.append(sl); A.append(att)
    return torch.stack(S, 1), torch.stack(A, 1), torch.stack(Pr, 1)

def imagine(bx, sl0, att0, cmd, t0, H_, gxy):
    """IMAGINATION en boucle ouverte depuis l'instant t0 : seulement les gestes (aucun sens futur) -> boîtes (B,H,K,ds), positions."""
    B, K, _ = sl0.shape; z1 = torch.zeros(B, 1, device=sl0.device); sl = sl0; pos = bx.slot_pos(sl0, gxy); S, Ps = [], []
    for h in range(1, H_ + 1):
        sl = bx.dyn(sl, pos, cmd[:, t0 + h], torch.zeros(B, 4, device=sl.device), z1, torch.zeros(B, 8, device=sl.device), torch.zeros(B, 128, device=sl.device), z1)
        _, _, al = bx.dec(sl); pos = bx.pos(al, gxy); S.append(sl); Ps.append(pos)
    return torch.stack(S, 1), torch.stack(Ps, 1)

@torch.no_grad()
def eye(enc, b, st, P, dev, nv, npf, d, T):
    tok, cmd = to_tok0(b, st, VIEW["0e"], P, dev); ix = torch.arange(nv, device=dev).expand(len(tok), -1)
    Z = F.layer_norm(enc(tok[:, :nv], ix).float(), (d,)).view(len(tok), T, npf, d)
    return Z, tok, cmd

@torch.no_grad()
def exam(bx, enc, st, P, dev, nv, npf, d, T, probes, gxy, tag):
    bx.eval(); res = {}
    # A ÉMERGENCE (monde 0e classique) : centres des boîtes vs vrais objets visibles + main
    w = probes["A"]; b = to_torch(w); errs_o, errs_h, hand_k = [], [], []
    for i in range(0, len(b["X"]), 50):
        bb = {k: x[i:i + 50] for k, x in b.items()}; Z, tok, cmd = eye(enc, bb, st, P, dev, nv, npf, d, T)
        S, A, _ = track(bx, Z, *senses(tok, cmd, nv, T), gxy)
        pos = bx.pos(A.transpose(2, 3).flatten(0, 1), gxy).view(len(Z), T, -1, 2).cpu().numpy()
        for j in range(len(Z)):
            g = i + j
            for t in (3, 8, 15):
                ents = [w["HAND"][g, t]] + [w["POS"][g, t, k] for k in range(w["NOBJ"][g]) if w["VIS"][g, t, k]]
                C = np.linalg.norm(np.array(ents)[:, None] - pos[j, t][None], axis=-1); r, c = linear_sum_assignment(C)
                for rr, cc in zip(r, c): (errs_h if rr == 0 else errs_o).append(C[rr, cc] * 32)
                hand_k.append(c[list(r).index(0)])
    res["obj_px"], res["main_px"] = float(np.mean(errs_o)), float(np.mean(errs_h))
    res["obj_pris"] = float(np.mean(np.array(errs_o) < 4)); res["main_prise"] = float(np.mean(np.array(errs_h) < 3))
    res["main_boite_stable"] = float(np.mean(np.array(hand_k).reshape(-1, 3).std(1) == 0))
    # B et C : imagination depuis t0 = 7, 8 pas, monde 0e à UN objet
    w = probes["C"]; b = to_torch(w); n = len(b["X"]); t0 = 7
    Pi, Pl, P0_, Ptrue_o, Ptrue_h, Ko, Kh = [], [], [], [], [], [], []
    for i in range(0, n, 50):
        bb = {k: x[i:i + 50] for k, x in b.items()}; Z, tok, cmd = eye(enc, bb, st, P, dev, nv, npf, d, T)
        sen = senses(tok, cmd, nv, T); S, A, _ = track(bx, Z[:, :t0 + 1], *(x[:, :t0 + 1] for x in sen), gxy)
        sl0, at0 = S[:, -1], A[:, -1]; p0 = bx.slot_pos(sl0, gxy)
        _, pi = imagine(bx, sl0, at0, cmd, t0, 8, gxy); _, pl = imagine(bx, sl0, at0, cmd.roll(1, 0), t0, 8, gxy)
        Pi.append(pi.cpu()); Pl.append(pl.cpu()); P0_.append(p0.cpu())
    Pi, Pl, P0_ = torch.cat(Pi).numpy(), torch.cat(Pl).numpy(), torch.cat(P0_).numpy()
    obj, hand = w["POS"][:, :, 0], w["HAND"]
    ko = np.linalg.norm(P0_ - obj[:, t0][:, None], axis=-1).argmin(1); kh = np.linalg.norm(P0_ - hand[:, t0][:, None], axis=-1).argmin(1)   # instrument : quelle boîte suit quoi à t0
    ar = np.arange(n); fut = slice(t0 + 1, t0 + 9)
    e_obj = np.linalg.norm(Pi[ar, :, ko] - obj[:, fut], axis=-1).mean(1) * 32; c_obj = np.linalg.norm(P0_[ar, ko][:, None] - obj[:, fut], axis=-1).mean(1) * 32
    e_hand = np.linalg.norm(Pi[ar, :, kh] - hand[:, fut], axis=-1).mean(1) * 32; l_hand = np.linalg.norm(Pl[ar, :, kh] - hand[:, fut], axis=-1).mean(1) * 32
    c_hand = np.linalg.norm(P0_[ar, kh][:, None] - hand[:, fut], axis=-1).mean(1) * 32
    still = np.linalg.norm(obj[:, t0 + 1:t0 + 9] - obj[:, t0:t0 + 1], axis=-1).max(1) * 32 < 0.3
    nocont = (w["TSRC"][:, t0 + 1:t0 + 9] == 0).all(1) & (ko != kh)
    drift = np.linalg.norm(Pi[ar, :, ko] - P0_[ar, ko][:, None], axis=-1).max(1) * 32
    res["immobile_derive"], res["immobile_err"], res["immobile_copie"] = float(drift[still].mean()), float(e_obj[still].mean()), float(c_obj[still].mean())
    res["pousse_err"], res["pousse_copie"] = float(e_obj[~still].mean()), float(c_obj[~still].mean())
    res["main_err"], res["main_autre"], res["main_copie"] = float(e_hand.mean()), float(l_hand.mean()), float(c_hand.mean())
    dk = np.linalg.norm(Pl - Pi, axis=-1).mean(1) * 32                                   # (n, K) : combien chaque boîte change si on ment
    res["mensonge_main"] = float(dk[ar, kh][nocont].mean()); res["mensonge_objet"] = float(dk[ar, ko][nocont].mean())
    print(f"  EXAMEN {tag}\n    A ÉMERGENCE : objets {res['obj_px']:.1f} px ({res['obj_pris']:.0%} pris < 4 px) | main {res['main_px']:.1f} px ({res['main_prise']:.0%} < 3 px) "
          f"| même boîte pour la main d'un bout à l'autre {res['main_boite_stable']:.0%}\n"
          f"    B TEST DE LA MAIN (hors contact, on ment sur la commande) : boîte-main bouge de {res['mensonge_main']:.1f} px, boîte-objet de {res['mensonge_objet']:.1f} px\n"
          f"    C IMAGINATION (8 pas) : objet IMMOBILE dérive {res['immobile_derive']:.1f} px (erreur {res['immobile_err']:.1f} vs copie {res['immobile_copie']:.1f}) "
          f"| objet poussé {res['pousse_err']:.1f} vs copie {res['pousse_copie']:.1f} | main {res['main_err']:.1f} (gestes d'un autre {res['main_autre']:.1f}, copie {res['main_copie']:.1f})", flush=True)
    bx.train(); return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eye", required=True, help="instantané de phase 0 (œil figé)"); p.add_argument("--ckpt", default="/content/slots0.pt")
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--K", type=int, default=5, help="5 = 3 objets + main + marge pour le fond (recherche : 4 ou 5)"); p.add_argument("--ds", type=int, default=64); p.add_argument("--H", type=int, default=8)
    p.add_argument("--w_dyn", type=float, default=1.0); p.add_argument("--w_img", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--n_probe", type=int, default=300); p.add_argument("--exam_every", type=int, default=5000)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0_ = time.time()
    ck = torch.load(a.eye, map_location=dev, weights_only=False); cfg, st = ck["cfg"], ck["norm"]
    baby = baby_from_cfg(cfg, dev); T = P0.T; enc = copy.deepcopy(baby.enc); enc.load_state_dict(ck["tgt"]); enc.eval()
    for q in enc.parameters(): q.requires_grad_(False)
    P = cfg["P"]; nP = H // P; npf = nP * nP; nv = T * npf; d = cfg["d"]; gxy = grid_xy(nP, dev); wkw = dict(cfg.get("wkw", {}))
    bx = Boxes(d, npf, a.K, a.ds).to(dev); opt = torch.optim.AdamW(bx.parameters(), a.lr, weight_decay=0.01)
    wA = gen_world0(a.n_probe, "0e", T, H, seed=5101); wC = gen_world0(3 * a.n_probe, "0e", T, H, seed=5102); keep = np.where(wC["NOBJ"] == 1)[0]
    probes = dict(A={**to_np([wA]), **{k: wA[k] for k in ("HAND", "POS", "NOBJ", "VIS")}},
                  C={**to_np([{k: wC[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}]), **{k: wC[k][keep] for k in ("HAND", "POS", "TSRC")}})
    print(f"œil figé : {a.eye} (pas {ck['state']['it']}) | {T} images | {a.K} boîtes de {a.ds} | imagination par paires | {time.time() - t0_:.0f}s", flush=True)
    state = dict(it=0, exams=[])
    if os.path.exists(a.ckpt):
        c2 = torch.load(a.ckpt, map_location=dev, weights_only=False); bx.load_state_dict(c2["bx"]); opt.load_state_dict(c2["opt"]); state = c2["state"]
        print(f"REPRISE au pas {state['it']}", flush=True)
    else: state["exams"].append(("init", exam(bx, enc, st, P, dev, nv, npf, d, T, probes, gxy, "init (boîtes aléatoires)")))
    S_ = Stream(state["it"], a.bs, a.workers, 0.3, wkw=wkw); ma = None
    while state["it"] < a.steps:
        it = state["it"] = state["it"] + 1; f = it / a.steps
        S_.stage = "0d" if f < 0.5 else "0e"; fs = (f / 0.5) if f < 0.5 else (f - 0.5) / 0.5      # monde lisible : calme 1 -> 0.7 -> 0.3
        S_.wkw["calm"] = 1 - 0.3 * fs if f < 0.5 else 0.7 - 0.4 * fs; S_.wkw["p_out"] = 0.1 + 0.2 * fs if f < 0.5 else 0.3 + 0.3 * fs
        w_mot, w_app = 1.0 - 0.8 * f, 0.2 + 0.8 * f                                        # MOUVEMENT d'abord, APPARENCE ensuite
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, it / 1000)
        Z, tok, cmd = eye(enc, S_.next(), st, P, dev, nv, npf, d, T); sen = senses(tok, cmd, nv, T)
        S, A, Pr = track(bx, Z, *sen, gxy, drop=0.3)
        app, mot, _ = bx.dec(S.flatten(0, 1)); app, mot = app.view(Z.shape), mot.view(Z.shape)
        l_app = F.mse_loss(app, Z); l_mot = F.mse_loss(mot[:, :-1], Z[:, 1:] - Z[:, :-1])
        t0 = int(np.random.randint(3, T - 2)); Hh = min(a.H, T - 1 - t0)
        Si, _ = imagine(bx, S[:, t0], A[:, t0], cmd, t0, Hh, gxy)
        l_dyn = F.mse_loss(Si, S[:, t0 + 1:t0 + 1 + Hh].detach())                         # JEPA : rejoindre les boîtes que l'œil trouvera
        ai, _, _ = bx.dec(Si.flatten(0, 1)); l_img = F.mse_loss(ai.view(len(Z), Hh, npf, d), Z[:, t0 + 1:t0 + 1 + Hh])   # ancrage dans la scène
        loss = w_app * l_app + w_mot * l_mot + a.w_dyn * l_dyn + a.w_img * l_img
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(bx.parameters(), 1.0); opt.step()
        cur = np.array([l_app.item(), l_mot.item(), l_dyn.item(), l_img.item()]); ma = cur if ma is None else 0.99 * ma + 0.01 * cur
        if it % 500 == 0:
            print(f"  pas {it:6d} | monde {S_.stage} calme {S_.wkw['calm']:.2f} | poids mouvement {w_mot:.2f} apparence {w_app:.2f} | apparence {ma[0]:.4f} "
                  f"mouvement {ma[1]:.4f} imagination(boîtes) {ma[2]:.4f} imagination(scène) {ma[3]:.4f} | {time.time() - t0_:.0f}s", flush=True)
        if it % a.exam_every == 0 or it == a.steps:
            state["exams"].append((f"pas {it}", exam(bx, enc, st, P, dev, nv, npf, d, T, probes, gxy, f"pas {it}")))
            torch.save(dict(bx=bx.state_dict(), state=state, cfg=vars(a)), a.ckpt.replace(".pt", f"_{it // 1000}k.pt"))
        if it % 2500 == 0 or it == a.steps:
            torch.save(dict(bx=bx.state_dict(), opt=opt.state_dict(), state=state, cfg=vars(a)), a.ckpt + ".tmp"); os.replace(a.ckpt + ".tmp", a.ckpt)
    print(f"total {time.time() - t0_:.0f}s")

if __name__ == "__main__":
    main()
