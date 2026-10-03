"""
PHASE 1 DÉVELOPPEMENTALE — un bébé entend d'abord, voit flou ensuite, net plus tard ; et on ne lui
demande que ce qu'un bébé de cet âge sait faire (capacités de SCÈNE, sans objets séparés).

Monde : les 2 disques à masse/matériau cachés (av_jepa.gen_world v2 : 16 frames, son 2 sous-fenêtres).
Modèle : JEPA maison qui APPREND sa vision (pixels 32 px, patches 8×8) et son ouïe (1 token audio/frame) —
pas de V-JEPA 2 ici : un cortex visuel déjà mature contredirait l'idée d'une vision qui se développe.
Recette : encodeur de contexte + encodeur CIBLE en moyenne mobile (EMA, recette Meta) ou SIGReg.

Bras (même nombre de pas, mêmes données) :
  dev : 1a SON SEUL (25 %) -> 1b + image TRÈS FLOUE (25 %) -> 1c flou qui diminue jusqu'à net (50 %)
  all : image nette + son dès le début
Références : init (encodeur aléatoire), raw (entrées brutes).

Tests adaptés à l'âge (lecteur transformer 2 couches, encodeur GELÉ, 1000 étiquettes, jeu tenu à l'écart) :
  matériaux PRÉSENTS dans la scène (son seul ; aucun liage à un disque)
  SYNCHRONIE : ce son va-t-il avec cette vidéo, ou est-il décalé de 4 frames ?
  LOCALISATION : le son vient-il du bon côté (stéréo correcte vs gauche/droite inversées) ?
  ANTICIPATION : en voyant/entendant les frames 0..7 seulement, un choc arrive-t-il aux frames 8-9 ?

  python av_dev.py --steps 5000            # Colab GPU
"""
import argparse, copy, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import gen_world, AVEncoder, NB, tube_masks
from vjepa import Predictor, sigreg, _idx, _gather

# ---------------------------------------------------------------- données -> tokens (GPU, avec flou)
def blur(X, sigma):
    """X (B,T,H,W,3) float -> flou gaussien par frame (sigma en pixels ; 0 = net)."""
    if sigma <= 0.05: return X
    B, T, H, W, C = X.shape; r = int(math.ceil(2.5 * sigma)); x = torch.arange(-r, r + 1, device=X.device).float()
    k = torch.exp(-x ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    im = X.permute(0, 1, 4, 2, 3).reshape(B * T * C, 1, H, W)
    im = F.conv2d(F.pad(im, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    im = F.conv2d(F.pad(im, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return im.view(B, T, C, H, W).permute(0, 1, 3, 4, 2)

def to_tokens(X, A, P, amu, asd, sigma=0.0):
    """X (B,T,H,W,3) uint8/float, A (B,T,2*a_sub,NB) -> tokens (B, T*npf + T, P*P*3) : vision puis audio."""
    X = blur(X.float() / (255.0 if X.dtype == torch.uint8 else 1.0), sigma)
    B, T, H, W, C = X.shape; nP = H // P
    v = X.reshape(B, T, nP, P, nP, P, C).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T * nP * nP, P * P * C)
    a = (A.float().reshape(B, T, -1) - amu) / asd
    pad = torch.zeros(B, T, v.size(-1), device=v.device); pad[..., :a.size(-1)] = a
    return torch.cat([v, pad], 1)

# ---------------------------------------------------------------- masques selon le stade
def masks(stage, B, T, nP, nv, rng, n_masks):
    """stage 'a' : son seul (blocs audio, futur audio) ; 'va' : vision + son (tube, bloc, futur,
    son-depuis-image, bloc audio). Retourne present (N,) et [(ctx (B,N), tgt (B,N))]."""
    npf = nP * nP; N = nv + T
    isv = np.zeros(N, bool); isv[:nv] = True; isa = ~isv
    frame = np.concatenate([np.arange(nv) // npf, np.arange(T)])
    present = isa.copy() if stage == "a" else np.ones(N, bool)
    strat = ["ablock", "afuture"] if stage == "a" else ["tube", "vblock", "future", "a_from_v", "ablock"]
    pairs = []
    for _ in range(n_masks):
        st = strat[rng.integers(len(strat))]; tg = np.zeros((B, N), bool)
        if st == "tube": tg[:, :nv] = tube_masks(B, T, nP, 0.5, 1, rng)[0].numpy()
        elif st in ("vblock", "ablock"):
            for b in range(B):
                t0 = rng.integers(0, T - 3 + 1); blk = (frame >= t0) & (frame < t0 + 3)
                tg[b] = blk & (isv if st == "vblock" else isa)
        elif st in ("future", "afuture"):
            t0 = rng.integers(3, T - 2); tg[:] = present & (frame > t0)
            pairs.append((np.broadcast_to(present & (frame <= t0), (B, N)).copy(), tg)); continue
        elif st == "a_from_v":
            tg[:] = isa; pairs.append((np.broadcast_to(isv, (B, N)).copy(), tg)); continue
        cm = present & ~tg
        if stage == "va" and st != "ablock" and rng.random() < 0.3: cm &= isv   # dropout de modalité
        pairs.append((cm, tg))
    return present, pairs

class DevJEPA(nn.Module):
    def __init__(s, dv, da, nv, T, d, nl, nh, pred_layers):
        super().__init__()
        s.nv, s.T = nv, T
        s.enc = AVEncoder(dv, da, d, nv + T, nv, nl, nh); s.pred = Predictor(d, nv + T, pred_layers, nh)

def schedule(arm, it, S, sig_max):
    """-> (stade, sigma du flou) au pas it."""
    if arm == "all": return "va", 0.0
    f = it / S
    if f < 0.25: return "a", 0.0
    if f < 0.5: return "va", sig_max
    return "va", sig_max * max(0.0, 1 - (f - 0.5) / 0.4)                 # net à 90 % de l'entraînement

def train(arm, pool, a, dev, nv, T, nP, rng, st):
    torch.manual_seed(a.seed)
    m = DevJEPA(a.P * a.P * 3, a.da, nv, T, a.d, a.nl, a.nh, a.pred_layers).to(dev)
    tgt = copy.deepcopy(m.enc).eval()
    for p_ in tgt.parameters(): p_.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05); t0 = time.time(); last = None
    for it in range(1, a.steps + 1):
        stage, sig = schedule(arm, it, a.steps, a.sig_max)
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, it / max(1, a.steps // 15))
        bi = torch.randint(0, len(pool["X"]), (a.bs,))
        o = to_tokens(pool["X"][bi].to(dev), pool["A"][bi].to(dev), a.P, st["amu"], st["asd"], sig)
        present, pairs = masks(stage, a.bs, T, nP, nv, rng, a.n_masks)
        B, N, _ = o.shape; pidx = _idx(torch.from_numpy(np.broadcast_to(present, (B, N)).copy()).to(dev))
        if a.recipe == "ema":
            with torch.no_grad(): z = F.layer_norm(tgt(_gather(o, pidx), pidx), (a.d,))
        else: z = m.enc(_gather(o, pidx), pidx)
        zf = torch.zeros(B, N, a.d, device=dev, dtype=z.dtype).scatter(1, pidx.unsqueeze(-1).expand(-1, -1, a.d), z)
        loss = 0.0
        for c_, g_ in pairs:
            cidx, tidx = _idx(torch.from_numpy(c_).to(dev)), _idx(torch.from_numpy(g_).to(dev))
            loss = loss + F.l1_loss(m.pred(m.enc(_gather(o, cidx), cidx), cidx, tidx), _gather(zf, tidx))
        loss = loss / len(pairs)
        if a.recipe == "sigreg": loss = loss + a.rw * sigreg(z.reshape(-1, a.d))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if a.recipe == "ema":
            mom = 1 - (1 - a.ema) * (math.cos(math.pi * it / a.steps) + 1) / 2
            with torch.no_grad():
                for pt, pc in zip(tgt.parameters(), m.enc.parameters()): pt.mul_(mom).add_(pc.detach(), alpha=1 - mom)
        if (stage, int(sig)) != last or it % max(1, a.steps // 10) == 0:
            last = (stage, int(sig))
            print(f"  [{arm}] step {it:5d}  stade {stage:>2s}  flou σ={sig:.1f}  pred {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    m.enc = tgt if a.recipe == "ema" else m.enc                          # protocole I-JEPA : on évalue la cible
    return m.eval()

# ---------------------------------------------------------------- tests adaptés à l'âge
class Reader(nn.Module):
    """lecteur transformer 2 couches + CLS (peut faire de l'appariement relationnel temps/espace)."""
    def __init__(s, din, ntok, nout, d=192, nl=2, nh=6):
        super().__init__()
        s.emb = nn.Linear(din, d); s.pos = nn.Parameter(torch.zeros(1, ntok, d)); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, nh, 2 * d, batch_first=True, dropout=0.1, activation="gelu")
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, nout)
    def forward(s, x):
        h = s.emb(x) + s.pos[:, :x.size(1)]
        return s.out(s.ln(s.tr(torch.cat([s.cls.expand(len(x), -1, -1), h], 1))[:, 0]))

def fit_reader(Rtr, ytr, Rte, task, dev, steps, bs=128):
    nout = ytr.size(-1) if task == "multi" else 2
    r = Reader(Rtr.size(-1), Rtr.size(1), nout).to(dev); opt = torch.optim.AdamW(r.parameters(), 3e-4, weight_decay=0.05)
    for _ in range(steps):
        bi = torch.randint(0, len(Rtr), (bs,)); out = r(Rtr[bi].to(dev).float()); y = ytr[bi].to(dev)
        l = F.binary_cross_entropy_with_logits(out, y.float()) if task == "multi" else F.cross_entropy(out, y)
        opt.zero_grad(); l.backward(); opt.step()
    r.eval()
    with torch.no_grad(): return torch.cat([r(Rte[i:i + 256].to(dev).float()).cpu() for i in range(0, len(Rte), 256)])

@torch.no_grad()
def represent(m, tok, keep, dev, bs=128):
    """m None = entrées brutes ; sinon encodeur GELÉ sur les seuls tokens gardés (aucune fuite)."""
    idx = torch.from_numpy(np.where(keep)[0]).to(dev); out = []
    for i in range(0, len(tok), bs):
        o = tok[i:i + bs].to(dev).float(); ix = idx.expand(len(o), -1)
        out.append((_gather(o, ix) if m is None else m.enc(_gather(o, ix), ix)).half().cpu())
    return torch.cat(out)

def bacc(p, y): return float(((p[y == 1] == 1).float().mean() + (p[y == 0] == 0).float().mean()) / 2)

def run_tests(m, name, probe, a, dev, nv, T):
    """probe : dict de tokens (normal, décalé, stéréo inversée) + étiquettes ; 1000 étiq. / test sur le reste."""
    L, n = a.labels, len(probe["tok"]); tr, te = slice(0, L), slice(n - a.n_test, n)
    N = nv + T; isv = np.zeros(N, bool); isv[:nv] = True; npf = nv // T
    frame = np.concatenate([np.arange(nv) // npf, np.arange(T)]); res = {}
    # 1) matériaux présents (son seul)
    R = represent(m, probe["tok"], ~isv, dev); y = probe["matset"]
    p = (fit_reader(R[tr], y[tr], R[te], "multi", dev, a.read_steps) > 0).long()
    res["matériaux (son)"] = float((p == y[te]).all(1).float().mean())
    # 2) synchronie et 3) localisation : moitié vrai son, moitié son modifié
    for key, mod in [("synchronie", "shift"), ("localisation", "swap")]:
        lab = (torch.arange(n) % 2).long()                              # 1 = son modifié
        tok = torch.where(lab.view(-1, 1, 1).bool(), probe[mod], probe["tok"])
        R = represent(m, tok, np.ones(N, bool), dev)
        p = fit_reader(R[tr], lab[tr], R[te], "bin", dev, a.read_steps).argmax(-1); res[key] = bacc(p, lab[te])
    # 4) anticipation : frames 0..7 seulement -> choc aux frames 8-9 ?
    keep = frame <= 7; R = represent(m, probe["tok"], keep, dev); y = probe["antic"]
    p = fit_reader(R[tr], y[tr], R[te], "bin", dev, a.read_steps).argmax(-1); res["anticipation"] = bacc(p, y[te])
    print(f"  {name:>5s} | " + " | ".join(f"{k} {v:.0%}" for k, v in res.items()), flush=True)
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pool", type=int, default=20000); p.add_argument("--n_probe", type=int, default=3000)
    p.add_argument("--T", type=int, default=16); p.add_argument("--H", type=int, default=32); p.add_argument("--P", type=int, default=8)
    p.add_argument("--a_sub", type=int, default=2); p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_masks", type=int, default=3); p.add_argument("--d", type=int, default=128)
    p.add_argument("--nl", type=int, default=4); p.add_argument("--nh", type=int, default=4); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--recipe", type=str, default="ema", choices=["ema", "sigreg"]); p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--rw", type=float, default=0.05); p.add_argument("--sig_max", type=float, default=3.0, help="flou initial (px)")
    p.add_argument("--arms", type=str, default="dev,all"); p.add_argument("--refs", type=str, default="raw,init")
    p.add_argument("--labels", type=int, default=1000); p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--read_steps", type=int, default=2000); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time(); rng = np.random.default_rng(a.seed)
    Xs, As = [], []                                                     # par morceaux : sinon OOM (20k × float32)
    for c0 in range(0, a.n_pool, 2000):
        w = gen_world(min(2000, a.n_pool - c0), a.T, a.H, seed=a.seed + 7919 * (c0 // 2000), a_sub=a.a_sub)
        Xs.append(torch.from_numpy((w["X"] * 255).round().astype(np.uint8))); As.append(torch.from_numpy(w["A"].astype(np.float16))); del w
    pool = dict(X=torch.cat(Xs), A=torch.cat(As)); del Xs, As
    wp = gen_world(a.n_probe, a.T, a.H, seed=1000, a_sub=a.a_sub)
    a.da = 2 * a.a_sub * NB; nP = a.H // a.P; nv = a.T * nP * nP
    Af = pool["A"][:4000].float(); Af = Af.reshape(len(Af), a.T, -1)
    st = dict(amu=Af.mean((0, 1)).to(dev), asd=(Af.std((0, 1)) + 1e-4).to(dev))
    print(f"pool {a.n_pool} + sonde {a.n_probe} séquences ({time.time() - t0:.0f}s) | {a.T} frames × {nP * nP} patches + {a.T} audio", flush=True)
    # jeu de sonde : tokens nets ; son décalé de 4 frames ; stéréo inversée
    Xp, Ap = torch.from_numpy(wp["X"]), torch.from_numpy(wp["A"])
    Ash = torch.roll(Ap, 4, dims=1); Asw = Ap[:, :, [1, 0, 3, 2][:Ap.size(2)]] if Ap.size(2) == 4 else Ap.flip(2)
    mk = lambda A_: torch.cat([to_tokens(Xp[i:i + 500].to(dev), A_[i:i + 500].to(dev), a.P, st["amu"], st["asd"]).half().cpu()
                               for i in range(0, a.n_probe, 500)])
    matset = torch.zeros(a.n_probe, 3, dtype=torch.long)
    for k in range(2): matset[torch.arange(a.n_probe), torch.from_numpy(wp["MAT"][:, k])] = 1
    probe = dict(tok=mk(Ap), shift=mk(Ash), swap=mk(Asw), matset=matset,
                 antic=torch.from_numpy(wp["IMP"][:, 8] | wp["IMP"][:, 9]).long())
    print(f"sonde : anticipation (choc frames 8-9) {probe['antic'].float().mean():.0%} positifs | matériaux distincts/scène "
          f"{matset.sum(1).float().mean():.2f}", flush=True)
    rows = []
    for ref in [r for r in a.refs.split(",") if r]:
        if ref == "raw": rows.append(("raw", run_tests(None, "raw", probe, a, dev, nv, a.T)))
        if ref == "init":
            torch.manual_seed(a.seed); m0 = DevJEPA(a.P * a.P * 3, a.da, nv, a.T, a.d, a.nl, a.nh, a.pred_layers).to(dev).eval()
            rows.append(("init", run_tests(m0, "init", probe, a, dev, nv, a.T)))
    for arm in a.arms.split(","):
        print(f"--- bras {arm.upper()} ({a.steps} pas, recette {a.recipe}, AUCUNE étiquette)", flush=True)
        m = train(arm, pool, a, dev, nv, a.T, nP, rng, st)
        rows.append((arm.upper(), run_tests(m, arm.upper(), probe, a, dev, nv, a.T)))
    keys = list(rows[0][1].keys())
    print("\n========== TESTS ADAPTÉS À L'ÂGE (lecteur gelé, 1000 étiquettes ; hasard : 50 % sauf matériaux) ==========")
    print(f"{'repr':>5s} | " + " | ".join(f"{k:>15s}" for k in keys))
    for name, r in rows: print(f"{name:>5s} | " + " | ".join(f"{r[k]:15.0%}" for k in keys))
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
