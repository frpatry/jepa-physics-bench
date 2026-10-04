"""
DIAGNOSTIC du run long (av_dev_long.py) — pourquoi la LOCALISATION (son <-> image) n'apparaît pas ?
Charge le checkpoint et isole la cause en 3 tests (encodeur cible EMA gelé ; réf. = encodeur aléatoire) :
  (a) PERCEPTION   : la vision sait-elle OÙ sont les disques ? (positions par frame, R²)
  (b) OBJECTIF     : le son PRÉDIT depuis l'image contient-il la bonne direction gauche/droite ?
                     (signe de la différence G-D aux frames d'impact, lu sur le latent audio prédit
                     vs sur le vrai latent audio)
  (c) EXAMEN       : localisation sur des représentations RÉSUMÉES par frame (16 tokens) et 2000 étiq.
  (d) OREILLES     : l'encodeur nourri du SEUL son sait-il de quel côté a eu lieu le choc ? (idée user :
                     l'ouïe localise, la vision devrait « demander » à l'ouïe)
  (e) SURPRISE     : violation d'attente, ZÉRO étiquette — l'erreur de prédiction du son (depuis l'image)
                     est-elle plus grande quand la stéréo est inversée ? (test du bébé)
Lecture : (a) bas -> perception ; (a) ok mais (b) bas -> l'objectif ignore la stéréo ; (c) ok -> c'était
l'examen (lecteur trop chargé).

  python av_dev_diag.py --ckpt /content/drive/MyDrive/jepa_runs/av_dev_long.pt
"""
import argparse, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import gen_world, NB
from av_dev import DevJEPA, fit_reader, bacc
from av_dev_long import to_tokens, stereo, surprise, T, H
from vjepa import _idx, _gather

class RegReader(nn.Module):
    def __init__(s, din, ntok, nout, d=128):
        super().__init__()
        s.emb = nn.Linear(din, d); s.pos = nn.Parameter(torch.zeros(1, ntok, d)); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.1), 2); s.out = nn.Linear(d, nout)
    def forward(s, x):
        h = s.emb(x) + s.pos[:, :x.size(1)]
        return s.out(s.tr(torch.cat([s.cls.expand(len(x), -1, -1), h], 1))[:, 0])

def fit_reg(Xtr, ytr, Xte, dev, steps):
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-6; r = RegReader(Xtr.size(-1), Xtr.size(1), ytr.size(-1)).to(dev)
    opt = torch.optim.AdamW(r.parameters(), 3e-4, weight_decay=0.05)
    for _ in range(steps):
        bi = torch.randint(0, len(Xtr), (256,)); l = F.mse_loss(r(Xtr[bi].to(dev).float()), ((ytr[bi] - mu) / sd).to(dev))
        opt.zero_grad(); l.backward(); opt.step()
    r.eval()
    with torch.no_grad(): return torch.cat([r(Xte[i:i + 512].to(dev).float()).cpu() for i in range(0, len(Xte), 512)]) * sd + mu

def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

@torch.no_grad()
def encode(enc, tok, dev, bs=64):
    idx = torch.arange(tok.size(1), device=dev); return torch.cat([enc(tok[i:i + bs].to(dev).float(), idx.expand(len(tok[i:i + bs]), -1)).half().cpu()
                                                                  for i in range(0, len(tok), bs)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="/content/drive/MyDrive/jepa_runs/av_dev_long.pt")
    p.add_argument("--P", type=int, default=4); p.add_argument("--d", type=int, default=192); p.add_argument("--nl", type=int, default=6)
    p.add_argument("--nh", type=int, default=6); p.add_argument("--pred_layers", type=int, default=3)
    p.add_argument("--n", type=int, default=3000); p.add_argument("--steps", type=int, default=1500); p.add_argument("--seed", type=int, default=0); p.add_argument("--hum", type=float, default=0.0)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    nP = H // a.P; npf = nP * nP; nv = T * npf; da = 2 * 2 * NB; W = max(a.P * a.P * 3, da)
    w0 = gen_world(2000, T, H, seed=a.seed, a_sub=2, hum=a.hum); A0 = stereo(torch.from_numpy(w0["A"])).reshape(2000, T, -1)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev)); del w0     # mêmes stats que le run
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    print(f"checkpoint : pas {ck['state']['it']} étape {ck['state']['stage']} ({time.time() - t0:.0f}s)", flush=True)
    m = DevJEPA(W, da, nv, T, a.d, a.nl, a.nh, a.pred_layers).to(dev); m.load_state_dict(ck["m"]); m.eval()
    tgt = copy.deepcopy(m.enc); tgt.load_state_dict(ck["tgt"]); tgt.eval()
    torch.manual_seed(a.seed); m0 = DevJEPA(W, da, nv, T, a.d, a.nl, a.nh, a.pred_layers).to(dev).eval()
    w = gen_world(a.n, T, H, seed=1000, a_sub=2, hum=a.hum)
    X, A = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)), torch.from_numpy(w["A"])
    tok = torch.cat([to_tokens(X[i:i + 250].to(dev), A[i:i + 250].to(dev), a.P, st).half().cpu() for i in range(0, a.n, 250)])
    ntr = int(0.7 * a.n); tr, te = slice(0, ntr), slice(ntr, a.n)
    P2 = w["POS"]; o_ = np.argsort(P2[..., 0], axis=-1)                              # (n, T, 2, 2) -> trié par x
    pos = torch.from_numpy(np.take_along_axis(P2, o_[..., None], axis=2).reshape(a.n, T, 4))   # disques indiscernables : gauche puis droite
    L, R = w["A"][:, :, 0::2].sum((2, 3)), w["A"][:, :, 1::2].sum((2, 3)); ild = torch.from_numpy(L - R)
    imp = torch.from_numpy(w["IMP"])
    for name, enc in [("init", m0.enc), ("run", tgt)]:
        Z = encode(enc, tok, dev)                                                   # (n, N, d)
        # (a) perception : tokens visuels de la frame t -> positions des 2 disques à t
        Zf = Z[:, :nv].reshape(a.n, T, npf, -1); yf = pos
        p_ = fit_reg(Zf[tr].flatten(0, 1), yf[tr].flatten(0, 1), Zf[te].flatten(0, 1), dev, a.steps)
        ra = r2(p_, yf[te].flatten(0, 1)); err = float((p_ - yf[te].flatten(0, 1)).abs().mean()) * 32
        # (c) examen allégé : 16 tokens (moyenne vision de la frame ‖ token audio) ; stéréo correcte vs inversée
        lab = (torch.arange(a.n) % 2).long(); Asw = A.clone(); Asw[lab == 1] = A[lab == 1][:, :, [1, 0, 3, 2]]
        toksw = torch.cat([to_tokens(X[i:i + 250].to(dev), Asw[i:i + 250].to(dev), a.P, st).half().cpu() for i in range(0, a.n, 250)])
        Zs = encode(enc, toksw, dev); comp = torch.cat([Zs[:, :nv].reshape(a.n, T, npf, -1).float().mean(2), Zs[:, nv:].float()], -1)
        pc = fit_reader(comp[:2000], lab[:2000], comp[2000:], "bin", dev, a.steps).argmax(-1); rc = bacc(pc, lab[2000:])
        msg = f"  {name:>4s} | (a) positions R² {ra:+.2f} (~{err:.1f} px) | (c) localisation résumée {rc:.0%}"
        if name == "run":
            # (b) objectif : latent audio PRÉDIT depuis toute la vision vs VRAI latent audio -> signe G-D aux impacts
            N = nv + T; vis = torch.arange(nv, device=dev); aud = torch.arange(nv, N, device=dev); preds = []
            with torch.no_grad():
                for i in range(0, a.n, 64):
                    o = tok[i:i + 64].to(dev).float(); B = len(o)
                    preds.append(m.pred(m.enc(_gather(o, vis.expand(B, -1)), vis.expand(B, -1)), vis.expand(B, -1), aud.expand(B, -1)).half().cpu())
            Pa = torch.cat(preds); Ta = Z[:, nv:]                                       # (n, T, d) prédit / vrai
            msk = imp & (ild.abs() > 2); y = (ild > 0).long()
            def side(Za):
                Xi, yi = Za[msk].unsqueeze(1), y[msk]; k = int(0.7 * len(yi))
                return bacc(fit_reader(Xi[:k], yi[:k], Xi[k:], "bin", dev, a.steps).argmax(-1), yi[k:])
            # (d) les OREILLES SEULES : encodeur sur les seuls tokens audio -> côté du choc (G/D) ?
            with torch.no_grad():
                Za = torch.cat([tgt(tok[i:i + 64, nv:].to(dev).float(), aud.expand(len(tok[i:i + 64]), -1)).half().cpu() for i in range(0, a.n, 64)])
            toksw_all = torch.cat([to_tokens(X[i:i + 250].to(dev), A[i:i + 250, :, [1, 0, 3, 2]].to(dev), a.P, st).half().cpu() for i in range(0, a.n, 250)])
            s_all, s_choc = surprise(m.pred, m.enc, tgt, tok, toksw_all, imp, nv, dev)
            msg += (f" | (b) côté du son : vrai latent {side(Ta):.0%} / PRÉDIT depuis l'image {side(Pa):.0%}"
                    f" | (d) oreilles seules {side(Za):.0%} | (e) SURPRISE stéréo inversée {s_all:.0%} (chocs {s_choc:.0%}, 0 étiq.)")
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
    print("Lecture : (a) bas -> la vision ne localise pas ; (b) prédit ≈ 50 % alors que vrai haut -> l'objectif ignore la stéréo ;"
          " (c) haut -> c'était l'examen.")

if __name__ == "__main__":
    main()
