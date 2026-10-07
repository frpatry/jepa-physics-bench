"""
DIAG — pourquoi l'encodeur bébé JETTE-t-il la main ? (lecteur de positions = instrument, comme l'examen)
Pour chaque instantané : R² disques / MAIN dans le monde avec main, selon le comportement de la main :
  script (viser-pousser à fond) | babillage | main IMMOBILE (rien ne bouge) ; + référence PIXELS bruts.
Lecture : main lisible quand immobile mais pas quand elle bouge -> l'encodeur jette ce qui est IMPRÉVISIBLE
(remède : donner la commande motrice) ; illisible même immobile -> objet NOUVEAU / trop petit (remède :
saillance, plus d'exposition).

  python av_hand_diag.py --ckpts /content/drive/MyDrive/jepa_runs/av_dev_v4_sep_40k.pt,/content/drive/MyDrive/jepa_runs/av_dev_v5_hand_50k.pt
"""
import argparse, copy, time
import numpy as np, torch
from av_jepa import gen_world, NB, enc_config
from av_act import gen_world_act
from av_world5 import gen_world_v5
from av_dev import DevJEPA
import torch.nn.functional as F
from vjepa import _gather
from av_dev_long import to_tokens, stereo, loc_r2, _PosReader, T, H

@torch.no_grad()
def encode(enc, tok, mode, nv, npf, dev, bs=64):
    """full = séquence entière (examen) | causal = frames 0..t (phase 2) | frame = la frame t SEULE -> (n, T, npf, d)"""
    out = []
    for i in range(0, len(tok), bs):
        o = tok[i:i + bs].to(dev).float(); B = len(o); zs = []
        if mode == "full":
            idx = torch.arange(o.size(1), device=dev).expand(B, -1); z = enc(o, idx)[:, :nv]
            out.append(z.reshape(B, T, npf, -1).half().cpu()); continue
        for t in range(T):
            if mode == "causal": idx = torch.cat([torch.arange((t + 1) * npf), nv + torch.arange(t + 1)]).to(dev)
            else: idx = torch.cat([torch.arange(t * npf, (t + 1) * npf), torch.tensor([nv + t])]).to(dev)
            idx = idx.expand(B, -1); z = enc(_gather(o, idx), idx)
            zs.append(z[:, t * npf:(t + 1) * npf] if mode == "causal" else z[:, :npf])
        out.append(torch.stack(zs, 1).half().cpu())
    return torch.cat(out)

def fit_r2(Z, y, dev, steps, groups):
    Z = Z.flatten(0, 1); y = y.flatten(0, 1); k = int(0.7 * len(Z)); mu, sd = y[:k].mean(0), y[:k].std(0) + 1e-6
    torch.manual_seed(0); r = _PosReader(Z.size(-1), Z.size(1), nout=y.size(-1)).to(dev); opt = torch.optim.AdamW(r.parameters(), 3e-4, weight_decay=0.05)
    for _ in range(steps):
        bi = torch.randint(0, k, (256,)); l = F.mse_loss(r(Z[bi].to(dev).float()), ((y[bi] - mu) / sd).to(dev)); opt.zero_grad(); l.backward(); opt.step()
    r.eval()
    with torch.no_grad(): p = torch.cat([r(Z[i:i + 512].to(dev).float()).cpu() for i in range(k, len(Z), 512)]) * sd + mu
    yt = y[k:]; return [float(1 - ((p[:, c] - yt[:, c]) ** 2).sum() / ((yt[:, c] - yt[:, c].mean(0)) ** 2).sum()) for c in groups]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=str, required=True); p.add_argument("--sep_enc", type=int, default=1)
    p.add_argument("--modes", type=str, default="full,causal,frame"); p.add_argument("--hum", type=float, default=0.15); p.add_argument("--n", type=int, default=1500); p.add_argument("--steps", type=int, default=1500)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    P = 4; nP = H // P; nv = T * nP * nP; da = 2 * 2 * NB; W = max(P * P * 3, da)
    w0 = gen_world(2000, T, H, seed=0, a_sub=2, hum=a.hum); A0 = stereo(torch.from_numpy(w0["A"])).reshape(2000, T, -1)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev)); del w0
    probes = {}
    for name, b in [("script", 0), ("babillage", 1), ("main immobile", 2)]:
        w = gen_world_act(a.n, T, H, seed=1001, a_sub=2, hum=a.hum, fric=0.02, babble=b)
        X, A = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)), torch.from_numpy(w["A"])
        P2 = w["POS"]; o_ = np.argsort(P2[..., 0], axis=-1)
        y = np.concatenate([np.take_along_axis(P2, o_[..., None], axis=2).reshape(a.n, T, 4), w["HAND"]], -1)
        tok = torch.cat([to_tokens(X[i:i + 250].to(dev), A[i:i + 250].to(dev), P, st).half().cpu() for i in range(0, a.n, 250)])
        mv = np.linalg.norm(np.diff(w["HAND"], axis=1), axis=-1).mean() * 32
        probes[name] = dict(htok=tok, hpos=torch.from_numpy(y).float(), G=[slice(0, 4), slice(4, 6)]); print(f"monde {name:>14s} : main bouge {mv:.2f} px/frame ({time.time() - t0:.0f}s)", flush=True)
    # sonde du T (monde v5, UN seul T parmi d'autres formes) : position (2) | orientation (cos, sin)
    wt = gen_world_v5(a.n, T, H, seed=1003, a_sub=2, hum=a.hum, force_T=True)
    Xt, At = torch.from_numpy((wt["X"] * 255).round().astype(np.uint8)), torch.from_numpy(wt["A"])
    tt = torch.cat([to_tokens(Xt[i:i + 250].to(dev), At[i:i + 250].to(dev), P, st).half().cpu() for i in range(0, a.n, 250)])
    yt = np.concatenate([wt["POS"][:, :, 0], np.cos(wt["ANG"][:, :, :1]), np.sin(wt["ANG"][:, :, :1])], -1)
    probes["T (pos | orient.)"] = dict(htok=tt, hpos=torch.from_numpy(yt).float(), G=[slice(0, 2), slice(2, 4)])
    print(f"\n{'encodeur':>28s} | " + " | ".join(f"{n_:>26s}" for n_ in probes)); print(" " * 28 + " | " + " | ".join(f"{'objets R²':>12s} {'MAIN/orient. R²':>13s}" for _ in probes))
    rows = [("PIXELS bruts (référence)", None)] + [(c.split("/")[-1][-12:] + f" [{md}]", c, md) for c in a.ckpts.split(",") for md in a.modes.split(",")]
    rows[0] = rows[0] + ("-",)
    for name, c, md in rows:
        if c is None: m = None
        else:
            m0 = DevJEPA(W, da, nv, T, 192, 6, 6, 3).to(dev); enc = copy.deepcopy(m0.enc)
            ck = torch.load(c, map_location=dev, weights_only=False); enc.load_state_dict(ck["tgt"]); enc.eval(); enc_config(enc, ck, a.sep_enc)
            m = type("W", (), {"enc": enc})()
        res = []
        for pr in probes.values():              # m None : patches bruts (represent renvoie les tokens tels quels)
            if m is None: res.append(loc_r2(m, pr, nv, dev, a.steps, tok_key="htok", y_key="hpos", groups=pr["G"]))
            else: res.append(fit_r2(encode(m.enc, pr["htok"], md, nv, nP * nP, dev), pr["hpos"], dev, a.steps, pr["G"]))
        print(f"{name:>28s} | " + " | ".join(f"{d:+12.2f} {h:+13.2f}" for d, h in res) + f"  ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
