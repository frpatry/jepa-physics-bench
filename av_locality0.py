"""
DIAGNOSTIC DE LOCALITÉ de l'œil (aucun entraînement du bébé) : chaque carré de vue décrit-il ce qui est À SON ENDROIT
(représentation dense, ce que vise V-JEPA 2.1), ou un résumé GLOBAL de la scène (hypothèse : la perte JEPA ne note que les
carrés cachés, les carrés visibles sont libres de devenir globaux, et la main domine) ?

Sur des séquences à UN objet du monde 0e, pour chaque image on prend trois carrés de vue (latents cibles normalisés) :
  SUR L'OBJET (le carré qui contient son centre) | SUR LA MAIN | LOIN (à ≥ --far carrés de l'objet ET de la main).
Sondes linéaires (ridge, instruments), séparées par séquence (80 % / 20 %) :
  1 position de l'OBJET lue dans un carré LOIN       -> local si ≈ hasard (prédire la moyenne)
  2 position de la MAIN lue dans un carré LOIN       -> la main est-elle partout ?
  3 COULEUR de l'objet (R²) : carré sur l'objet vs carré loin -> l'apparence est-elle à son endroit ?
  4 position de l'objet lue dans le carré de la MAIN -> contamination main/objet
  5 repères : position de l'objet lue dans le carré SUR l'objet, et dans la MOYENNE de tous les carrés
Références : un œil ALÉATOIRE (même architecture, non entraîné) et les PIXELS bruts du carré.

  python av_locality0.py --ckpts phase0_0e_60k.pt,phase0_r5_0e_60k.pt,phase0_r5x_0e_100k.pt --dir /content/drive/MyDrive/jepa_runs
"""
import argparse, copy, os, time
import numpy as np, torch, torch.nn.functional as F
from av_world0 import gen_world0
import av_phase0
from av_phase0 import baby_from_cfg, to_tok0, to_np, to_torch, VIEW, H


def ridge(Xtr, Ytr, Xte, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """ridge sur variables standardisées ; λ choisi sur 20 % du jeu d'entraînement. -> prédictions sur Xte."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0); Ytr = Ytr - ym; nv = len(Xtr) // 5
    def fit(X, Y, lam):
        A = X.T @ X + lam * len(X) * torch.eye(X.size(1), device=X.device, dtype=X.dtype); return torch.linalg.solve(A, X.T @ Y)
    best = min(lams, key=lambda l: float(((Xtr[:nv] @ fit(Xtr[nv:], Ytr[nv:], l) - Ytr[:nv]) ** 2).mean()))
    return Xte @ fit(Xtr, Ytr, best) + ym


def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True, help="instantanés séparés par des virgules (noms relatifs à --dir) ; « init » = œil aléatoire")
    p.add_argument("--dir", default="/content/drive/MyDrive/jepa_runs"); p.add_argument("--n", type=int, default=2000)
    p.add_argument("--far", type=int, default=3, help="distance minimale (en carrés, max des écarts x/y) du carré LOIN à l'objet et à la main")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    names = [c.strip() for c in a.ckpts.split(",")]
    ref = next((c for c in names if c != "init"), None)
    cks = {c: torch.load(os.path.join(a.dir, c), map_location=dev, weights_only=False) for c in names if c != "init"}
    cfg0 = cks[ref]["cfg"] if ref else None
    T = int(cfg0.get("T", 16)); av_phase0.T = T; P = cfg0["P"]; nP = H // P; npf = nP * nP; nv = T * npf; v = VIEW["0e"]
    # ---------- données : un seul monde pour tous les yeux (celui du premier instantané)
    w = gen_world0(a.n, "0e", T, H, seed=4242 + a.seed, **cfg0.get("wkw", {}))
    keep = np.where(w["NOBJ"] == 1)[0]; b = to_torch(to_np([{k: w[k][keep] for k in ("X", "A", "TOUCH", "PROP", "CMD")}])); n = len(keep)
    obj, hand = w["POS"][keep, :, 0], w["HAND"][keep]                                            # (n, T, 2) x,y dans [0,1]
    ok = np.isfinite(obj).all(-1) & (obj > 0).all(-1) & (obj < 1).all(-1) & (hand > 0).all(-1) & (hand < 1).all(-1)
    cell = lambda q: (np.clip((q[..., 1] * nP).astype(int), 0, nP - 1), np.clip((q[..., 0] * nP).astype(int), 0, nP - 1))   # (ligne, colonne)
    (ro, co), (rh, ch) = cell(np.nan_to_num(obj)), cell(hand)
    ok &= np.maximum(abs(ro - rh), abs(co - ch)) >= 1                                            # objet et main dans des carrés différents
    rng = np.random.default_rng(a.seed); far = np.full((n, T), -1)
    gr, gc = np.mgrid[0:nP, 0:nP]
    for i in range(n):
        for t in range(T):
            if not ok[i, t]: continue
            c_ = np.where((np.maximum(abs(gr - ro[i, t]), abs(gc - co[i, t])) >= a.far) & (np.maximum(abs(gr - rh[i, t]), abs(gc - ch[i, t])) >= a.far))
            if len(c_[0]): j = rng.integers(len(c_[0])); far[i, t] = c_[0][j] * nP + c_[1][j]
    ok &= far >= 0
    X = w["X"][keep]; col = X[np.arange(n)[:, None], np.arange(T)[None], (np.nan_to_num(obj[..., 1]) * H).astype(int).clip(0, H - 1),
                            (np.nan_to_num(obj[..., 0]) * H).astype(int).clip(0, H - 1)]           # couleur au centre de l'objet (n, T, 3)
    ii, tt = np.where(ok); split = ii < int(0.8 * n)
    idx = dict(objet=ro[ii, tt] * nP + co[ii, tt], main=rh[ii, tt] * nP + ch[ii, tt], loin=far[ii, tt])
    Yo, Yh, Yc = [torch.from_numpy(q[ii, tt]).float().to(dev) for q in (np.nan_to_num(obj), hand, col)]
    print(f"{n} séquences à 1 objet, {len(ii)} images retenues (objet, main et carré LOIN distincts ; loin ≥ {a.far} carrés) | {time.time() - t0:.0f}s", flush=True)

    def report(tag, F_):                       # F_ : dict lieu -> (N, dim) features des images retenues (+ « moyenne »)
        tr, te = torch.from_numpy(split).to(dev), torch.from_numpy(~split).to(dev)
        def pos_err(Fx, Y):
            pr = ridge(Fx[tr], Y[tr], Fx[te]); return float((pr - Y[te]).norm(dim=-1).mean() * 32)
        ch_o, ch_h = float((Yo[te] - Yo[tr].mean(0)).norm(dim=-1).mean() * 32), float((Yh[te] - Yh[tr].mean(0)).norm(dim=-1).mean() * 32)
        c_on = r2(ridge(F_["objet"][tr], Yc[tr], F_["objet"][te]), Yc[te]); c_far = r2(ridge(F_["loin"][tr], Yc[tr], F_["loin"][te]), Yc[te])
        print(f"  {tag:>28s} | 1 objet lu LOIN {pos_err(F_['loin'], Yo):5.2f} px (hasard {ch_o:.2f}) | 2 main lue LOIN {pos_err(F_['loin'], Yh):5.2f} px "
              f"(hasard {ch_h:.2f}) | 3 couleur R² sur l'objet {c_on:+.2f} / loin {c_far:+.2f} | 4 objet lu SUR LA MAIN {pos_err(F_['main'], Yo):5.2f} px "
              f"| 5 objet lu SUR L'OBJET {pos_err(F_['objet'], Yo):5.2f} px, dans la MOYENNE {pos_err(F_['moyenne'], Yo):5.2f} px", flush=True)

    # ---------- référence : pixels bruts du carré (vue 0e : nette, en couleur)
    tok, _ = to_tok0({k: x[:1] for k, x in b.items()}, cfg0 and cks[ref]["norm"], v, P, dev)
    pix = []
    for i0 in range(0, n, 100):
        tk, _ = to_tok0({k: x[i0:i0 + 100] for k, x in b.items()}, cks[ref]["norm"], v, P, dev); pix.append(tk[:, :nv].float().view(-1, T, npf, tk.size(-1)).cpu())
    pix = torch.cat(pix)[..., :P * P * 3]
    sel = lambda Z, k: Z[ii, tt, torch.from_numpy(idx[k])].float().to(dev)
    report("PIXELS bruts du carré", {**{k: sel(pix, k) for k in idx}, "moyenne": pix[ii, tt].mean(1).float().to(dev)})
    # ---------- chaque œil
    for c in names:
        if c == "init":
            torch.manual_seed(a.seed); m = baby_from_cfg(cfg0, dev); enc = m.enc.eval(); st = cks[ref]["norm"]; d = cfg0["d"]; lab = "œil ALÉATOIRE (init)"
        else:
            ck = cks[c]; cfg = ck["cfg"]; st = ck["norm"]; d = cfg["d"]; m = baby_from_cfg(cfg, dev); m.load_state_dict(ck["m"])
            enc = copy.deepcopy(m.enc); enc.load_state_dict(ck["tgt"]); enc.eval(); lab = f"{c.replace('.pt', '')} (pas {ck['state']['it']})"
        Z = []
        with torch.no_grad():
            for i0 in range(0, n, 50):
                tk, _ = to_tok0({k: x[i0:i0 + 50] for k, x in b.items()}, st, v, P, dev); ix = torch.arange(nv, device=dev).expand(len(tk), -1)
                Z.append(F.layer_norm(enc(tk[:, :nv], ix).float(), (d,)).view(len(tk), T, npf, d).half().cpu())
        Z = torch.cat(Z)
        report(lab, {**{k: sel(Z, k) for k in idx}, "moyenne": Z[ii, tt].float().mean(1).to(dev)})
        del Z, m, enc; torch.cuda.empty_cache() if dev == "cuda" else None
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
