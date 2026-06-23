"""
Jouet conduite 2D — PHASE 2 : world model JEPA + sonde macro "conflit ?".

Fidele LeCun :
  - observation = IMAGE ego-centree (pixels), JAMAIS l'etat verite-terrain.
  - world model = encodeur + predicteur, predit les latents des frames MASQUEES,
    anti-collapse = SIGReg (package lejepa), pas d'EMA / stop-grad / reconstruction.
  - eval = ANTICIPATION : predire le conflit depuis la 1re MOITIE du vol seulement
    (avant qu'il arrive), via une sonde gelee. vs baseline sur pixels bruts.

  python drive_model.py --steps 1500 --n_train 4000
"""
import argparse, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import drive

# --------------------------- SIGReg officiel --------------------------------
_LE = None
def sigreg(z):
    global _LE
    if _LE is None:
        import lejepa
        _LE = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(t_max=3, n_points=17), num_slices=1024)
        try: _LE = _LE.to(z.device)
        except Exception: pass
    return _LE(z)

# --------------------------- rendu image ego-centree ------------------------
def render(ego_x, ped_x, ped_y, G):
    """(n,T) -> images (n,T,G*G). Vue top-down centree sur l'ego : x relatif a l'ego."""
    XLO, XHI, YLO, YHI = -1.0, 7.0, -1.6, 1.6
    xr = ped_x - ego_x                                        # (n,T) position relative du pieton
    col = (xr - XLO) / (XHI - XLO) * G
    row = (YHI - ped_y) / (YHI - YLO) * G
    idx = torch.arange(G).float()
    ry = YHI - (idx + 0.5) / G * (YHI - YLO)                  # world-y du centre de chaque ligne
    road = (ry.abs() < 0.9).float().view(1, 1, G, 1) * 0.3    # bande route (constante)
    n, T = xr.shape
    img = road.expand(n, T, G, G).clone()
    dcol = idx.view(1, 1, 1, G) - col.unsqueeze(-1).unsqueeze(-1)
    drow = idx.view(1, 1, G, 1) - row.unsqueeze(-1).unsqueeze(-1)
    blob = torch.exp(-(dcol ** 2 + drow ** 2) / (2 * 1.1 ** 2))
    inwin = ((xr > XLO) & (xr < XHI) & (ped_y > YLO) & (ped_y < YHI)).float().unsqueeze(-1).unsqueeze(-1)
    img = torch.clamp(img + blob * inwin, 0, 1)
    return img.reshape(n, T, G * G)

def make_data(n, env, G, seed):
    rng = np.random.default_rng(seed)
    eps = [drive.gen_episode(rng, env) for _ in range(n)]
    ego = torch.tensor([e["ego_x"] for e in eps]).float()
    pedy = torch.tensor([e["ped_y"] for e in eps]).float()
    pedx = torch.tensor([[e["ped_x"]] for e in eps]).float()
    obs = render(ego, pedx, pedy, G)
    y = torch.tensor([1 if e["conflict"] else 0 for e in eps]).long()
    return obs, y, eps

# --------------------------- world model (JEPA) -----------------------------
class Enc(nn.Module):
    def __init__(s, obs, d, T, nl, nh):
        super().__init__()
        s.emb = nn.Linear(obs, d); s.pos = nn.Embedding(T, d)
        s.mtok = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d)
    def forward(s, o, m=None):
        e = s.emb(o)
        if m is not None: e = torch.where(m.unsqueeze(-1), s.mtok, e)
        x = e + s.pos(torch.arange(o.size(1), device=o.device))
        return s.ln(s.tr(x))                                  # (B,T,d)

class WM(nn.Module):
    def __init__(s, obs, d, T, nl, nh, regw):
        super().__init__()
        s.enc = Enc(obs, d, T, nl, nh); s.regw = regw
        s.pred = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(s, o, mask_ratio):
        B, T, _ = o.shape
        m = torch.rand(B, T, device=o.device) < mask_ratio
        m[:, 0] = False                                       # garde au moins la 1re frame
        hc = s.enc(o, m); z = s.enc(o, None)
        p = s.pred(hc)
        pred = F.smooth_l1_loss(p[m], z[m]) if m.any() else (p - z).pow(2).mean()
        return pred + s.regw * sigreg(z.reshape(-1, z.size(-1)))
    @torch.no_grad()
    def context(s, o, half):                                 # repr de la 1re moitie (2e moitie masquee)
        m = torch.zeros(o.size(0), o.size(1), dtype=torch.bool, device=o.device); m[:, half:] = True
        return s.enc(o, m).reshape(o.size(0), -1)

# --------------------------- sonde / eval -----------------------------------
def probe(Xtr, ytr, Xte, yte, dev, steps=400):
    clf = nn.Linear(Xtr.size(1), 2).to(dev); opt = torch.optim.Adam(clf.parameters(), 1e-2)
    w = torch.tensor([1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())], device=dev)  # classe rare ponderee
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(clf(Xtr), ytr, weight=w).backward(); opt.step()
    with torch.no_grad():
        pr = clf(Xte).argmax(-1)
    acc = (pr == yte).float().mean().item()
    rec_pos = ((pr == 1) & (yte == 1)).sum().item() / max(1, (yte == 1).sum().item())
    rec_neg = ((pr == 0) & (yte == 0)).sum().item() / max(1, (yte == 0).sum().item())
    return acc, 0.5 * (rec_pos + rec_neg), rec_pos

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--G", type=int, default=24); p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=3); p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--reg_w", type=float, default=25.0)
    p.add_argument("--n_train", type=int, default=4000); p.add_argument("--n_test", type=int, default=1500)
    p.add_argument("--p_cross", type=float, default=0.6); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump_n", type=int, default=0, help="dump N episodes test avec proba conflit predite")
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    env = argparse.Namespace(T=24, dt=1.0, ego_speed=0.5, cw_x=6.0, road=0.9, p_cross=a.p_cross, cross_dur=6)
    T = env.T; half = T // 2; obs_dim = a.G * a.G
    tr_o, tr_y, _ = make_data(a.n_train, env, a.G, 1000 + a.seed)
    te_o, te_y, te_eps = make_data(a.n_test, env, a.G, 99999)
    print(f"device={dev}  conduite G={a.G} T={T}  conflits train={tr_y.float().mean():.0%} test={te_y.float().mean():.0%}", flush=True)
    torch.manual_seed(a.seed)
    m = WM(obs_dim, a.d_model, T, a.n_layer, a.n_head, a.reg_w).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr); m.train()
    for st in range(a.steps):                                # SSL : apprendre le world model (sans labels)
        o = tr_o[torch.randint(0, a.n_train, (a.bs,))].to(dev)
        loss = m(o, a.mask_ratio); opt.zero_grad(); loss.backward(); opt.step()
        if st % 300 == 0: print(f"  step {st}  loss {loss.item():.3f}", flush=True)
    # ANTICIPATION : conflit depuis la 1re moitie seulement
    Xtr = m.context(tr_o.to(dev), half); Xte = m.context(te_o.to(dev), half)
    acc, bal, rec = probe(Xtr, tr_y.to(dev), Xte, te_y.to(dev), dev)
    rawtr = tr_o[:, :half].reshape(a.n_train, -1).to(dev); rawte = te_o[:, :half].reshape(a.n_test, -1).to(dev)
    racc, rbal, rrec = probe(rawtr, tr_y.to(dev), rawte, te_y.to(dev), dev)
    maj = max(te_y.float().mean().item(), 1 - te_y.float().mean().item())
    print(f"\n=== ANTICIPATION du conflit (1re moitie seulement) ===", flush=True)
    print(f"  WORLD MODEL : acc {acc:.2f}  bal-acc {bal:.2f}  rappel-conflit {rec:.2f}", flush=True)
    print(f"  pixels bruts: acc {racc:.2f}  bal-acc {rbal:.2f}  rappel-conflit {rrec:.2f}", flush=True)
    print(f"  plancher (classe majoritaire) : acc {maj:.2f}  bal-acc 0.50", flush=True)
    if a.dump_n:
        clf = nn.Linear(Xtr.size(1), 2).to(dev); opt2 = torch.optim.Adam(clf.parameters(), 1e-2)
        w = torch.tensor([1.0, (tr_y == 0).sum() / max(1, (tr_y == 1).sum())], device=dev)
        for _ in range(400):
            opt2.zero_grad(); F.cross_entropy(clf(Xtr), tr_y.to(dev), weight=w).backward(); opt2.step()
        with torch.no_grad(): prob = clf(Xte).softmax(-1)[:, 1].cpu()
        out = [{"ego_x": te_eps[k]["ego_x"], "ped_x": te_eps[k]["ped_x"], "ped_y": te_eps[k]["ped_y"],
                "conflict": te_eps[k]["conflict"], "p_conf": round(float(prob[k]), 2)} for k in range(a.dump_n)]
        print("DRIVE2_DUMP " + json.dumps(out), flush=True)

if __name__ == "__main__":
    main()
