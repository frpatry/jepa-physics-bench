"""
Banc anti-collapse DANS un JEPA (style LeJEPA), sur données PHYSIQUE jouet.

Structure JEPA cohérente partout (comme AMI Labs / LeJEPA) :
  contexte (vue MASQUÉE) -> encodeur -> PRÉDICTEUR -> p
  cible    (vue PROPRE)  -> encodeur ------------- -> z
  perte de prédiction = smooth-L1(p, z)   [PAS d'EMA, PAS de stop-grad]
  + terme ANTI-COLLAPSE sur z (c'est ce qu'on compare) :
     - none   : aucun -> doit COLLAPSER (témoin)
     - vicreg : variance (std>=1) + covariance (décorrélation)   [2 premiers moments]
     - sigreg : SIGReg EXACT (LeJEPA, Balestriero & LeCun) = test EPPS-PULLEY
                (fonction caractéristique) sur projections 1D aléatoires -> pousse
                z vers une GAUSSIENNE ISOTROPE. Pas d'EMA/stop-grad/scheduler.

Données : projectile sous gravité, g varie -> SONDE sur g = la rep a-t-elle capté
la physique. Métriques : accuracy sonde-g, rang effectif (anti-collapse), × budget
de données × graines.

  python cl_sslbench.py --regs none vicreg sigreg --n_trains 300 1000 3000 --seeds 3
"""
import argparse, json, os, statistics, torch, torch.nn as nn, torch.nn.functional as F

# ---- briques inlinees (banc autonome, aucune dependance projet) ------------
def make_mask(B, T, ratio, gen):
    r = torch.rand(B, T, generator=gen)
    nmask = max(1, int(ratio * T))
    idx = r.topk(nmask, dim=1).indices
    m = torch.zeros(B, T, dtype=torch.bool); m.scatter_(1, idx, True)
    return m

def effective_rank(Z):
    """participation ratio des valeurs propres de la cov des representations,
    dans [1, d] : haut = dims decorrelees utilisees (fin), bas = collapse."""
    Z = Z - Z.mean(0, keepdim=True)
    C = (Z.t() @ Z) / max(1, Z.size(0))
    ev = torch.linalg.eigvalsh(C).clamp(min=1e-12)
    return (ev.sum() ** 2 / (ev ** 2).sum()).item()

class Block(nn.Module):
    """bloc Transformer standard (attn bidirectionnel + FFN), pre-LN."""
    def __init__(self, a):
        super().__init__()
        self.ln1 = nn.LayerNorm(a.d_model); self.ln2 = nn.LayerNorm(a.d_model)
        self.attn = nn.MultiheadAttention(a.d_model, a.n_head, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(a.d_model, a.d_ff), nn.ReLU(),
                                 nn.Linear(a.d_ff, a.d_model))
    def forward(self, x, _ctx=None):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + h
        return x + self.ffn(self.ln2(x))

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regs", nargs="+", default=["none", "vicreg", "sigreg"])
    p.add_argument("--n_trains", type=int, nargs="+", default=[300, 1000, 3000])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--reg_w", type=float, default=25.0, help="poids du terme anti-collapse (seul hyperparam ; ~25 = échelle VICReg)")
    p.add_argument("--n_proj", type=int, default=1024, help="projections sketchées SIGReg (officiel LeJEPA = 1024)")
    p.add_argument("--T", type=int, default=16); p.add_argument("--obs", type=int, default=32)
    p.add_argument("--n_gbins", type=int, default=6)
    p.add_argument("--nuisance", type=float, default=1.0,
                   help="echelle des params parasites x0/y0/vx0/vy0 ; <1 = g domine = tache + apprenable ; =1 reproduit l'original")
    p.add_argument("--nonstat", type=float, default=0.0,
                   help="derive STOCHASTIQUE de g par sequence (s~N(0,.) independant de g0) ; 0=stationnaire (theoreme LeJEPA), >0 casse l'identifiabilite (g0 n'est plus recuperable)")
    p.add_argument("--bounce", type=float, default=0.0,
                   help="REBOND au sol (coeff de restitution) ; 0=desactive (parabole), >0=la balle rebondit (cassures = TRANSITIONS DE PHASE, cas d'echec nomme par la theorie). g RESTE recuperable en principe.")
    p.add_argument("--wind", type=float, default=0.0,
                   help="VENT : rafales aleatoires (force exterieure imprevisible) ajoutees a la vitesse a CHAQUE instant. 0=calme, >0=turbulent. Realiste et non-stationnaire (vs gravite magique). Combinable avec --bounce.")
    p.add_argument("--vision", action="store_true",
                   help="MODE VISION : la balle est DESSINEE en image grid x grid (blob gaussien) a chaque instant = mini-video, au lieu de coordonnees projetees. Plus riche et redondant (facon V-JEPA).")
    p.add_argument("--grid", type=int, default=16, help="cote de l'image en mode vision (grid x grid pixels)")
    p.add_argument("--vnoise", type=float, default=0.0,
                   help="BRUIT VISUEL : grain ajoute aux PIXELS (corruption de l'image, pas de la trajectoire). Different du vent (qui perturbe le mouvement).")
    p.add_argument("--readout", choices=["pool", "time"], default="pool",
                   help="pool=moyenne temporelle (ancien, ecrase le temps) ; time=fidele au temps (par instant, predit les instants masques, sonde sur toute la trajectoire) comme V-JEPA")
    p.add_argument("--oracle", action="store_true",
                   help="PLAFOND supervise : sonde directe sur donnees brutes (zero SSL). Repond : g est-il recuperable de ces donnees ? lin(traj)=identif. lineaire ; lin(moyenne)=ce que pool peut au mieux ; MLP(traj)=recuperable du tout ?")
    p.add_argument("--d_model", type=int, default=128); p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--n_head", type=int, default=4); p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--k", type=int, default=64); p.add_argument("--route_dim", type=int, default=32)
    p.add_argument("--mode", default="dense"); p.add_argument("--seq", type=int, default=16)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--probe_steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", type=str, default="runs_ssl")
    return p.parse_args()

# --------------------------- données physique -------------------------------
def gen_physics(n, a, gen):
    gvals = torch.linspace(5.0, 15.0, a.n_gbins)
    gi = torch.randint(0, a.n_gbins, (n,), generator=gen); g = gvals[gi]
    nz = a.nuisance                                          # echelle des parasites (1.0 = original)
    x0 = (torch.rand(n, generator=gen) * 2 - 1) * nz
    y0 = torch.rand(n, generator=gen) * nz
    vx0 = (torch.rand(n, generator=gen) * 2 - 1) * nz
    vy0 = 2.0 + (torch.rand(n, generator=gen) * 2 - 1) * nz  # moyenne 2, etalement +-nz (nz=1 -> [1,3])
    t = torch.arange(a.T).float() * 0.1
    x = x0[:, None] + vx0[:, None] * t[None, :]
    if a.bounce > 0 or a.wind > 0:                           # SIMULATION pas-a-pas (rebond et/ou vent)
        dt = 0.1
        xp, xv = x0.clone(), vx0.clone()
        yp, yv = y0.clone(), vy0.clone()
        xs, ys = [], []
        for _ in range(a.T):
            xs.append(xp.clone()); ys.append(yp.clone())
            if a.wind > 0:                                   # rafales aleatoires (force exterieure imprevisible)
                xv = xv + a.wind * torch.randn(n, generator=gen) * dt
                yv = yv + a.wind * torch.randn(n, generator=gen) * dt
            xp = xp + xv * dt
            yp = yp + yv * dt
            yv = yv - g * dt                                 # gravite
            if a.bounce > 0:                                 # rebond au sol (transition de phase)
                below = yp < 0
                yp = torch.where(below, -yp, yp)
                yv = torch.where(below, -a.bounce * yv, yv)
        x = torch.stack(xs, dim=1); y = torch.stack(ys, dim=1)
    elif a.nonstat > 0:                                      # derive STOCHASTIQUE : a(t)=-(g0 + s t),
        tmax = float(t[-1]) if float(t[-1]) > 0 else 1.0     #   s aleatoire PAR sequence, INDEPENDANT de g0
        z = torch.randn(n, generator=gen)                    #   -> g0 ne determine plus la trajectoire
        drift = (a.nonstat / (6 * tmax)) * g * z             #   -> vraie NON-identifiabilite (pas une reparam)
        y = (y0[:, None] + vy0[:, None] * t[None, :]
             - 0.5 * g[:, None] * (t[None, :] ** 2)
             - drift[:, None] * (t[None, :] ** 3))
    else:                                                    # g constant -> parabole (stationnaire)
        y = y0[:, None] + vy0[:, None] * t[None, :] - 0.5 * g[:, None] * (t[None, :] ** 2)
    if a.vision:                                            # MODE VISION : rendre la balle en image
        G = a.grid                                          #   grid x grid (blob gaussien), fenetre FIXE
        xb0, xb1, yb0, yb1 = -3.0, 3.0, -0.2, 3.0           #   monde fixe (partage train/test)
        gxc = (x - xb0) / (xb1 - xb0) * G                   # (n,T) colonne flottante
        gyc = (yb1 - y) / (yb1 - yb0) * G                   # (n,T) ligne (y vers le haut)
        idx = torch.arange(G).float()
        dcol = idx.view(1, 1, 1, G) - gxc.unsqueeze(-1).unsqueeze(-1)   # (n,T,1,G)
        drow = idx.view(1, 1, G, 1) - gyc.unsqueeze(-1).unsqueeze(-1)   # (n,T,G,1)
        img = torch.exp(-(dcol ** 2 + drow ** 2) / (2 * 1.1 ** 2))      # (n,T,G,G) blob
        img = img.reshape(n, a.T, G * G)
        img = img + a.vnoise * torch.randn(n, a.T, G * G, generator=gen)  # bruit VISUEL (pixels)
        return img, gi
    pos = torch.stack([x, y], -1)
    # CAPTEUR FIXE : W partage entre train ET test (sinon le modele est teste dans un
    # espace projete qu'il n'a jamais vu -> tout au hasard). Seed dedie, INDEPENDANT de `gen`.
    W = torch.randn(2, a.obs, generator=torch.Generator().manual_seed(777)) / (2 ** 0.5)
    return pos @ W + 0.05 * torch.randn(n, a.T, a.obs, generator=gen), gi

# --------------------------- termes anti-collapse ---------------------------
def off_diag(M):
    n = M.size(0); return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_reg(z):
    z = z - z.mean(0)
    std = (z.var(0) + 1e-4).sqrt()
    var = F.relu(1 - std).mean()
    cov = (z.T @ z) / (z.size(0) - 1)
    return var + off_diag(cov).pow(2).sum() / z.size(1)

def sigreg_reg(z, n_proj, n_t=17, t_max=5.0):
    """SIGReg EXACT (LeJEPA, Balestriero & LeCun) : test EPPS-PULLEY sur des
    projections 1D aléatoires. Compare la fonction caractéristique empirique de
    chaque projection à celle de N(0,1), pondérée gaussienne :
        EP = ∫ |φ̂_X(t) - e^{-t²/2}|² · e^{-t²/σ²} dt   (quadrature, σ=1)
    Gradient VIVANT au collapse : la fct caractéristique d'un point-masse est
    e^{itc} (module 1), radicalement ≠ de la gaussienne -> pousse à s'étaler.
    Isotrope-gaussien <=> toute projection 1D ~ N(0,1)."""
    V = torch.randn(z.size(1), n_proj, device=z.device); V = V / V.norm(dim=0, keepdim=True)
    P = z @ V                                             # (B, n_proj)
    t = torch.linspace(-t_max, t_max, n_t, device=z.device)
    tX = P.unsqueeze(-1) * t                              # (B, n_proj, n_t)
    re = tX.cos().mean(0) - torch.exp(-0.5 * t ** 2)      # φ̂ réelle - φ_N(0,1)
    im = tX.sin().mean(0)                                 # φ̂ imaginaire (φ_N(0,1) imag = 0)
    w = torch.exp(-(t ** 2))                              # poids gaussien (σ=1)
    return ((re ** 2 + im ** 2) * w).mean()

_LEJEPA_LOSS = None
def sigreg_official(z):
    """SIGReg via le PACKAGE OFFICIEL lejepa (rbalestr-lab/lejepa) -> vraie
    implémentation, plus de réimplémentation maison. `pip install` requis."""
    global _LEJEPA_LOSS
    if _LEJEPA_LOSS is None:
        import lejepa
        _LEJEPA_LOSS = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(t_max=3, n_points=17),
            num_slices=1024)
        try: _LEJEPA_LOSS = _LEJEPA_LOSS.to(z.device)
        except Exception: pass
    return _LEJEPA_LOSS(z)

def reg_term(name, z, a):
    if name == "none":    return z.new_zeros(())
    if name == "vicreg":  return vicreg_reg(z)
    if name == "sigreg":  return sigreg_reg(z, a.n_proj)        # ma réimpl (Epps-Pulley quadrature)
    if name == "sigreg_off": return sigreg_official(z)          # PACKAGE OFFICIEL lejepa
    raise ValueError(name)

# --------------------------- JEPA (prédicteur, sans EMA/stop-grad) ----------
class Encoder(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.embed = nn.Linear(a.obs, a.d_model)
        self.pos = nn.Embedding(a.T, a.d_model)
        self.blocks = nn.ModuleList([Block(a) for _ in range(a.n_layer)])
        self.ln = nn.LayerNorm(a.d_model)
        self.mask_token = nn.Parameter(torch.zeros(a.d_model))
    def forward(self, o, mask=None):
        e = self.embed(o)
        if mask is not None:
            e = torch.where(mask.unsqueeze(-1), self.mask_token, e)
        x = e + self.pos(torch.arange(o.size(1), device=o.device))
        for b in self.blocks: x = b(x, None)
        return self.ln(x)                                 # (B, T, d) -- par instant, PAS de moyenne

class JEPA(nn.Module):
    def __init__(self, a, reg):
        super().__init__()
        d = a.d_model; self.reg = reg
        self.enc = Encoder(a)
        self.predictor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.readout = getattr(a, "readout", "pool")      # "pool" (moyenne, ancien) | "time" (par instant)
    def forward(self, o, a):
        mask = make_mask(o.size(0), o.size(1), a.mask_ratio, None).to(o.device)
        hc = self.enc(o, mask)                             # contexte masqué  (B,T,d)
        z = self.enc(o, None)                              # cible propre (PAS de stop-grad) (B,T,d)
        if self.readout == "time":                         # predit les INSTANTS masqués
            p = self.predictor(hc)
            pred = F.smooth_l1_loss(p[mask], z[mask])
            zr = z.reshape(-1, z.size(-1))                 # (B*T, d) -> reg par instant
        else:                                              # "pool" : moyenne temporelle (ancien comportement)
            p = self.predictor(hc.mean(1))
            zr = z.mean(1)
            pred = F.smooth_l1_loss(p, zr)
        reg = reg_term(self.reg, zr, a)
        return pred + a.reg_w * reg, pred.item(), zr.std(0).mean().item()
    @torch.no_grad()
    def features(self, o):
        h = self.enc(o)                                    # (B,T,d)
        return h.reshape(h.size(0), -1) if self.readout == "time" else h.mean(1)

# --------------------------- entraînement / sonde ---------------------------
def train(model, data, a, dev):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    for _ in range(a.steps):
        o = data[torch.randint(0, data.size(0), (a.bs,))].to(dev)
        loss, _, _ = model(o, a)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def feats(model, o, dev, bs=256):
    return torch.cat([model.features(o[i:i+bs].to(dev)) for i in range(0, o.size(0), bs)])

def probe_acc(model, tr_o, tr_y, te_o, te_y, a, dev):
    X = feats(model, tr_o, dev); y = tr_y.to(dev)
    clf = nn.Linear(X.size(1), a.n_gbins).to(dev); opt = torch.optim.Adam(clf.parameters(), 1e-2)
    for _ in range(a.probe_steps):
        opt.zero_grad(); F.cross_entropy(clf(X), y).backward(); opt.step()
    with torch.no_grad():
        return (clf(feats(model, te_o, dev)).argmax(-1) == te_y.to(dev)).float().mean().item()

# --------------------------- oracle (plafond supervisé) ---------------------
def oracle_ceiling(a, dev):
    """Sonde SUPERVISÉE directe sur les données brutes (aucun SSL) = plafond.
    g est-il récupérable de ces données ? lin(traj) = identifiabilité linéaire ;
    lin(moyenne) = ce que la lecture 'pool' peut au mieux ; MLP(traj) = du tout ?"""
    chance = 1 / a.n_gbins
    te_o, te_y = gen_physics(1500, a, torch.Generator().manual_seed(99999))
    def probe(Xtr, ytr, Xte, yte, mlp):
        Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
        clf = (nn.Sequential(nn.Linear(Xtr.size(1), 256), nn.ReLU(),
                             nn.Linear(256, a.n_gbins)) if mlp
               else nn.Linear(Xtr.size(1), a.n_gbins)).to(dev)
        opt = torch.optim.Adam(clf.parameters(), 1e-2)
        for _ in range(500):
            opt.zero_grad(); F.cross_entropy(clf(Xtr), ytr).backward(); opt.step()
        with torch.no_grad():
            return (clf(Xte).argmax(-1) == yte).float().mean().item()
    print(f"\n=== ORACLE (plafond supervisé) bounce={a.bounce} wind={a.wind} nonstat={a.nonstat} "
          f"chance={chance:.0%} ===", flush=True)
    for ntr in a.n_trains:
        tr_o, tr_y = gen_physics(ntr, a, torch.Generator().manual_seed(1234))
        lin_full = probe(tr_o.reshape(ntr, -1), tr_y, te_o.reshape(te_o.size(0), -1), te_y, False)
        lin_mean = probe(tr_o.mean(1), tr_y, te_o.mean(1), te_y, False)
        mlp_full = probe(tr_o.reshape(ntr, -1), tr_y, te_o.reshape(te_o.size(0), -1), te_y, True)
        print(f"  n={ntr}: lin(traj)={lin_full:.2f}  lin(moyenne)={lin_mean:.2f}  "
              f"MLP(traj)={mlp_full:.2f}", flush=True)

# --------------------------- main -------------------------------------------
def main():
    a = get_args(); a.seq = a.T
    if a.vision: a.obs = a.grid ** 2                          # l'entree devient grid x grid pixels
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available()    # GPU Apple Silicon (Metal)
           else "cpu")
    if a.oracle:                                              # plafond supervisé, puis stop
        oracle_ceiling(a, dev); return
    os.makedirs(a.out, exist_ok=True)
    te_o, te_y = gen_physics(1500, a, torch.Generator().manual_seed(99999))
    print(f"device={dev}  physique T={a.T} obs={a.obs} g_bins={a.n_gbins} "
          f"{'VISION '+str(a.grid)+'x'+str(a.grid)+' vnoise='+str(a.vnoise)+' ' if a.vision else ''}"
          f"(chance sonde={1/a.n_gbins:.0%})  reg_w={a.reg_w}  "
          f"nuisance={a.nuisance}  nonstat={a.nonstat}  bounce={a.bounce}  wind={a.wind}  readout={a.readout} "
          f"({'+'.join([s for s in ['REBOND' if a.bounce>0 else '', 'VENT' if a.wind>0 else '', 'DERIVE' if a.nonstat>0 else ''] if s]) or 'STATIONNAIRE'})\n", flush=True)

    res = {}
    for ntr in a.n_trains:
        for s in range(a.seeds):
            tr_o, tr_y = gen_physics(ntr, a, torch.Generator().manual_seed(1000 + s))
            ptr = int(0.8 * ntr)
            for r in a.regs:
                torch.manual_seed(a.seed + s)
                model = JEPA(a, r).to(dev); train(model, tr_o[:ptr], a, dev)
                acc = probe_acc(model, tr_o[:ptr], tr_y[:ptr], te_o, te_y, a, dev)
                rank = effective_rank(feats(model, te_o[:512], dev))
                res.setdefault((r, ntr), {"acc": [], "rank": []})
                res[(r, ntr)]["acc"].append(acc); res[(r, ntr)]["rank"].append(rank)
            print(f"  n={ntr} seed={s}: " + " | ".join(
                f"{r}: acc {res[(r,ntr)]['acc'][-1]:.2f} rang {res[(r,ntr)]['rank'][-1]:.1f}"
                for r in a.regs), flush=True)

    def ms(xs): return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)
    summary = {}
    for title, key in [("SONDE-g (qualité)", "acc"), ("RANG EFFECTIF (anti-collapse)", "rank")]:
        print(f"\n========== {title} — terme × budget ==========")
        print(f"{'reg':>9} | " + " | ".join(f"n={n:>5}" for n in a.n_trains))
        for r in a.regs:
            cells = []
            for n in a.n_trains:
                m, sd = ms(res[(r, n)][key])
                cells.append(f"{m:.2f}±{sd:.2f}" if key == "acc" else f"{m:.1f}")
                summary.setdefault(f"{r}|{n}", {})[key] = m; summary[f"{r}|{n}"][key + "_std"] = sd
            print(f"{r:>9} | " + " | ".join(f"{c:>9}" for c in cells))

    with open(os.path.join(a.out, "ssl_metrics.json"), "w") as f:
        json.dump({"meta": {"chance": 1/a.n_gbins, "n_trains": a.n_trains, "regs": a.regs,
                            "reg_w": a.reg_w, "nuisance": a.nuisance, "nonstat": a.nonstat,
                            "bounce": a.bounce, "wind": a.wind, "readout": a.readout,
                            "vision": a.vision, "vnoise": a.vnoise, "grid": a.grid},
                   "summary": summary}, f, indent=2)
    print(f"\n[json] -> {os.path.join(a.out, 'ssl_metrics.json')}")

if __name__ == "__main__":
    main()
