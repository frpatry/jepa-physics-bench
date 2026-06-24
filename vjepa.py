"""
V-JEPA FIDELE (style I-JEPA / V-JEPA Meta, anti-collapse LeJEPA SIGReg).

Ecarts corriges par rapport a l'ancien lejepa_video.py (qui etait un JEPA simplifie
facon MAE) — on respecte ici l'architecture de la vision LeCun :

  1. ENCODEUR-CONTEXTE sur les tokens VISIBLES UNIQUEMENT (pas de mask-token dans
     l'encodeur, contrairement a BERT/MAE). L'encodeur ne "voit" jamais ce qu'il doit predire.
  2. PREDICTEUR ATTENTIONNEL (Transformer) : recoit les latents visibles + des
     MASK-TOKENS POSITIONNELS aux positions cibles, et predit leurs latents par attention.
     (avant : simple MLP token-wise, incapable de router l'info spatiale.)
  3. MASQUAGE PAR BLOCS / TUBELETS spatio-temporels (et masquage TEMPOREL pour imaginer
     le futur), pas du Bernoulli aleatoire token-par-token.
  4. MULTI-MASQUE : N cibles differentes par clip a chaque pas (1 cible propre, N contextes).

Anti-collapse = SIGReg officiel (LeJEPA, Balestriero & LeCun). PAS d'EMA, PAS de stop-grad,
PAS de scheduler : c'est la these de LeJEPA (le terme statistique gaussien-isotrope suffit).

Module PUR torch (aucune dependance donnees/cv2) : les loaders vivent ailleurs.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ---------------------------------------------------------------- SIGReg (LeJEPA officiel)
_LE = None
def sigreg(z):
    """Terme anti-collapse SIGReg (test EPPS-PULLEY sur 1024 projections 1D) -> z gaussien isotrope."""
    global _LE
    if _LE is None:
        import lejepa
        _LE = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(t_max=3, n_points=17), num_slices=1024)
        try: _LE = _LE.to(z.device)
        except Exception: pass
    return _LE(z)

# ---------------------------------------------------------------- tokenisation spatio-temporelle
def patchify(X, P):
    """(n,T,H,H,3) -> (n, T*nP*nP, P*P*3). Tokens ordonnes frame-major :
    token = t*npf + cell  =>  frame_of = arange(ntok)//npf, cell = arange(ntok)%npf."""
    n, T, H, _, C = X.shape; nP = H // P
    X = X[:, :, :nP * P, :nP * P, :].reshape(n, T, nP, P, nP, P, C)
    X = X.transpose(0, 1, 2, 4, 3, 5, 6).reshape(n, T * nP * nP, P * P * C)
    return X

# ---------------------------------------------------------------- masquage par blocs / temporel
def tube_masks(B, T, nP, ratio, n, rng):
    """n masques TUBELET : on choisit ~ratio*npf cellules spatiales (par rectangles) puis on les
    masque sur TOUTES les frames (tube). Cardinalite par echantillon CONSTANTE (k*T) -> gather rectangulaire."""
    npf = nP * nP; k = max(1, min(npf - 1, round(ratio * npf)))
    out = []
    for _ in range(n):
        M = np.zeros((B, T * npf), bool)
        for b in range(B):
            cells = np.zeros(npf, bool)
            while cells.sum() < k:                                  # empile des rectangles
                h = rng.integers(1, nP // 2 + 2); w = rng.integers(1, nP // 2 + 2)
                r0 = rng.integers(0, nP - h + 1); c0 = rng.integers(0, nP - w + 1)
                cells.reshape(nP, nP)[r0:r0 + h, c0:c0 + w] = True
            on = np.where(cells)[0]
            if len(on) > k: cells[rng.choice(on, len(on) - k, replace=False)] = False  # ramene a exactement k
            M[b] = np.tile(cells, T)                                # tubelet sur tout l'axe temporel
        out.append(torch.from_numpy(M))
    return out

def temporal_mask(B, T, nP, t0, device):
    """masque toutes les frames > t0 (imaginer le futur). Cardinalite constante = (T-t0-1)*npf."""
    npf = nP * nP
    frame_of = torch.arange(T * npf, device=device) // npf
    return (frame_of > t0).unsqueeze(0).expand(B, -1).clone()

# ---------------------------------------------------------------- gather d'indices (cardinalite constante par ligne)
def _idx(m):
    """m:(B,N) bool, meme nb de True par ligne -> indices (B,c) tries."""
    c = int(m[0].sum().item())
    return m.float().topk(c, dim=1).indices.sort(1).values

def _gather(x, idx):
    """x:(B,N,D), idx:(B,c) -> (B,c,D)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

# ---------------------------------------------------------------- encodeur-contexte (ViT sur tokens donnes)
class Encoder(nn.Module):
    def __init__(s, obs, d, ntok, nl, nh):
        super().__init__()
        s.emb = nn.Linear(obs, d); s.pos = nn.Embedding(ntok, d)
        layer = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d)
    def forward(s, tokens, idx):
        """tokens:(B,k,obs) aux positions idx:(B,k). Aucun mask-token : l'encodeur ne voit QUE ce qu'on lui donne."""
        return s.ln(s.tr(s.emb(tokens) + s.pos(idx)))

# ---------------------------------------------------------------- predicteur ATTENTIONNEL (cross-attn vers cibles)
class Predictor(nn.Module):
    def __init__(s, d, ntok, nl, nh):
        super().__init__()
        s.mask_token = nn.Parameter(torch.zeros(d)); s.pos = nn.Embedding(ntok, d)
        layer = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, d)
    def forward(s, ctx, ctx_idx, tgt_idx):
        """ctx:(B,kc,d) latents visibles (deja position-aware). On ajoute des mask-tokens
        positionnels aux positions cibles, on melange par attention, on lit les cibles."""
        tg = s.mask_token + s.pos(tgt_idx)                          # requetes aux positions a predire
        x = s.tr(torch.cat([ctx, tg], dim=1))
        kt = tgt_idx.size(1)
        return s.head(s.ln(x[:, -kt:]))                             # latents predits aux cibles

# ---------------------------------------------------------------- V-JEPA complet
class VJEPA(nn.Module):
    def __init__(s, obs, d, ntok, nl=4, nh=4, rw=1.0, pred_layers=2):
        super().__init__()
        s.enc = Encoder(obs, d, ntok, nl, nh)
        s.pred = Predictor(d, ntok, pred_layers, nh)
        s.rw = rw; s.ntok = ntok

    def forward(s, o, masks):
        """o:(B,ntok,obs) clip propre ; masks = liste de masques-cibles bool (B,ntok). Multi-masque."""
        B, N, _ = o.shape
        allidx = torch.arange(N, device=o.device).expand(B, -1)
        z = s.enc(o, allidx)                                        # CIBLE : encodeur sur clip propre (poids partages, pas d'EMA/stop-grad)
        pl = 0.0
        for m in masks:
            vis, tgt = _idx(~m), _idx(m)
            ctx = s.enc(_gather(o, vis), vis)                       # CONTEXTE : encodeur sur VISIBLES seulement
            pred = s.pred(ctx, vis, tgt)                            # predit les cibles par attention
            pl = pl + F.smooth_l1_loss(pred, _gather(z, tgt))
        pl = pl / len(masks)
        return pl + s.rw * sigreg(z.reshape(-1, z.size(-1)))        # + anti-collapse SIGReg sur les latents propres

    @torch.no_grad()
    def feat(s, o):
        """representation pour la sonde : encodeur complet -> pooling moyen (B,d).
        ATTENTION : a haute resolution (beaucoup de tokens) la moyenne DILUE un signal localise
        (ex. une collision sur quelques patches) -> preferer tokens()+sonde attentive."""
        B, N, _ = o.shape
        allidx = torch.arange(N, device=o.device).expand(B, -1)
        return s.enc(o, allidx).mean(1)

    @torch.no_grad()
    def tokens(s, o):
        """latents PAR TOKEN (B,ntok,d), non poolés -> pour la sonde attentive (readout non diluant)."""
        B, N, _ = o.shape
        allidx = torch.arange(N, device=o.device).expand(B, -1)
        return s.enc(o, allidx)

    def predict_targets(s, o, m):
        """contexte observe (pool) + latents cibles imagines, pour rollout / MPC.
        ctx_pool:(B,d)  pred:(B,ntgt,d). (gradients possibles -> heatmap de saillance.)"""
        vis, tgt = _idx(~m), _idx(m)
        ctx = s.enc(_gather(o, vis), vis)
        return ctx.mean(1), s.pred(ctx, vis, tgt)

# ---------------------------------------------------------------- sonde lineaire gelee
def probe(Xtr, ytr, Xte, yte, dev, nc, steps=500):
    clf = nn.Linear(Xtr.size(1), nc).to(dev); opt = torch.optim.Adam(clf.parameters(), 1e-2)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(clf(Xtr), ytr).backward(); opt.step()
    with torch.no_grad(): return (clf(Xte).argmax(-1) == yte).float().mean().item()

# ---------------------------------------------------------------- sonde ATTENTIVE (protocole V-JEPA)
class AttentiveProbe(nn.Module):
    """une requete apprise pondere les tokens (cross-attention) -> classif. Encodeur GELE ;
    seule la sonde apprend => readout qui NE DILUE PAS (apprend OU regarder, ex. la collision)."""
    def __init__(s, d, nc, nh=4):
        super().__init__()
        s.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        s.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        s.ln = nn.LayerNorm(d); s.fc = nn.Linear(d, nc)
    def forward(s, toks):                                          # toks:(B,ntok,d) latents geles
        q = s.q.expand(toks.size(0), -1, -1)
        pooled, _ = s.attn(q, toks, toks)                          # (B,1,d)
        return s.fc(s.ln(pooled[:, 0]))

def attentive_probe(Ttr, ytr, Tte, yte, dev, nc, steps=800, lr=1e-3, bs=64):
    """Ttr/Tte:(N,ntok,d) latents geles (CPU ok, batches vers dev). ytr/yte:(N,) sur CPU."""
    clf = AttentiveProbe(Ttr.size(-1), nc).to(dev); opt = torch.optim.Adam(clf.parameters(), lr)
    n = len(Ttr)
    for _ in range(steps):
        bi = torch.randint(0, n, (min(bs, n),))
        opt.zero_grad(); F.cross_entropy(clf(Ttr[bi].to(dev)), ytr[bi].to(dev)).backward(); opt.step()
    with torch.no_grad():
        pr = torch.cat([clf(Tte[i:i+bs].to(dev)).argmax(-1).cpu() for i in range(0, len(Tte), bs)])
    return (pr == yte).float().mean().item()
