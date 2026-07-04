"""
OBJECT-CENTRIC ÉMERGENT — Slot Attention (Locatello et al.) sur le monde d'objets.

Fondation principielle : des SLOTS se disputent l'explication de l'image et DÉCOUVRENT les
objets SANS supervision (aucun objet codé en dur). Chaque slot -> une entité + son masque.
Objectif = reconstruction (les slots doivent tout expliquer -> ils se répartissent les objets).

Note honnête : on utilise un embedding positionnel (comme TOUT ViT / DINOv2 / V-JEPA 2.1) —
c'est le "où est ce patch" générique, pas des coordonnées d'objet injectées. Les objets émergent.

Mesure : les masques localisent-ils les objets ? (centre de masse par slot -> apparié aux vrais
objets par Hungarian -> erreur de position). C'est la qualité de découverte, émergente.

Anti-collapse (leçon du run GPU) : un décodeur conv puissant + slots 64d = reconstruction
paresseuse (un slot peint tout, erreur position plate). Fix : goulot slot_dim=16 + Spatial
Broadcast Decoder en convs 1x1 (faible) -> peindre 1 objet/slot devient la solution optimale.

  python slots.py --n 5000 --n_obj 3 --H 48 --K 4 --steps 15000 --bs 64
"""
import argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

COLS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], np.float32)

def gen(n, H, n_obj, r, seed=0):
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    X = np.zeros((n, H, H, 3), np.float32); P = np.zeros((n, n_obj, 2), np.float32)
    for i in range(n):
        for k in range(n_obj):
            for _ in range(50):
                c = rng.uniform(r, 1 - r, 2).astype(np.float32)
                if k == 0 or np.all(np.linalg.norm(P[i, :k] - c, axis=1) > 2.2 * r): P[i, k] = c; break
            X[i][(xx - P[i, k, 0]) ** 2 + (yy - P[i, k, 1]) ** 2 < r * r] = COLS[k]
    return X, P

def build_grid(res):
    r = np.linspace(0., 1., res, dtype=np.float32); x, y = np.meshgrid(r, r)
    return torch.tensor(np.stack([x, y, 1 - x, 1 - y], -1))              # (res,res,4)

class PosEmbed(nn.Module):
    def __init__(s, D, res):
        super().__init__(); s.proj = nn.Linear(4, D); s.register_buffer("grid", build_grid(res))
    def forward(s, x): return x + s.proj(s.grid)                          # x:(B,res,res,D)

class SlotAttention(nn.Module):
    def __init__(s, K, dim, iters=3, init="learned"):
        super().__init__(); s.K, s.dim, s.iters, s.init = K, dim, iters, init
        if init == "learned":                                             # K inits DISTINCTES apprises : brise la
            s.init_slots = nn.Parameter(torch.randn(1, K, dim) * 0.5)     # symétrie entre slots (sinon clones)
        else:
            s.mu = nn.Parameter(torch.randn(1, 1, dim)); s.logsig = nn.Parameter(torch.zeros(1, 1, dim))
        s.q = nn.Linear(dim, dim, bias=False); s.k = nn.Linear(dim, dim, bias=False); s.v = nn.Linear(dim, dim, bias=False)
        s.gru = nn.GRUCell(dim, dim); s.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        s.ni = nn.LayerNorm(dim); s.ns = nn.LayerNorm(dim); s.nm = nn.LayerNorm(dim)
    def forward(s, inp, scope=None):                                      # inp:(B,N,dim)  scope:(B,N,1) optionnel
        B, N, _ = inp.shape
        if s.init == "learned":
            slots = s.init_slots.expand(B, -1, -1)
        else:
            slots = s.mu + s.logsig.exp() * torch.randn(B, s.K, s.dim, device=inp.device)
        inp = s.ni(inp); k = s.k(inp); v = s.v(inp)
        attn = None
        for _ in range(s.iters):
            prev = slots; q = s.q(s.ns(slots))
            att = torch.softmax((k @ q.transpose(-1, -2)) * s.dim ** -0.5, dim=-1)  # (B,N,K) : les slots SE DISPUTENT
            attn = att + 1e-8
            if scope is not None: attn = attn * scope                               # ne lit que ce qui reste à expliquer
            w = attn / attn.sum(1, keepdim=True)                                    # moyenne pondérée sur N
            upd = w.transpose(-1, -2) @ v                                           # (B,K,dim)
            slots = s.gru(upd.reshape(-1, s.dim), prev.reshape(-1, s.dim)).reshape(B, s.K, s.dim)
            slots = slots + s.mlp(s.nm(slots))
        return slots, attn                                               # attn:(B,N,K) masques

class SeqAttention(nn.Module):
    """Décomposition SÉQUENTIELLE (explaining away, esprit MONet) : le slot j lit l'image
    via attention restreinte au SCOPE (ce qui reste à expliquer), peint sa part, le scope
    est réduit multiplicativement scope*(1-masque), le slot suivant ne voit que le reste.
    Le dernier slot ramasse le reliquat (fond). Brise la symétrie PAR CONSTRUCTION :
    aucune égalité possible entre slots. Émergent : rien ne dit où/quoi sont les objets."""
    def __init__(s, K, dim, iters=3):
        super().__init__(); s.K, s.dim, s.iters = K, dim, iters
        s.q0 = nn.Parameter(torch.randn(1, K, dim) * 0.5)                 # une requête apprise par étape
        s.q = nn.Linear(dim, dim, bias=False); s.k = nn.Linear(dim, dim, bias=False); s.v = nn.Linear(dim, dim, bias=False)
        s.ni = nn.LayerNorm(dim); s.ns = nn.LayerNorm(dim)
        s.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def read(s, kf, vf, j, logscope, B):                                  # lecture attentionnelle sous scope
        q = s.q0[:, j].expand(B, -1)                                      # (B,dim)
        for _ in range(s.iters):
            logit = (kf @ s.q(s.ns(q)).unsqueeze(-1)).squeeze(-1) * s.dim ** -0.5
            att = torch.softmax(logit + logscope, dim=1)                  # (B,N) : ne regarde que le reste
            u = (att.unsqueeze(1) @ vf).squeeze(1)                        # (B,dim)
            q = u + s.mlp(q)
        return q

class Model(nn.Module):
    def __init__(s, H, K, D=64, res=24, slot_dim=16, dec="sbd", dec_w=32, iters=3, init="learned", mode="par"):
        super().__init__(); s.res = res; s.K = K; s.D = D; s.slot_dim = slot_dim; s.mode = mode
        s.enc = nn.Sequential(nn.Conv2d(3, D, 5, 1, 2), nn.ReLU(), nn.Conv2d(D, D, 5, 2, 2), nn.ReLU(),
                              nn.Conv2d(D, D, 5, 1, 2), nn.ReLU())        # H -> res (H=48 -> 24, grille fine)
        s.pe = PosEmbed(D, res); s.mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        # peel = le mécanisme gagnant du run 4 (compétition SlotAttention) réduit à 2 slots aux rôles
        # appris ("objet" vs "reste"), appliqué RÉCURSIVEMENT : il cerne un objet, on le retire du
        # scope, il recommence sur le reste. K-1 rounds + reliquat final = fond. Poids partagés.
        s.sa = (SeqAttention(K, D, iters) if mode == "seq" else
                SlotAttention(2, D, iters, init) if mode == "peel" else SlotAttention(K, D, iters, init))
        s.down = nn.Linear(D, slot_dim)   # goulot : 16 dims ne suffisent pas pour encoder TOUTE la scène dans un slot
        if dec == "sbd":                  # Spatial Broadcast Decoder : convs 1x1 = pointwise, délibérément FAIBLE
            s.dres = H; s.pe_d = PosEmbed(slot_dim, H)
            s.dec = nn.Sequential(nn.Conv2d(slot_dim, dec_w, 1), nn.ReLU(),
                                  nn.Conv2d(dec_w, dec_w, 1), nn.ReLU(), nn.Conv2d(dec_w, 4, 1))
        else:                             # ancien décodeur conv (trop puissant -> collapse paresseux observé)
            s.dres = res; s.pe_d = PosEmbed(slot_dim, res)
            s.dec = nn.Sequential(nn.ConvTranspose2d(slot_dim, D, 5, 2, 2, 1), nn.ReLU(),
                                  nn.Conv2d(D, D, 5, 1, 2), nn.ReLU(), nn.Conv2d(D, 4, 3, 1, 1))  # res -> H
    def decode_one(s, sl, H):                                            # sl:(B,slot_dim) -> rgb, logit alpha
        B = sl.size(0)
        x = sl.reshape(B, s.slot_dim, 1, 1).expand(-1, -1, s.dres, s.dres).permute(0, 2, 3, 1)
        x = s.pe_d(x).permute(0, 3, 1, 2)
        out = s.dec(x)                                                   # (B,4,H,H)
        return out[:, :3], out[:, 3:4]
    def forward(s, img):                                                 # img:(B,3,H,H)
        B, H = img.size(0), img.size(2); f = s.enc(img)                  # (B,D,res,res)
        f = f.permute(0, 2, 3, 1); f = s.pe(f).reshape(B, s.res * s.res, s.D); f = s.mlp(f)
        if s.mode == "peel":                                             # épluchage : run-4 récursif
            scope = torch.ones(B, 1, H, H, device=img.device)            # ce qui RESTE à expliquer
            rgbs, masks = [], []
            for j in range(s.K - 1):
                sc = F.adaptive_avg_pool2d(scope, s.res).reshape(B, s.res * s.res, 1)
                slots2, _ = s.sa(f, scope=sc)                            # compétition objet/reste sous scope
                sl = s.down(slots2).reshape(B * 2, s.slot_dim)
                rgb, alog = s.decode_one(sl, H)                          # (B*2,3,H,H), (B*2,1,H,H)
                rgb = rgb.reshape(B, 2, 3, H, H); alog = alog.reshape(B, 2, 1, H, H)
                a = torch.softmax(alog, dim=1)                           # partition objet/reste DANS le scope
                masks.append(scope * a[:, 0]); rgbs.append(rgb[:, 0])    # l'objet cerné est engrangé
                scope = scope * a[:, 1]                                  # on passe au suivant
            masks.append(scope); rgbs.append(rgb[:, 1])                  # reliquat = fond (rgb du dernier "reste")
            a = torch.stack(masks, 1); rgbs = torch.stack(rgbs, 1)       # (B,K,1,H,H) somme=1 ; (B,K,3,H,H)
            return (rgbs * a).sum(1), a[:, :, 0], None, rgbs
        if s.mode == "seq":                                              # explaining away séquentiel
            fn = s.sa.ni(f); kf = s.sa.k(fn); vf = s.sa.v(fn)
            scope = torch.ones(B, 1, H, H, device=img.device)            # ce qui RESTE à expliquer
            rgbs, masks = [], []
            for j in range(s.K):
                sc = F.adaptive_avg_pool2d(scope, s.res).reshape(B, s.res * s.res)
                slot = s.sa.read(kf, vf, j, (sc + 1e-8).log(), B)        # attention restreinte au scope
                rgb, alog = s.decode_one(s.down(slot), H)
                if j < s.K - 1:
                    m = torch.sigmoid(alog)                              # part réclamée par ce slot
                    masks.append(scope * m); scope = scope * (1 - m)     # retrait multiplicatif
                else:
                    masks.append(scope)                                  # le dernier ramasse le reliquat (fond)
                rgbs.append(rgb)
            a = torch.stack(masks, 1); rgbs = torch.stack(rgbs, 1)       # (B,K,1,H,H) somme=1 ; (B,K,3,H,H)
            return (rgbs * a).sum(1), a[:, :, 0], None, rgbs
        slots, attn = s.sa(f)                                            # slots:(B,K,D)  attn:(B,N,K)
        sl = s.down(slots)                                               # (B,K,slot_dim)
        x = sl.reshape(B * s.K, s.slot_dim, 1, 1).expand(-1, -1, s.dres, s.dres).permute(0, 2, 3, 1)
        x = s.pe_d(x).permute(0, 3, 1, 2)
        out = s.dec(x).reshape(B, s.K, 4, img.size(2), img.size(3))
        rgb, a = out[:, :, :3], out[:, :, 3:4]; a = torch.softmax(a, dim=1)   # masques de décodage (compétition)
        recon = (rgb * a).sum(1)                                         # (B,3,H,H)
        return recon, a[:, :, 0], attn, rgb                             # a masks (B,K,H,H)

def mixture_nll(img, rgbs, masks, sig=0.1):
    """-log p(x) sous le modèle de MÉLANGE par pixel (MONet/IODINE) : p(x) = Σ_k m_k N(x|rgb_k, σ²).
    Interdit le mélange de couleurs : l'optimum = UN slot explique le pixel avec un poids -> 1."""
    d2 = ((img.unsqueeze(1) - rgbs) ** 2).sum(2)                         # (B,K,H,H)
    logp = (masks + 1e-8).log() - d2 / (2 * sig * sig)
    return -torch.logsumexp(logp, 1).mean()

def match_error(masks, P, H):
    """centre de masse de chaque masque -> apparié aux vrais objets (Hungarian) -> erreur position."""
    try: from scipy.optimize import linear_sum_assignment
    except Exception: linear_sum_assignment = None
    B, K, _, _ = masks.shape; yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    gx = torch.tensor(xx).to(masks.device); gy = torch.tensor(yy).to(masks.device); errs = []
    for i in range(B):
        m = masks[i]; w = m / (m.sum((-1, -2), keepdim=True) + 1e-8)
        cx = (w * gx).sum((-1, -2)); cy = (w * gy).sum((-1, -2))         # (K,)
        cen = torch.stack([cx, cy], -1).detach().cpu().numpy()           # (K,2)
        gt = P[i]                                                        # (n_obj,2)
        cost = np.linalg.norm(cen[:, None] - gt[None], axis=-1)         # (K,n_obj)
        if linear_sum_assignment is not None:
            ri, ci = linear_sum_assignment(cost); errs.append(cost[ri, ci].mean())
        else:
            errs.append(min(cost[k].min() for k in range(len(gt))))     # secours grossier
    return float(np.mean(errs))

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000); p.add_argument("--n_obj", type=int, default=3)
    p.add_argument("--H", type=int, default=48); p.add_argument("--r", type=float, default=0.15)
    p.add_argument("--K", type=int, default=4); p.add_argument("--D", type=int, default=64)
    p.add_argument("--steps", type=int, default=8000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--slot_dim", type=int, default=16)                    # goulot par slot avant décodage
    p.add_argument("--dec", choices=["sbd", "conv"], default="sbd")       # sbd = décodeur faible (anti-collapse)
    p.add_argument("--dec_w", type=int, default=32); p.add_argument("--iters", type=int, default=3)
    p.add_argument("--init", choices=["learned", "gauss"], default="learned")  # learned = brise la symétrie
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["par", "seq", "peel"], default="par")  # peel = run-4 récursif (objet par objet)
    p.add_argument("--loss", choices=["mse", "mix"], default="mse")       # mix = vraisemblance de mélange par pixel
    p.add_argument("--sig", type=float, default=0.1)                      # écart-type du mélange (loss mix)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, P = gen(a.n, a.H, a.n_obj, a.r)
    Xt = torch.tensor(X.transpose(0, 3, 1, 2))
    print(f"device={dev}  {a.n} images  {a.n_obj} objets  K={a.K} slots  mode={a.mode}  init={a.init}"
          f"  seed={a.seed}  (découverte SANS étiquettes)", flush=True)
    m = Model(a.H, a.K, a.D, slot_dim=a.slot_dim, dec=a.dec, dec_w=a.dec_w, iters=a.iters, init=a.init,
              mode=a.mode).to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    warm = max(1, a.steps // 20)                                          # warmup LR : crucial pour que les slots se différencient
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    for st in range(a.steps):
        bi = np.random.randint(0, a.n, a.bs); img = Xt[bi].to(dev)
        recon, masks_tr, _, rgbs = m(img)
        mse = F.mse_loss(recon, img)
        loss = mixture_nll(img, rgbs, masks_tr, a.sig) if a.loss == "mix" else mse
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if st % 500 == 0:
            with torch.no_grad():
                te = torch.tensor(gen(200, a.H, a.n_obj, a.r, seed=7)[0].transpose(0, 3, 1, 2)).to(dev)
                recon_te, masks, _, _ = m(te); err = match_error(masks, gen(200, a.H, a.n_obj, a.r, seed=7)[1], a.H)
                # séparation = moyenne du max sur K par pixel : 1/K = slots clones, ->1 = décomposition dure
                sep = masks.max(1).values.mean().item()
            print(f"  step {st:5d}  recon {mse.item():.4f}  erreur position (slots->objets) {err:.3f}"
                  f"  séparation {sep:.2f} (1/K={1/a.K:.2f}=clones, 1=dur)", flush=True)
    # figure : image + masques par slot
    try:
        import matplotlib, os; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with torch.no_grad():
            v = torch.tensor(gen(4, a.H, a.n_obj, a.r, seed=3)[0].transpose(0, 3, 1, 2)).to(dev)
            recon, masks, _, _ = m(v)
        fig, ax = plt.subplots(4, a.K + 2, figsize=(2 * (a.K + 2), 8))
        for i in range(4):
            ax[i, 0].imshow(v[i].permute(1, 2, 0).cpu().clip(0, 1)); ax[i, 0].set_title("image" if i == 0 else "")
            ax[i, 1].imshow(recon[i].permute(1, 2, 0).cpu().clip(0, 1)); ax[i, 1].set_title("recon" if i == 0 else "")
            for k in range(a.K):
                ax[i, k + 2].imshow(masks[i, k].cpu(), cmap="viridis"); ax[i, k + 2].set_title(f"slot {k}" if i == 0 else "")
            for j in range(a.K + 2): ax[i, j].axis("off")
        out = "/content/slots.png" if os.path.isdir("/content") else "slots.png"
        plt.tight_layout(); plt.savefig(out); print(f"\nfigure -> {out}  (chaque slot devrait capturer UN objet)", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
