"""
DYNAMIQUE SUR LES SLOTS — le world model object-centré (SAVi-lite sur la recette gagnante de slots.py).

La fondation statique est acquise (slots.py : err 0.048, 1 slot/objet, comptage émergent).
Ici on pose la dynamique DESSUS : le monde bouge (disques avec vitesses + rebonds murs), le modèle
voit les frames t0 et t1, et doit IMAGINER t2 — qui n'est jamais encodée (prédiction honnête,
contrairement au mur n°2 où l'encodeur bidirectionnel trichait en voyant le futur).

Mécanique :
  t0 : épluchage (peel) -> slots S0                    [perception, recette slots.py]
  t1 : épluchage initialisé par S0 -> slots S1          [binding temporel à la SAVi : le round j
                                                         repart de « son » objet -> suivi gratuit]
  t2..t4 : ROLLOUT pur latent — Ŝ_{t+1} = S_t + g(...)  [g = interaction entre slots (attention
                                                         slot-à-slot, esprit interaction networks) :
                                                         nécessaire pour les COLLISIONS objet-objet,
                                                         où le futur de A dépend de l'état de B]
       décodage de chaque Ŝ SEUL (aucune image) = futur imaginé, comparé aux vraies frames.

Le monde a des COLLISIONS ÉLASTIQUES entre disques (--collide 1) : la factorisation pure
(un MLP par slot indépendant) est exactement fausse au moment du choc — le test dur.

Mesures honnêtes : err t0 (perception) ; err IMAGINÉ à t+1/t+2/t+3 vs COPIE (masques de t1 face
aux positions futures). Validé run 13 (sans collisions, 1 pas) : imaginé 0.067 ≈ perception 0.064,
copie 0.103 — le mur n°2 (prédiction honnête) est tombé.

  python slots_dyn.py --n 5000 --n_obj 4 --H 48 --K 5 --steps 20000 --bs 64
"""
import argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from slots import COLS, PosEmbed, SlotAttention, mixture_nll, match_error

def gen_seq(n, H, n_obj, r, T=5, seed=0, vary=1, smin=0.04, smax=0.12, collide=1):
    """Séquences de T frames : disques à vitesse constante + rebonds murs + COLLISIONS ÉLASTIQUES
    entre disques (masses égales : échange des composantes de vitesse le long de la normale).
    vary=1 : nombre d'objets 1..n_obj + couleurs au hasard (la leçon du run 12)."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    X = np.zeros((n, T, H, H, 3), np.float32); P = np.zeros((n, T, n_obj, 2), np.float32)
    C = np.full(n, n_obj, np.int64); hit = np.zeros(n, bool)               # hit[i] = un choc a eu lieu
    for i in range(n):
        if vary: C[i] = rng.integers(1, n_obj + 1)
        cols = rng.permutation(len(COLS))[:C[i]] if vary else np.arange(C[i])
        for k in range(C[i]):
            for _ in range(50):
                c = rng.uniform(r, 1 - r, 2).astype(np.float32)
                if k == 0 or np.all(np.linalg.norm(P[i, 0, :k] - c, axis=1) > 2.2 * r): P[i, 0, k] = c; break
        sp = rng.uniform(smin, smax, C[i]).astype(np.float32)
        th = rng.uniform(0, 2 * np.pi, C[i]).astype(np.float32)
        V = np.stack([sp * np.cos(th), sp * np.sin(th)], -1)
        for t in range(T):
            if t > 0:
                for k in range(C[i]):                                      # 1) déplacement + rebond mur
                    p = P[i, t - 1, k] + V[k]
                    for d in range(2):
                        if p[d] < r: p[d] = 2 * r - p[d]; V[k, d] = -V[k, d]
                        if p[d] > 1 - r: p[d] = 2 * (1 - r) - p[d]; V[k, d] = -V[k, d]
                    P[i, t, k] = p
                if collide:                                                # 2) chocs élastiques AVANT rendu :
                    for _ in range(3):                                     #    3 passes (chaînes à 3 corps)
                        for a_ in range(C[i]):                             #    impulsion + séparation au contact
                            for b_ in range(a_ + 1, C[i]):
                                d = P[i, t, a_] - P[i, t, b_]; dist = float(np.linalg.norm(d))
                                if 1e-6 < dist < 2 * r:
                                    nv = d / dist; s_ = float((V[a_] - V[b_]) @ nv)
                                    if s_ < 0: V[a_] -= s_ * nv; V[b_] += s_ * nv; hit[i] = True
                                    push = (2 * r - dist) / 2                      # sépare au contact
                                    P[i, t, a_] = np.clip(P[i, t, a_] + push * nv, r, 1 - r)
                                    P[i, t, b_] = np.clip(P[i, t, b_] - push * nv, r, 1 - r)
            for k in range(C[i]):
                X[i, t][(xx - P[i, t, k, 0]) ** 2 + (yy - P[i, t, k, 1]) ** 2 < r * r] = COLS[cols[k]]
    return X, P, C, hit

class DynModel(nn.Module):
    def __init__(s, H, K, D=64, res=24, slot_dim=16, dec_w=32, iters=3):
        super().__init__(); s.res, s.K, s.D, s.slot_dim, s.H = res, K, D, slot_dim, H
        s.enc = nn.Sequential(nn.Conv2d(3, D, 5, 1, 2), nn.ReLU(), nn.Conv2d(D, D, 5, 2, 2), nn.ReLU(),
                              nn.Conv2d(D, D, 5, 1, 2), nn.ReLU())
        s.pe = PosEmbed(D, res); s.mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        s.sa = SlotAttention(2, D, iters, "learned")                       # le cerneur objet/reste
        s.peel_init = nn.Parameter(torch.randn(1, K - 1, 2, D) * 0.5)      # rôles distincts par round
        s.down = nn.Linear(D, slot_dim)                                    # goulot 16d
        s.pe_d = PosEmbed(slot_dim, H)
        s.dec = nn.Sequential(nn.Conv2d(slot_dim, dec_w, 1), nn.ReLU(),    # SBD 1x1 faible
                              nn.Conv2d(dec_w, dec_w, 1), nn.ReLU(), nn.Conv2d(dec_w, 4, 1))
        s.bg = nn.Parameter(torch.zeros(1, 3, 1, 1))                       # reliquat = couleur unie
        s.bg_logit = nn.Parameter(torch.zeros(1))                          # compositing du futur imaginé
        # dynamique AVEC INTERACTION : chaque slot voit les autres via attention (esprit interaction
        # networks) — indispensable pour les collisions, où le futur de A dépend de l'état de B
        s.g_in = nn.Linear(2 * D, D)                                       # [état, vitesse] -> h
        s.g_att = nn.MultiheadAttention(D, 4, batch_first=True)            # les slots se regardent
        s.g_out = nn.Sequential(nn.Linear(2 * D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))
    def feats(s, img):
        B = img.size(0); f = s.enc(img).permute(0, 2, 3, 1)
        return s.mlp(s.pe(f).reshape(B, s.res * s.res, s.D))
    def decode_one(s, sl):                                                 # sl:(B,slot_dim) -> rgb, logit alpha
        B = sl.size(0)
        x = sl.reshape(B, s.slot_dim, 1, 1).expand(-1, -1, s.H, s.H).permute(0, 2, 3, 1)
        x = s.pe_d(x).permute(0, 3, 1, 2); out = s.dec(x)
        return out[:, :3], out[:, 3:4]
    def peel(s, f, init=None):
        """Épluchage (recette slots.py). init:(B,K-1,D) = slots objets du pas précédent
        (corrector à la SAVi : le round j repart de « son » objet -> binding temporel)."""
        B, H = f.size(0), s.H
        scope = torch.ones(B, 1, H, H, device=f.device)
        masks, rgbs, S = [], [], []
        for j in range(s.K - 1):
            sc = F.adaptive_avg_pool2d(scope, s.res).reshape(B, s.res * s.res, 1)
            if init is None: s0 = s.peel_init[:, j].expand(B, -1, -1)
            else: s0 = torch.stack([init[:, j], s.peel_init[:, j, 1].expand(B, -1)], 1)
            slots2, _ = s.sa(f, scope=sc, slots0=s0)
            S.append(slots2[:, 0])
            rgb, alog = s.decode_one(s.down(slots2).reshape(B * 2, s.slot_dim))
            rgb = rgb.reshape(B, 2, 3, H, H); alog = alog.reshape(B, 2, 1, H, H)
            a = torch.softmax(alog, 1)
            masks.append(scope * a[:, 0]); rgbs.append(rgb[:, 0]); scope = scope * a[:, 1]
        masks.append(scope); rgbs.append(s.bg.expand(B, 3, H, H))
        return torch.stack(masks, 1), torch.stack(rgbs, 1), torch.stack(S, 1)  # (B,K,1,H,H) (B,K,3,H,H) (B,K-1,D)
    def imagine(s, S):
        """Décode le futur depuis les slots PRÉDITS seuls (aucune image) : chaque slot peint sa part,
        compétition softmax par pixel + logit fond appris."""
        B, Km1, _ = S.shape; H = s.H
        rgb, alog = s.decode_one(s.down(S).reshape(B * Km1, s.slot_dim))
        rgb = rgb.reshape(B, Km1, 3, H, H); alog = alog.reshape(B, Km1, 1, H, H)
        logits = torch.cat([alog, s.bg_logit.reshape(1, 1, 1, 1, 1).expand(B, 1, 1, H, H)], 1)
        a = torch.softmax(logits, 1)
        rgbs = torch.cat([rgb, s.bg.expand(B, 3, H, H).unsqueeze(1)], 1)
        return a, rgbs                                                     # (B,K,1,H,H) (B,K,3,H,H)
    def step(s, S, Sprev):
        """Un pas de dynamique : Ŝ_{t+1} = S_t + interaction(S_t, vitesse). Partagé entre slots."""
        h = s.g_in(torch.cat([S, S - Sprev], -1))                          # (B,K-1,D)
        att, _ = s.g_att(h, h, h)                                          # chaque slot lit les autres
        return S + s.g_out(torch.cat([h, att], -1))
    def forward(s, x0, x1, horizons=1):
        m0, r0, S0 = s.peel(s.feats(x0))
        m1, r1, S1 = s.peel(s.feats(x1), init=S0)                          # suivi : repart des slots de t0
        outs, prev, cur = [], S0, S1
        for _ in range(horizons):                                          # rollout PUR LATENT
            nxt = s.step(cur, prev)
            outs.append(s.imagine(nxt))
            prev, cur = cur, nxt
        return (m0, r0), (m1, r1), outs

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000); p.add_argument("--n_obj", type=int, default=4)
    p.add_argument("--H", type=int, default=48); p.add_argument("--r", type=float, default=0.15)
    p.add_argument("--K", type=int, default=5); p.add_argument("--D", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-4); p.add_argument("--iters", type=int, default=3)
    p.add_argument("--slot_dim", type=int, default=16); p.add_argument("--dec_w", type=int, default=32)
    p.add_argument("--sig", type=float, default=0.1); p.add_argument("--vary", type=int, default=1)
    p.add_argument("--smin", type=float, default=0.04); p.add_argument("--smax", type=float, default=0.12)
    p.add_argument("--T", type=int, default=5)                             # t0,t1 contexte + T-2 imaginés
    p.add_argument("--collide", type=int, default=1)                       # chocs élastiques entre disques
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    HOR = a.T - 2                                                          # nb de pas imaginés
    X, P, C, _ = gen_seq(a.n, a.H, a.n_obj, a.r, T=a.T, seed=a.seed, vary=a.vary,
                         smin=a.smin, smax=a.smax, collide=a.collide)
    Xt = torch.tensor(X.transpose(0, 1, 4, 2, 3))                          # (n,T,3,H,H)
    nobj = f"1..{a.n_obj} objets (variable)" if a.vary else f"{a.n_obj} objets"
    print(f"device={dev}  {a.n} séquences de {a.T} frames  {nobj}  collisions={'OUI' if a.collide else 'non'}"
          f"  K={a.K}  seed={a.seed}  (voit t0,t1 -> IMAGINE t2..t{a.T-1}, jamais encodés)", flush=True)
    m = DynModel(a.H, a.K, a.D, slot_dim=a.slot_dim, dec_w=a.dec_w, iters=a.iters).to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    Xe, Pe, Ce, He = gen_seq(400, a.H, a.n_obj, a.r, T=a.T, seed=7, vary=a.vary,
                             smin=a.smin, smax=a.smax, collide=a.collide)
    Xet = torch.tensor(Xe.transpose(0, 1, 4, 2, 3)).to(dev)
    hid = np.where(He)[0]                                                  # séquences AVEC choc : le test dur
    print(f"éval : {len(He)} séquences dont {len(hid)} avec choc entre objets", flush=True)
    for st in range(a.steps):
        bi = np.random.randint(0, a.n, a.bs)
        x = Xt[bi].to(dev)                                                 # (bs,T,3,H,H)
        (m0, r0), (m1, r1), outs = m(x[:, 0], x[:, 1], horizons=HOR)
        loss = mixture_nll(x[:, 0], r0, m0[:, :, 0], a.sig) + mixture_nll(x[:, 1], r1, m1[:, :, 0], a.sig)
        for h, (mh, rh) in enumerate(outs):                                # chaque frame imaginée vs la vraie
            loss = loss + mixture_nll(x[:, 2 + h], rh, mh[:, :, 0], a.sig)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if st % 500 == 0:
            with torch.no_grad():
                (tm0, _), (tm1, _), touts = m(Xet[:, 0], Xet[:, 1], horizons=HOR)
                err0 = match_error(tm0[:, :, 0], Pe[:, 0], a.H, Ce)        # perception t0
                eI = [match_error(touts[h][0][:, :, 0], Pe[:, 2 + h], a.H, Ce) for h in range(HOR)]
                eC = [match_error(tm1[:, :, 0], Pe[:, 2 + h], a.H, Ce) for h in range(HOR)]  # copie de t1
                # le test dur : uniquement les séquences AVEC choc (la moyenne est dominée par le balistique)
                hI = [match_error(touts[h][0][hid][:, :, 0], Pe[hid, 2 + h], a.H, Ce[hid]) for h in range(HOR)]
                hC = [match_error(tm1[hid][:, :, 0], Pe[hid, 2 + h], a.H, Ce[hid]) for h in range(HOR)]
            imag = " ".join(f"t+{h+1} {eI[h]:.3f}/{eC[h]:.3f}" for h in range(HOR))
            choc = " ".join(f"t+{h+1} {hI[h]:.3f}/{hC[h]:.3f}" for h in range(HOR))
            print(f"  step {st:5d}  perception t0 {err0:.3f}  |  imaginé/copie : {imag}"
                  f"  |  AVEC CHOC : {choc}", flush=True)
    # figure : contexte t0,t1 puis pour chaque horizon [vrai | imaginé]
    try:
        import matplotlib, os; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with torch.no_grad():
            Xv, Pv, Cv, Hv = gen_seq(60, a.H, a.n_obj, a.r, T=a.T, seed=3, vary=a.vary,
                                     smin=a.smin, smax=a.smax, collide=a.collide)
            ids = (list(np.where(Hv)[0])[:2] + list(np.where(~Hv)[0])[:2])[:4]  # 2 avec choc, 2 sans
            Xv = Xv[ids]
            Xvt = torch.tensor(Xv.transpose(0, 1, 4, 2, 3)).to(dev)
            _, _, outs = m(Xvt[:, 0], Xvt[:, 1], horizons=HOR)
        ncol = 2 + 2 * HOR
        fig, ax = plt.subplots(4, ncol, figsize=(2 * ncol, 8))
        for i in range(4):
            ax[i, 0].imshow(Xv[i, 0].clip(0, 1)); ax[i, 0].set_title("t0" if i == 0 else "")
            ax[i, 1].imshow(Xv[i, 1].clip(0, 1)); ax[i, 1].set_title("t1" if i == 0 else "")
            for h in range(HOR):
                mh, rh = outs[h]; rec = (rh * mh).sum(1)
                ax[i, 2 + 2 * h].imshow(Xv[i, 2 + h].clip(0, 1))
                ax[i, 2 + 2 * h].set_title(f"t{2+h} (vrai)" if i == 0 else "")
                ax[i, 3 + 2 * h].imshow(rec[i].permute(1, 2, 0).cpu().clip(0, 1))
                ax[i, 3 + 2 * h].set_title(f"t{2+h} IMAGINÉ" if i == 0 else "")
            for j in range(ncol): ax[i, j].axis("off")
        out = "/content/slots_dyn.png" if os.path.isdir("/content") else "slots_dyn.png"
        plt.tight_layout(); plt.savefig(out)
        print(f"\nfigure -> {out}  (chaque frame imaginée doit matcher la vraie, collisions comprises)", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
