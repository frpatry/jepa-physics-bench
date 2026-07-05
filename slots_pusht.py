"""
PUSH-T — BLOC 2 : world model object-centré action-conditionné sur le VRAI Push-T (gym-pusht).

La recette complète validée (slots.py -> slots_dyn.py -> slots_act.py), portée en 96×96 :
peel récursif (K slots : agent bleu, bloc T gris, marqueur cible vert, bordure, fond blanc en
reliquat const), loss de mélange par pixel, inits par round, g relationnel (attention inter-slots)
conditionné sur l'action (delta normalisé (cible − agent)/256 — l'env attend des positions
absolues, g consomme le geste local).

Données : pusht_data.py (jeu aléatoire, aucune démonstration). Première montée en résolution du
projet : grille de features 48×48 (2304 tokens).

Métriques honnêtes (poses vraies embarquées dans le npz) : erreur position {agent, bloc} des
masques décodés (Hungarian, comme depuis le run 4) — perception t0, puis imaginé vs COPIE par
horizon, + sous-ensemble AVEC CONTACT. L'angle du T viendra au bloc 3 (planner pose complète).

  python slots_pusht.py --steps 20000 --bs 32            # entraîne + sauvegarde (Drive si monté)
  python slots_pusht.py --load 1 ...                     # recharge sans réentraîner
"""
import argparse, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from slots import mixture_nll, match_error
from slots_act import ActModel

class PushTModel(ActModel):
    """[RÉSULTAT NÉGATIF, gardé pour mémoire — NON UTILISÉ] Hypothèse FiLM réfutée par le test
    contrôlé sur T tournés : additif w32 0.0197, FiLM w32 0.0235 (aucun avantage), additif
    **w96 0.0048** (net !), additif w96 depth3 0.0482 (la profondeur nuit). Le goulot du run 1
    Push-T (perception plate 0.184, tout flou) était la LARGEUR du décodeur, pas le type de
    couplage slot×position — une somme+ReLU à largeur 96 peint très bien un T tourné.
    Fix retenu : --dec_w 96, zéro changement d'architecture."""
    def __init__(s, H, K, D=64, **kw):
        super().__init__(H, K, D, **kw)
        s.film_g = nn.Linear(s.slot_dim, s.slot_dim); s.film_b = nn.Linear(s.slot_dim, s.slot_dim)
    def decode_one(s, sl):                                                 # sl:(B,slot_dim)
        B = sl.size(0)
        g = s.pe_d.proj(s.pe_d.grid)                                       # (H,H,slot_dim) canaux positionnels
        x = g.unsqueeze(0) * s.film_g(sl).reshape(B, 1, 1, -1) + s.film_b(sl).reshape(B, 1, 1, -1)
        out = s.dec(x.permute(0, 3, 1, 2))
        return out[:, :3], out[:, 3:4]

def default_path(name):
    if os.path.isdir("/content/drive/MyDrive"): return f"/content/drive/MyDrive/{name}"
    if os.path.isdir("/content"): return f"/content/{name}"
    return name

def load_data(path):
    d = np.load(path)
    X, A, AG, BP, CT = d["X"], d["A"], d["AG"], d["BP"], d["CT"]
    dA = np.clip((A - AG[:, :-1]) / 256.0, -1.0, 1.0).astype(np.float32)   # geste local normalisé
    P = np.stack([AG / 512.0, BP[:, :, :2] / 512.0], 2).astype(np.float32) # (n,T,2,2) : agent, bloc
    return X, dA, P, CT

def to_batch(X, ids, dev):
    return torch.tensor(X[ids]).to(dev).permute(0, 1, 4, 2, 3).float().div(255.0)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=""); p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--load", type=int, default=0)
    p.add_argument("--K", type=int, default=5); p.add_argument("--D", type=int, default=64)
    p.add_argument("--slot_dim", type=int, default=16); p.add_argument("--dec_w", type=int, default=96)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4); p.add_argument("--sig", type=float, default=0.1)
    p.add_argument("--n_eval", type=int, default=150); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diag", type=int, default=0)                          # 1 = figure des masques par slot
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = a.data or default_path("pusht_data.npz"); ckpt = a.ckpt or default_path("pusht_wm.pt")
    X, dA, P, CT = load_data(data)
    n, T, H = X.shape[0], X.shape[1], X.shape[2]; HOR = T - 2
    ne = min(a.n_eval, n // 10); ntr = n - ne                              # éval = dernières séquences
    print(f"device={dev}  {n} séquences de {T} frames {H}x{H} (jeu aléatoire, vrai gym-pusht)"
          f"  K={a.K}  train {ntr} / éval {ne}", flush=True)
    m = ActModel(H, a.K, a.D, res=H // 2, slot_dim=a.slot_dim, dec_w=a.dec_w, iters=a.iters).to(dev)
    if a.load:
        m.load_state_dict(torch.load(ckpt, map_location=dev)["model"]); m.eval()
        print(f"modèle chargé <- {ckpt}", flush=True)
    else:
        opt = torch.optim.Adam(m.parameters(), a.lr)
        warm = max(1, a.steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                                  0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
        xe = to_batch(X, np.arange(ntr, n), dev); ae = torch.tensor(dA[ntr:]).to(dev)
        Pe = P[ntr:]; Ce = np.full(ne, 2, np.int64); hid = np.where(CT[ntr:].any(1))[0]
        print(f"éval : {ne} séquences dont {len(hid)} avec contact", flush=True)
        for st in range(a.steps):
            bi = np.random.randint(0, ntr, a.bs)
            x = to_batch(X, bi, dev); acts = torch.tensor(dA[bi]).to(dev)
            (m0, r0), (m1, r1), outs = m.rollout(x[:, 0], x[:, 1], acts[:, 1:])
            loss = mixture_nll(x[:, 0], r0, m0[:, :, 0], a.sig) + mixture_nll(x[:, 1], r1, m1[:, :, 0], a.sig)
            for h, (mh, rh) in enumerate(outs):
                loss = loss + mixture_nll(x[:, 2 + h], rh, mh[:, :, 0], a.sig)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step()
            if st % 500 == 0:
                with torch.no_grad():
                    (tm0, _), (tm1, _), touts = m.rollout(xe[:, 0], xe[:, 1], ae[:, 1:])
                    err0 = match_error(tm0[:, :, 0], Pe[:, 0], H, Ce)
                    eI = [match_error(touts[h][0][:, :, 0], Pe[:, 2 + h], H, Ce) for h in range(HOR)]
                    eC = [match_error(tm1[:, :, 0], Pe[:, 2 + h], H, Ce) for h in range(HOR)]
                    hI = [match_error(touts[h][0][hid][:, :, 0], Pe[hid, 2 + h], H, Ce[hid]) for h in range(HOR)]
                    hC = [match_error(tm1[hid][:, :, 0], Pe[hid, 2 + h], H, Ce[hid]) for h in range(HOR)]
                imag = " ".join(f"t+{h+1} {eI[h]:.3f}/{eC[h]:.3f}" for h in range(HOR))
                cont = " ".join(f"t+{h+1} {hI[h]:.3f}/{hC[h]:.3f}" for h in range(HOR))
                print(f"  step {st:5d}  perception t0 {err0:.3f}  |  imaginé/copie : {imag}"
                      f"  |  AVEC CONTACT : {cont}", flush=True)
        torch.save({"model": m.state_dict(), "args": vars(a)}, ckpt)
        print(f"modèle sauvegardé -> {ckpt}", flush=True)
    if a.diag:                                                             # QUI capture QUOI ? (la figure qui
        try:                                                               #  a tout appris aux runs 3-12)
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            ids = np.array([ntr, ntr + 5, ntr + 11])
            xv = to_batch(X, ids, dev)
            with torch.no_grad():
                mk, rgb, _ = m.peel(m.feats(xv[:, 0]))
            fig, ax = plt.subplots(len(ids), a.K + 2, figsize=(2 * (a.K + 2), 2.2 * len(ids)))
            for i in range(len(ids)):
                ax[i, 0].imshow(X[ids[i], 0]); ax[i, 0].set_title("image" if i == 0 else "")
                rec = (rgb * mk).sum(1)[i].permute(1, 2, 0).cpu().clip(0, 1)
                ax[i, 1].imshow(rec.numpy()); ax[i, 1].set_title("recon" if i == 0 else "")
                for k in range(a.K):
                    ax[i, k + 2].imshow(mk[i, k, 0].cpu(), cmap="viridis")
                    nom = "reliquat" if k == a.K - 1 else f"prise {k}"
                    ax[i, k + 2].set_title(nom if i == 0 else "")
                for j in range(a.K + 2): ax[i, j].axis("off")
            out = "/content/slots_pusht_diag.png" if os.path.isdir("/content") else "slots_pusht_diag.png"
            plt.tight_layout(); plt.savefig(out); print(f"diag masques -> {out}", flush=True)
        except Exception as e:
            print("diag skip:", str(e)[:60], flush=True)
    # figure : contexte + [vrai | imaginé] par horizon, sur des séquences d'éval avec contact
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        hid_all = np.where(CT.any(1))[0]; ids = list(hid_all[hid_all >= ntr][:3]) + [n - 1]
        xv = to_batch(X, np.array(ids), dev); av = torch.tensor(dA[np.array(ids)]).to(dev)
        with torch.no_grad():
            _, _, outs = m.rollout(xv[:, 0], xv[:, 1], av[:, 1:])
        ncol = 2 + 2 * HOR
        fig, ax = plt.subplots(len(ids), ncol, figsize=(2 * ncol, 2.2 * len(ids)))
        for i in range(len(ids)):
            ax[i, 0].imshow(X[ids[i], 0]); ax[i, 0].set_title("t0" if i == 0 else "")
            ax[i, 1].imshow(X[ids[i], 1]); ax[i, 1].set_title("t1" if i == 0 else "")
            for h in range(HOR):
                mh, rh = outs[h]; rec = (rh * mh).sum(1)
                ax[i, 2 + 2 * h].imshow(X[ids[i], 2 + h])
                ax[i, 2 + 2 * h].set_title(f"t{2+h} (vrai)" if i == 0 else "")
                ax[i, 3 + 2 * h].imshow(rec[i].permute(1, 2, 0).cpu().clip(0, 1).numpy())
                ax[i, 3 + 2 * h].set_title(f"t{2+h} IMAGINÉ" if i == 0 else "")
            for j in range(ncol): ax[i, j].axis("off")
        out = "/content/slots_pusht.png" if os.path.isdir("/content") else "slots_pusht.png"
        plt.tight_layout(); plt.savefig(out)
        print(f"figure -> {out}  (le T imaginé doit bouger/tourner comme le vrai)", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
