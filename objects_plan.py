"""
MONDE D'OBJETS — SYSTEM-2 : un agent traverse un champ d'OBSTACLES MOBILES en se servant du
world model pour IMAGINER où ils seront, puis planifie l'évitement.

  - World model V-JEPA (LeJEPA, sans étiquettes) sur des vidéos d'obstacles -> encodeur GELÉ.
  - Forward model : depuis 2 latents observés -> positions futures des obstacles (one-shot).
  - Agent (objet contrôlé) doit aller de gauche à droite SANS toucher les obstacles.
  - 3 politiques, mêmes trajectoires d'obstacles (comparaison juste) :
      naïf      : fonce vers la cible (ignore les obstacles)
      System-1  : évite les positions ACTUELLES (suppose obstacles immobiles)
      System-2  : évite les positions IMAGINÉES par le world model (anticipe)
  Métriques : % de collisions et % d'arrivées. System-2 doit toucher MOINS que System-1.

  python objects_plan.py --n 2500 --n_test 300 --T 14 --H 48 --n_obj 3 --ssl_steps 5000
"""
import argparse, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vjepa import patchify, VJEPA, tube_masks, temporal_mask
from objects import gen_clips, render, per_frame_latent, r2

def unit_dirs(n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32)

def run_episode(start, target, pred_obs, true_obs, t0, a, dirs):
    """pred_obs (T,n_obj,2) = positions obstacles supposées (imaginées/figées), None=naïf.
    Collision toujours vérifiée contre true_obs. Renvoie (touché, arrivé)."""
    pos = start.copy(); hit = 0
    for t in range(t0, a.T):
        if pred_obs is None:                               # naïf : tout droit vers la cible
            d = target - pos; d = d / (np.linalg.norm(d) + 1e-9)
        else:
            best = dirs[0]; bestc = 1e9
            for dd in dirs:
                nxt = np.clip(pos + dd * a.aspeed * a.dt, 0.0, 1.0)
                prog = np.linalg.norm(nxt - target)
                dmin = min(np.linalg.norm(nxt - pred_obs[min(t, a.T - 1), k]) for k in range(pred_obs.shape[1]))
                c = prog + (a.w_avoid if dmin < a.r_safe else 0.0)
                if c < bestc: bestc = c; best = dd
            d = best
        pos = np.clip(pos + d * a.aspeed * a.dt, 0.0, 1.0)
        if min(np.linalg.norm(pos - true_obs[t, k]) for k in range(true_obs.shape[1])) < a.r_hit:
            hit = 1; break
        if np.linalg.norm(pos - target) < a.tol: break
    reached = int(np.linalg.norm(pos - target) < a.tol and not hit)
    return hit, reached

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2500); p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--T", type=int, default=12); p.add_argument("--H", type=int, default=48)
    p.add_argument("--patch", type=int, default=8); p.add_argument("--n_obj", type=int, default=2)
    p.add_argument("--r", type=float, default=0.09); p.add_argument("--dt", type=float, default=0.045)
    p.add_argument("--bounce", type=int, default=0, help="0=ligne droite (prédictible, démo System-2), 1=rebonds (dur)")
    p.add_argument("--d_model", type=int, default=256); p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--reg_w", type=float, default=1.0); p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--n_mask", type=int, default=2); p.add_argument("--pred_k", type=int, default=2)
    p.add_argument("--ssl_steps", type=int, default=5000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--t_obs", type=int, default=5)
    p.add_argument("--aspeed", type=float, default=3.3); p.add_argument("--n_dirs", type=int, default=16)
    p.add_argument("--r_safe", type=float, default=0.22); p.add_argument("--r_hit", type=float, default=0.16)
    p.add_argument("--tol", type=float, default=0.08); p.add_argument("--w_avoid", type=float, default=50.0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    nP = a.H // a.patch; npf = nP * nP; ntok = a.T * npf; d = a.d_model
    print(f"device={dev}  agent + {a.n_obj} obstacles  H={a.H} -> {ntok} tokens/clip", flush=True)
    F0, S, C = gen_clips(a.n, a.T, a.n_obj, a.H, r=a.r, dt=a.dt, seed=0, bounce=bool(a.bounce))   # vidéos d'obstacles (SSL)
    Xt = torch.tensor(patchify(F0, a.patch)); obs = Xt.shape[2]
    g = torch.Generator().manual_seed(1); pm = torch.randperm(a.n, generator=g)
    tr = pm[:int(0.85 * a.n)].numpy()

    # ---- world model : V-JEPA SSL (sans étiquettes) ----
    torch.manual_seed(0); m = VJEPA(obs, d, ntok, a.n_layer, a.n_head, a.reg_w, a.pred_layers).to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr); rng = np.random.default_rng(0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.ssl_steps)        # LR -> 0 : convergence stable
    for st in range(a.ssl_steps):
        bi = tr[np.random.randint(0, len(tr), a.bs)]; o = Xt[bi].to(dev)
        if np.random.rand() < 0.5:
            masks = [mk.to(dev) for mk in tube_masks(a.bs, a.T, nP, a.mask_ratio, a.n_mask, rng)]
        else:
            t0 = np.random.randint(2, a.T - 1); masks = [temporal_mask(a.bs, a.T, nP, t0, dev)]  # contextes de longueur VARIÉE
        loss = m(o, masks); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if st % 500 == 0: print(f"  [SSL] step {st} loss {loss.item():.3f}", flush=True)
    for prm in m.parameters(): prm.requires_grad = False

    # ---- forward model HONNÊTE : context_latents(0..t0) -> positions futures t0+1..t0+nf ----
    # train ET inférence utilisent le MÊME encodage (encodeur ne voit que 0..t0) -> pas de décalage, pas de fuite.
    P = torch.tensor(S[..., :2].reshape(a.n, a.T, -1)).float()
    Xtr = Xt[tr]; Ptr = P[tr]; fd = 2 * d                                       # descripteur par frame = moyenne+max -> 2d
    nf = a.T - 1 - a.t_obs                                                      # frames à imaginer
    fwd = nn.Sequential(nn.Linear(2 * fd, 512), nn.GELU(), nn.Linear(512, nf * a.n_obj * 2)).to(dev)
    ofw = torch.optim.Adam(fwd.parameters(), 1e-3)
    for _ in range(4000):
        t0 = np.random.randint(2, a.t_obs + 1)                                  # contexte >=3 frames (cohérent avec l'encodeur)
        bi = np.random.randint(0, len(tr), 256)
        cl = m.context_latents(Xtr[bi].to(dev), t0, npf)                        # (256, t0+1, 2d) honnête
        zin = torch.cat([cl[:, t0 - 1], cl[:, t0]], -1)                         # 2 derniers latents observés
        tgt = Ptr[bi][:, t0 + 1:t0 + 1 + nf].reshape(256, -1).to(dev)
        ofw.zero_grad(); F.mse_loss(fwd(zin), tgt).backward(); ofw.step()

    # ---- ÉPISODES de navigation (mêmes obstacles pour les 3 politiques) ----
    Fe, Se, _ = gen_clips(a.n_test, a.T, a.n_obj, a.H, r=a.r, dt=a.dt, seed=777, bounce=bool(a.bounce))  # obstacles test
    true_obs = Se[..., :2]                                                       # (n_test,T,n_obj,2)
    Xe = torch.tensor(patchify(Fe, a.patch))
    with torch.no_grad():                                                        # encodage HONNÊTE des frames 0..t_obs
        cl = torch.cat([m.context_latents(Xe[i:i+64].to(dev), a.t_obs, npf).cpu() for i in range(0, a.n_test, 64)])
        zin = torch.cat([cl[:, a.t_obs - 1], cl[:, a.t_obs]], -1).to(dev)
        imag = fwd(zin).reshape(a.n_test, nf, a.n_obj, 2).cpu().numpy()           # obstacles IMAGINÉS
    start = np.array([0.05, 0.5], np.float32); target = np.array([0.95, 0.5], np.float32)
    dirs = unit_dirs(a.n_dirs)
    def field(kind, i):                                                          # positions obstacles supposées (T,n_obj,2)
        f = true_obs[i].copy()
        if kind == "s1": f[a.t_obs + 1:] = true_obs[i, a.t_obs]                   # figé (System-1)
        elif kind == "s2": f[a.t_obs + 1:] = imag[i]                             # imaginé (System-2)
        return f
    res = {}
    for name, kind in [("naïf (ignore)", None), ("System-1 (positions actuelles)", "s1"),
                       ("System-2 (positions imaginées)", "s2")]:
        hits = reach = 0
        for i in range(a.n_test):
            po = None if kind is None else field(kind, i)
            h, rc = run_episode(start, target, po, true_obs[i], a.t_obs, a, dirs)
            hits += h; reach += rc
        res[name] = (hits / a.n_test, reach / a.n_test)

    # qualité de l'imagination (diagnostic) : R² des positions futures imaginées vs vraies
    gt = true_obs[:, a.t_obs + 1:].reshape(-1, a.n_obj * 2)
    r2_imag = r2(torch.tensor(imag.reshape(-1, a.n_obj * 2)), torch.tensor(gt))
    frozen = np.repeat(true_obs[:, a.t_obs:a.t_obs + 1], nf, 1).reshape(-1, a.n_obj * 2)
    r2_froz = r2(torch.tensor(frozen), torch.tensor(gt))

    print(f"\n=== SYSTEM-2 : NAVIGUER PARMI DES OBSTACLES MOBILES ({a.n_test} épisodes) ===", flush=True)
    print(f"  imagination obstacles (R² futur) : imaginé={r2_imag:.2f}  vs  figé={r2_froz:.2f}", flush=True)
    for name, (col, rc) in res.items():
        print(f"  {name:34s} : collisions {col:.0%}   arrivées {rc:.0%}", flush=True)

if __name__ == "__main__":
    main()
