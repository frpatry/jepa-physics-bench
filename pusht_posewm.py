"""
PUSH-T — WORLD MODEL BASÉ POSE : perception par GÉOMÉTRIE, dynamique et planif en NOMBRES.

Virage final (insight utilisateur) : on arrête de compresser en slots (qui jettent l'orientation)
et de repeindre (qui refloute). On lit la POSE directement des pixels par géométrie — validé :
agent 0.7px, bloc 7.6px, ANGLE 1° médian (vs slots 89° = hasard). L'orientation ÉTAIT toujours
dans les pixels ; les slots la jetaient.

Trois briques :
  1. PERCEPTION = géométrie (aucun apprentissage) : masques couleur (agent bleu / bloc gris /
     cible verte) -> centroïde = position, moments 2e+3e ordre = angle du bloc.
  2. DYNAMIQUE = petit MLP appris sur les données de jeu : (pose bloc, pose agent, action cible)
     -> (pose bloc, pose agent) au pas suivant. Espace de quelques nombres, net, pas de pixels.
  3. PLANIF = CEM dans l'espace des poses : on imagine les poses futures, on compare à la pose
     but (position + angle). MPC vs ORACLE (vrai sim) vs aléatoire, succès = coverage officiel.

  python pusht_posewm.py --steps 8000 --tasks 10
"""
import argparse, math, os
import numpy as np, torch, torch.nn as nn
from pusht_plan import make_env, reset_faithful, oracle_plan

# ---------- BRIQUE 1 : perception géométrique (pixels -> pose, aucun apprentissage) ----------
def _moment_pose(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 5: return None
    x = xs - xs.mean(); y = ys - ys.mean()
    mu20 = (x * x).mean(); mu02 = (y * y).mean(); mu11 = (x * y).mean()
    phi = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    u = x * np.cos(phi) + y * np.sin(phi)
    if (u ** 3).mean() < 0: phi += np.pi                                  # 3e moment : sens de la tige (360°)
    return xs.mean(), ys.mean(), phi

def pose_from_image(img, S=512):
    """img HxHx3 (float 0..1 ou uint8). -> (bx, by, bangle, ax, ay) en unités env [0,S], ou None."""
    if img.dtype == np.uint8: img = img.astype(np.float32) / 255
    H = img.shape[0]; R, G, Bl = img[..., 0], img[..., 1], img[..., 2]
    agent = (Bl > 0.8) & (R < 0.55) & (G < 0.75)
    green = (G > 0.75) & (R < 0.75) & (Bl < 0.75)
    block = (R > 0.35) & (R < 0.72) & (np.abs(R - G) < 0.12) & (Bl - R > 0.02) & (Bl - R < 0.30) & (~green)
    pa = _moment_pose(agent); pb = _moment_pose(block)
    if pa is None or pb is None: return None
    k = S / H
    return np.array([pb[0] * k, pb[1] * k, pb[2], pa[0] * k, pa[1] * k], np.float32)

# ---------- BRIQUE 2 : dynamique apprise sur les poses ----------
class PoseDyn(nn.Module):
    """(bloc x,y,sinθ,cosθ ; agent x,y ; action cible dx,dy) -> Δ(bloc 4 ; agent 2). Résiduel."""
    def __init__(s, hid=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(8, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                              nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 6))
    def forward(s, blk, ag, act):                                        # blk:(B,4) ag:(B,2) act:(B,2)
        d = s.net(torch.cat([blk, ag, act], -1))
        nb = blk + d[:, :4]
        nb = torch.cat([nb[:, :2], nb[:, 2:4] / (nb[:, 2:4].norm(dim=-1, keepdim=True) + 1e-8)], -1)
        return nb, ag + d[:, 4:6]

def to_state(bp, ag, S=512.0):
    """poses env -> vecteurs normalisés. bp:(...,3) ag:(...,2)."""
    blk = np.concatenate([bp[..., :2] / S, np.sin(bp[..., 2:3]), np.cos(bp[..., 2:3])], -1)
    return blk.astype(np.float32), (ag / S).astype(np.float32)

# ---------- BRIQUE 3 : planification dans l'espace des poses ----------
def pose_cost(blk, gblk, w_ang=0.3):
    pos = (blk[:, :2] - gblk[:2]).norm(dim=-1)
    ang = (blk[:, 2:4] - gblk[2:4]).norm(dim=-1)                          # distance (sinθ,cosθ)
    return pos + w_ang * ang

def cem_pose(dyn, blk0, ag0, gblk, Hp=4, pop=192, iters=3, elite=16, amax=0.5, dev="cpu", mu0=None):
    with torch.no_grad():
        blk = blk0.expand(pop, -1).clone(); ag = ag0.expand(pop, -1).clone()
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.25, device=dev)
        for _ in range(iters):
            eps = torch.randn(pop, Hp, 2, device=dev)
            eps[: pop // 2] = torch.randn(pop // 2, 1, 2, device=dev)      # bruit corrélé (iCEM)
            A = (mu + sg * eps).clamp(-amax, amax)
            b, a, cost = blk, ag, 0.
            for h in range(Hp):
                b, a = dyn(b, a, A[:, h])
                if h >= Hp - 2:
                    cost = cost + pose_cost(b, gblk)
                    cost = cost + 0.2 * (a - b[:, :2]).norm(dim=-1)        # approche (agent près du bloc)
            el = A[cost.argsort()[:elite]]
            mu = el.mean(0); sg = (el.std(0) + 1e-4).clamp_min(0.05)
    return mu.clamp(-amax, amax)

def calib_offset(X, BP, ns=400):
    """Décalage CONSTANT (repère du bloc) entre centroïde visuel et référence BP (pose pymunk).
    Mesuré ~(0,41)px, écart-type ~2.5 -> le convertir permet à la perception de viser la même
    référence que la dynamique et le but."""
    loc = []
    n, T = X.shape[0], X.shape[1]
    for i in range(min(ns, n)):
        for t in range(T):
            p = pose_from_image(X[i, t])
            if p is None: continue
            dx = p[0] - BP[i, t, 0]; dy = p[1] - BP[i, t, 1]; th = BP[i, t, 2]
            loc.append([dx * np.cos(-th) - dy * np.sin(-th), dx * np.sin(-th) + dy * np.cos(-th)])
    return np.array(loc).mean(0)                                         # (ox, oy) en repère bloc

def run_episode(dyn, seed, dev, policy, max_steps=100, plan_h=4, scratch=None, offset=np.zeros(2)):
    env = make_env(); rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    gblk = torch.tensor(np.concatenate([gp[:2] / 512, [np.sin(gp[2]), np.cos(gp[2])]]), dtype=torch.float32, device=dev)
    ag = np.array(obs["agent_pos"], np.float32); best = float(info.get("coverage", 0.0)); mu = None; mu_np = None
    for _ in range(max_steps):
        if policy == "mpc":
            pose = pose_from_image(obs["pixels"])                        # perception géométrique
            if pose is None: bp = np.array([*info["block_pose"]], np.float32)  # secours si masque perdu
            else:
                th = pose[2]; ox, oy = offset                            # centroïde -> référence BP (calibré)
                bx = pose[0] - (ox * np.cos(th) - oy * np.sin(th))
                by = pose[1] - (ox * np.sin(th) + oy * np.cos(th))
                bp = np.array([bx, by, th], np.float32)
            blk, agn = to_state(bp[None], ag[None]); blk = torch.tensor(blk, device=dev); agn = torch.tensor(agn, device=dev)
            plan = cem_pose(dyn, blk, agn, gblk, Hp=plan_h, dev=dev, mu0=mu)
            delta = plan[0].cpu().numpy(); mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])
            act = np.clip(ag + delta * 512.0, 0, 512).astype(np.float32)
        elif policy == "oracle":
            if mu_np is not None and len(mu_np): d0 = mu_np[0]; mu_np = None
            else:
                st = np.array([ag[0], ag[1], *np.array(info["block_pose"], np.float32)], np.float32)
                pl = oracle_plan(scratch, st, gp, rng, Hp=plan_h, pop=32, iters=2); d0 = pl[0]; mu_np = pl[1:2]
            act = np.clip(ag + d0 * 256.0, 0, 512).astype(np.float32)
        else:
            th = rng.uniform(0, 2 * np.pi); act = np.clip(ag + rng.uniform(0.2, 1.0) * 256 *
                                                          np.array([np.cos(th), np.sin(th)]), 0, 512).astype(np.float32)
        obs, _, term, trunc, info = env.step(act); ag = np.array(obs["agent_pos"], np.float32)
        best = max(best, float(info.get("coverage", 0.0)))
        if info.get("is_success", False) or term or trunc: break
    env.close(); return best

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=""); p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--bs", type=int, default=128); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tasks", type=int, default=10); p.add_argument("--plan_h", type=int, default=4)
    p.add_argument("--policies", type=str, default="mpc,oracle,random"); p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from slots_pusht import default_path
    data = a.data or default_path("pusht_data.npz")
    d = np.load(data); X, A, AG, BP = d["X"], d["A"], d["AG"], d["BP"]
    n, T = X.shape[0], X.shape[1]
    # validation perception (échantillon)
    ea, eb, pr, tr = [], [], [], []
    for i in range(min(200, n)):
        p = pose_from_image(X[i, 0])
        if p is None: continue
        ea.append(np.linalg.norm(p[3:5] - AG[i, 0])); eb.append(np.linalg.norm(p[:2] - BP[i, 0, :2]))
        pr.append(p[2]); tr.append(BP[i, 0, 2])
    pr, tr = np.array(pr), np.array(tr); off = np.angle(np.mean(np.exp(1j * (pr - tr))))
    eang = np.degrees(np.abs(np.angle(np.exp(1j * (pr - tr - off)))))
    offset = calib_offset(X, BP)
    print(f"device={dev}  PERCEPTION géométrique : agent {np.mean(ea):.1f}px  bloc {np.mean(eb):.1f}px"
          f"  angle méd {np.median(eang):.0f}°  |  décalage calibré ({offset[0]:.0f},{offset[1]:.0f})px", flush=True)
    # BRIQUE 2 : entraîner la dynamique sur les vraies poses (perception validée ≈ vérité)
    blk, ag = to_state(BP, AG)                                            # (n,T,4),(n,T,2)
    dA = np.clip((A - AG[:, :-1]) / 512.0, -1, 1).astype(np.float32)      # action normalisée
    blk = torch.tensor(blk).to(dev); ag = torch.tensor(ag).to(dev); act = torch.tensor(dA).to(dev)
    dyn = PoseDyn().to(dev); opt = torch.optim.Adam(dyn.parameters(), a.lr)
    ne = min(300, max(1, n // 10)); ntr = n - ne
    print("--- dynamique (pose,action)->pose ---", flush=True)
    for st in range(a.steps):
        bi = np.random.randint(0, ntr, a.bs); ti = np.random.randint(0, T - 1, a.bs)
        b0 = blk[bi, ti]; a0 = ag[bi, ti]; ac = act[bi, ti]
        nb, na = dyn(b0, a0, ac)
        loss = ((nb - blk[bi, ti + 1]) ** 2).sum(-1).mean() + ((na - ag[bi, ti + 1]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 1000 == 0:
            with torch.no_grad():
                bv, tv = blk[ntr:], ag[ntr:]
                nb, _ = dyn(bv[:, 0], tv[:, 0], act[ntr:, 0])
                ep = (nb[:, :2] - bv[:, 1, :2]).norm(dim=-1).mean().item() * 512
                ec = (bv[:, 0, :2] - bv[:, 1, :2]).norm(dim=-1).mean().item() * 512   # copie (bloc statique)
            print(f"  step {st:5d}  bloc t+1 : imaginé {ep:.1f}px  vs copie {ec:.1f}px", flush=True)
    # BRIQUE 3 : planification
    if a.tasks > 0:
        scratch = make_env(); scratch.reset(seed=0)
        pols = a.policies.split(","); sc = {q: [] for q in pols}
        for k in range(a.tasks):
            line = []
            for q in pols:
                cov = run_episode(dyn, 3000 + k, dev, q, plan_h=a.plan_h, scratch=scratch, offset=offset)
                sc[q].append(cov); line.append(f"{q} {cov:.2f}")
            print(f"  tâche {k:2d}  " + "  |  ".join(line), flush=True)
        print("\nCOVERAGE MAX MOYENNE : " + "  |  ".join(
            f"{q} {np.mean(sc[q]):.2f} (succès>0.95 {sum(v>0.95 for v in sc[q])}/{a.tasks})" for q in pols), flush=True)
        scratch.close()

if __name__ == "__main__":
    main()
