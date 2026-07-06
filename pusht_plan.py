"""
PUSH-T — BLOC 3 : planification MPC avec le world model slots (protocole DINO-WM).

Tâche = celle de l'env : pousser le T sur la zone cible verte. But spécifié par IMAGE (une scène
rendue avec le T posé sur la cible), ≤25 pas, succès = critère coverage officiel de gym-pusht.

Leçons appliquées (toy-Push + bloc 2) :
  - coût 100% en espace DÉCODÉ certifié : centroïde + covariance du masque du slot-T imaginé
    (la covariance porte l'ORIENTATION du T) vs stats des pixels gris de l'image but ;
  - on ne décode qu'APRÈS des pas de g (les alphas sont calibrés post-dynamique — DIAG PIPELINE) ;
  - CEM : bruit corrélé iCEM + plancher de bruit + budget modéré (l'over-optimization nuit) ;
  - shaping d'approche générique (agent proche du T) ;
  - ORACLE = même CEM sur le VRAI simulateur via reset_to_state (borne supérieure qui sépare
    erreur de modèle et erreur de recherche) ; baseline aléatoire sur les mêmes tâches.

  python pusht_plan.py --tasks 10
"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
from slots_pusht import WM, default_path

def make_env():
    import gymnasium as gym, gym_pusht  # noqa: F401
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

def to64(img, hin, dev):
    x = torch.tensor(img).to(dev).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return F.interpolate(x, size=hin, mode="area")

def reset_faithful(env, state):
    """reset_to_state place le bloc avec un décalage dépendant de l'angle (référence de pose
    différente entre l'API et info['block_pose'] — mesuré ~20-50px). Calibration par DOUBLE
    reset : on mesure l'erreur du premier, on compense au second. Exact, sans constante magique."""
    obs, info = env.reset(options={"reset_to_state": state.copy()})
    err = np.array(info["block_pose"], np.float32)[:2] - state[2:4]
    fixed = state.copy(); fixed[2:4] = state[2:4] - err
    return env.reset(options={"reset_to_state": fixed})

def gray_mask(img64):
    """Pixels du bloc T dans une image (1,3,H,W) : gris = canaux proches, luminance moyenne.
    Prétraitement de la SPÉCIFICATION du but (l'image but est donnée), pas de la perception."""
    x = img64[0]
    spread = x.max(0).values - x.min(0).values
    lum = x.mean(0)
    m = ((spread < 0.12) & (lum > 0.35) & (lum < 0.80)).float()
    m[:2] = 0; m[-2:] = 0; m[:, :2] = 0; m[:, -2:] = 0                     # exclure la bordure
    return m                                                               # (H,W)

def mask_stats(w, eps=1e-8):
    """Centroïde (x,y) et covariance 2x2 d'un masque-poids (B,H,W) -> (B,2), (B,2,2)."""
    B, H, _ = w.shape
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    gx = torch.tensor(xx).to(w.device); gy = torch.tensor(yy).to(w.device)
    s = w.sum((-1, -2)) + eps
    cx = (w * gx).sum((-1, -2)) / s; cy = (w * gy).sum((-1, -2)) / s
    dx = gx.unsqueeze(0) - cx.reshape(-1, 1, 1); dy = gy.unsqueeze(0) - cy.reshape(-1, 1, 1)
    cxx = (w * dx * dx).sum((-1, -2)) / s; cyy = (w * dy * dy).sum((-1, -2)) / s
    cxy = (w * dx * dy).sum((-1, -2)) / s
    C = torch.stack([torch.stack([cxx, cxy], -1), torch.stack([cxy, cyy], -1)], -2)
    return torch.stack([cx, cy], -1), C

def pose_cost(c, C, cg, Cg, w_ang=0.15):
    """Position + orientation (covariances normalisées par leur trace : forme pure)."""
    pos = (c - cg).norm(dim=-1)
    Cn = C / (C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) + 1e-8)
    Cgn = Cg / (Cg.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) + 1e-8)
    return pos + w_ang * (Cn - Cgn).flatten(-2).norm(dim=-1)

GRIS = torch.tensor([0.55, 0.58, 0.62]); BLEU = torch.tensor([0.35, 0.50, 0.90])

def slot_ids(m, x0, x1, dev):
    """Quel slot est le T, lequel est l'agent ? (couleur décodée — lecture du canvas, comme toy v4)."""
    with torch.no_grad():
        _, _, S0 = m.peel(m.feats(x0))
        mk, rgb, S1 = m.peel(m.feats(x1), init=S0)
        col = (rgb * mk).sum((-1, -2)) / (mk.sum((-1, -2)) + 1e-8)         # (1,K,3)
        grabs = col[0, :-1]
        jT = int((grabs - GRIS.to(dev)).norm(dim=-1).argmin())
        jA = int((grabs - BLEU.to(dev)).norm(dim=-1).argmin())
    return jT, jA, S0, S1

def cem_plan(m, x0, x1, cg, Cg, Hp=8, pop=192, iters=5, elite=16, amax=0.8, dev="cpu", mu0=None):
    with torch.no_grad():
        jT, jA, S0, S1 = slot_ids(m, x0, x1, dev)
        S0 = S0.expand(pop, -1, -1); S1 = S1.expand(pop, -1, -1)
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.30, device=dev)
        for _ in range(iters):
            eps = torch.randn(pop, Hp, 2, device=dev)
            eps[: pop // 2] = torch.randn(pop // 2, 1, 2, device=dev)      # bruit corrélé (iCEM)
            A = (mu + sg * eps).clamp(-amax, amax)
            prev, cur, cost = S0, S1, 0.
            for h in range(Hp):
                nxt = m.step_a(cur, prev, A[:, h]); prev, cur = cur, nxt
                if h >= Hp - 2:
                    mm, _ = m.imagine(cur)                                 # décodage POST-g (calibré)
                    cT, CT_ = mask_stats(mm[:, jT, 0])
                    cA, _ = mask_stats(mm[:, jA, 0])
                    cost = cost + pose_cost(cT, CT_, cg, Cg)
                    cost = cost + 0.15 * ((cA - cT).norm(dim=-1) - 0.12).clamp_min(0.)  # approche
                    # SATURANTE : récompense jusqu'au contact, puis plus rien — sinon percuter
                    # le bloc HORS de la cible est rentable quand il démarre dessus (toy v2 bis)
            el = A[cost.argsort()[:elite]]
            mu = el.mean(0); sg = (el.std(0) + 1e-4).clamp_min(0.08)       # plancher de bruit (iCEM)
    return mu.clamp(-amax, amax)

def oracle_plan(scratch, state, gp, rng, Hp=8, pop=48, iters=3, elite=8, amax=0.8):
    """Borne sup : même CEM, VRAI simulateur (reset_to_state), vrai coût de pose."""
    mu = np.zeros((Hp, 2), np.float32); sg = np.full((Hp, 2), 0.30, np.float32)
    for _ in range(iters):
        eps = rng.standard_normal((pop, Hp, 2)).astype(np.float32)
        eps[: pop // 2] = rng.standard_normal((pop // 2, 1, 2)).astype(np.float32)
        A = np.clip(mu + sg * eps, -amax, amax)
        cost = np.zeros(pop, np.float32)
        for i in range(pop):
            obs, info = reset_faithful(scratch, state)
            ag = np.array(obs["agent_pos"], np.float32)
            for h in range(Hp):
                obs, _, term, trunc, info = scratch.step(np.clip(ag + A[i, h] * 256.0, 0, 512).astype(np.float32))
                ag = np.array(obs["agent_pos"], np.float32)
                if term or trunc: break
            bp = np.array(info["block_pose"], np.float32)
            dang = abs((bp[2] - gp[2] + np.pi) % (2 * np.pi) - np.pi)
            cost[i] = (np.linalg.norm(bp[:2] - gp[:2]) / 512.0 + 0.10 * dang
                       + 0.10 * max(np.linalg.norm(ag - bp[:2]) / 512.0 - 0.10, 0.0))  # approche saturante
        el = A[np.argsort(cost)[:elite]]
        mu = el.mean(0); sg = np.maximum(el.std(0) + 1e-4, 0.08)
    return np.clip(mu, -amax, amax)

def run_episode(m, hin, seed, dev, policy, max_steps=25, plan_h=8, pop=192, iters=5, scratch=None):
    env = make_env()
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    # image BUT : le T posé sur la cible, agent parqué loin (via un env scratch)
    far = np.array([40.0, 40.0], np.float32) if np.linalg.norm(gp[:2] - 40) > 120 else np.array([470.0, 470.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    g64 = to64(gob["pixels"], hin, dev)
    gm = gray_mask(g64)
    cg, Cg = mask_stats(gm.unsqueeze(0)); cg, Cg = cg[0], Cg[0]
    # 2 frames de contexte (un pas immobile)
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    obs, _, _, _, info = env.step(ag.copy())
    f1 = obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
    best_cov, success, mu, mu_np = float(info.get("coverage", 0.0)), False, None, None
    for _ in range(max_steps):
        if policy == "mpc":
            x0 = to64(f0, hin, dev); x1 = to64(f1, hin, dev)
            plan = cem_plan(m, x0, x1, cg, Cg, Hp=plan_h, pop=pop, iters=iters, dev=dev, mu0=mu)
            delta = plan[0].cpu().numpy(); mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])
        elif policy == "oracle":
            state = np.array([ag[0], ag[1], *np.array(info["block_pose"], np.float32)], np.float32)
            plan = oracle_plan(scratch, state, gp, rng, Hp=plan_h)
            delta = plan[0]; mu_np = np.concatenate([plan[1:], np.zeros((1, 2), np.float32)])
        else:
            th = rng.uniform(0, 2 * np.pi); delta = (rng.uniform(0.2, 1.0) * 0.8 *
                                                     np.array([np.cos(th), np.sin(th)], np.float32))
        act = np.clip(ag + delta * 256.0, 0, 512).astype(np.float32)
        obs, _, term, trunc, info = env.step(act)
        f0, f1 = f1, obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
        best_cov = max(best_cov, float(info.get("coverage", 0.0)))
        if info.get("is_success", False) or term:
            success = True; break
        if trunc: break
    env.close()
    return success, best_cov

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=""); p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--plan_h", type=int, default=8); p.add_argument("--plan_pop", type=int, default=192)
    p.add_argument("--plan_iters", type=int, default=5); p.add_argument("--max_steps", type=int, default=25)
    p.add_argument("--policies", type=str, default="mpc,oracle,random")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = a.ckpt or default_path("pusht_wm.pt")
    ck = torch.load(ckpt, map_location=dev); sa = ck["args"]
    m = WM(sa["hin"], sa["K"], sa["D"], res=sa["hin"] // 2, slot_dim=sa["slot_dim"],
           dec_w=sa["dec_w"], iters=sa["iters"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    print(f"modèle chargé <- {ckpt} (hin {sa['hin']}, slot_dim {sa['slot_dim']})", flush=True)
    scratch = make_env(); scratch.reset(seed=0)
    pols = a.policies.split(",")
    scores = {q: [] for q in pols}; covs = {q: [] for q in pols}
    for k in range(a.tasks):
        line = []
        for q in pols:
            ok, cov = run_episode(m, sa["hin"], 3000 + k, dev, q, max_steps=a.max_steps,
                                  plan_h=a.plan_h, pop=a.plan_pop, iters=a.plan_iters, scratch=scratch)
            scores[q].append(ok); covs[q].append(cov)
            line.append(f"{q} {'OK ' if ok else 'échec'} (cov max {cov:.2f})")
        print(f"  tâche {k:2d}  " + "  |  ".join(line), flush=True)
    print("\nSUCCÈS (coverage>0.95 en ≤%d pas) : " % a.max_steps +
          "  |  ".join(f"{q} {sum(scores[q])}/{a.tasks} (cov moy {np.mean(covs[q]):.2f})" for q in pols), flush=True)
    scratch.close()

if __name__ == "__main__":
    main()
