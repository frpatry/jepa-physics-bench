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

BLANC = torch.tensor([0.97, 0.97, 0.97]); VERT = torch.tensor([0.70, 0.90, 0.70])

def soft_color_w(img, ref, tau=0.15):
    """Assignation COMPÉTITIVE : chaque pixel du canvas est réparti par softmax entre les 4
    couleurs de la scène (blanc/gris/bleu/vert) — un pixel mi-bleu mi-gris va au plus proche,
    les cartes gris et bleu ne peuvent plus co-brûler sur le même mélange (leçon diag : avec
    des similarités indépendantes, le flou faisait cT≈cA -> coût dégénéré). Lecture du canvas
    décodé, aucune dépendance à l'identité des slots."""
    refs = torch.stack([BLANC, GRIS, BLEU, VERT]).to(img.device)           # (4,3)
    d2 = ((img.unsqueeze(1) - refs.reshape(1, 4, 3, 1, 1)) ** 2).sum(2)    # (B,4,H,W)
    w = torch.softmax(-d2 / (2 * tau * tau), dim=1)
    idx = {"blanc": 0, "gris": 1, "bleu": 2, "vert": 3}[ref]
    out = w[:, idx] * (1.0 - w[:, 0])                                      # atténuer ce qui est surtout blanc
    out[..., :2, :] = 0; out[..., -2:, :] = 0; out[..., :, :2] = 0; out[..., :, -2:] = 0
    return out

def gray_mask(img64):
    """Pixels du bloc T dans une image (1,3,H,W) : gris = canaux proches, luminance moyenne.
    Prétraitement de la SPÉCIFICATION du but (l'image but est donnée), pas de la perception."""
    x = img64[0]
    spread = x.max(0).values - x.min(0).values
    lum = x.mean(0)
    m = ((spread < 0.12) & (lum > 0.35) & (lum < 0.80)).float()
    m[:2] = 0; m[-2:] = 0; m[:, :2] = 0; m[:, -2:] = 0                     # exclure la bordure
    return m                                                               # (H,W)

def t_stats(canvas, dev=None):
    """Stats du T dans un canvas, AVEC EXCLUSION d'un disque autour de l'agent (centroïde bleu) :
    dans le flou, les pixels du mélange bleu-gris se répartissent entre les deux cartes -> la
    carte grise gagne de la masse LÀ OÙ EST L'AGENT et son centroïde est tiré vers lui — 'se
    garer entre le T et le but' ressemblait à du progrès (contamination, MPC figé à 0.11)."""
    H = canvas.shape[-1]
    wB = soft_color_w(canvas, "bleu"); cA, _ = mask_stats(wB)
    wG = soft_color_w(canvas, "gris")
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    gx = torch.tensor(xx).to(canvas.device); gy = torch.tensor(yy).to(canvas.device)
    d2 = (gx.unsqueeze(0) - cA[:, 0].reshape(-1, 1, 1)) ** 2 + (gy.unsqueeze(0) - cA[:, 1].reshape(-1, 1, 1)) ** 2
    wG = wG * (d2 > 0.07 ** 2).float()                                     # exclusion autour de l'agent
    cT, CT_ = mask_stats(wG)
    return cT, CT_, cA

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

def cem_plan(m, x0, x1, cg, Cg, Hp=4, pop=192, iters=3, elite=16, amax=0.8, dev="cpu", mu0=None):
    """Leçon finale (trace diag 3) : élite CEM à 0.05 < meilleur plan honnête 0.079 = EXPLOITATION
    adversariale — des séquences d'actions erratiques hors distribution font dériver l'imagination.
    Parade : planifier DANS LE SUPPORT des données d'entraînement — horizon 4 (= horizon entraîné
    de g, au-delà rien n'est certifié) et candidats PERSISTANTS PAR MORCEAUX (2 segments constants,
    la forme exacte des actions de jeu), 3 itérations (l'over-optimization nuit)."""
    with torch.no_grad():
        _, _, S0 = m.peel(m.feats(x0)); _, _, S1 = m.peel(m.feats(x1), init=S0)
        S0 = S0.expand(pop, -1, -1); S1 = S1.expand(pop, -1, -1)
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.30, device=dev)
        for _ in range(iters):
            e1 = torch.randn(pop, 1, 2, device=dev).expand(-1, Hp, -1).clone()   # constant
            e2 = torch.randn(pop, 2, 2, device=dev).repeat_interleave((Hp + 1) // 2, 1)[:, :Hp]  # 2 segments
            eps = torch.where((torch.arange(pop, device=dev) % 2 == 0).reshape(-1, 1, 1), e1, e2)
            A = (mu + sg * eps).clamp(-amax, amax)
            prev, cur, cost = S0, S1, 0.
            for h in range(Hp):
                nxt = m.step_a(cur, prev, A[:, h]); prev, cur = cur, nxt
                if h >= Hp - 2:
                    mm, rr = m.imagine(cur)                                # décodage POST-g (calibré)
                    canvas = (rr * mm).sum(1)                              # (pop,3,H,W) : le canvas imaginé
                    cT, CT_, cA = t_stats(canvas)                          # stats T avec exclusion agent
                    cost = cost + pose_cost(cT, CT_, cg, Cg)
                    cost = cost + 0.25 * ((cA - cT).norm(dim=-1) - 0.12).clamp_min(0.)  # approche
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

def run_episode(m, hin, seed, dev, policy, max_steps=100, plan_h=8, pop=192, iters=5, scratch=None):
    env = make_env()
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    # image BUT : le T posé sur la cible, agent parqué loin (via un env scratch)
    far = np.array([40.0, 40.0], np.float32) if np.linalg.norm(gp[:2] - 40) > 120 else np.array([470.0, 470.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    g64 = to64(gob["pixels"], hin, dev)
    cg, Cg = mask_stats(soft_color_w(g64, "gris")); cg, Cg = cg[0], Cg[0]
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
            if mu_np is not None and len(mu_np):                           # exécuter 2 actions par plan
                delta = mu_np[0]; mu_np = None
            else:
                state = np.array([ag[0], ag[1], *np.array(info["block_pose"], np.float32)], np.float32)
                plan = oracle_plan(scratch, state, gp, rng, Hp=plan_h, pop=32, iters=2)
                delta = plan[0]; mu_np = plan[1:2]
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

def diag(m, hin, dev, scratch, seed=3000):
    """Les yeux du MPC : image but + gray_mask + couleurs décodées des slots + masques jT/jA
    (peel) + masque jT imaginé après 2 pas. Sauve pusht_plan_diag.png."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    env = make_env(); obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    far = np.array([40.0, 40.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    g64 = to64(gob["pixels"], hin, dev)
    cg, _ = mask_stats(soft_color_w(g64, "gris"))
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    obs, _, _, _, info = env.step(ag.copy()); f1 = obs["pixels"]
    x0 = to64(f0, hin, dev); x1 = to64(f1, hin, dev)
    with torch.no_grad():
        _, _, S0 = m.peel(m.feats(x0))
        mk, rgb, S1 = m.peel(m.feats(x1), init=S0)
        col = (rgb * mk).sum((-1, -2)) / (mk.sum((-1, -2)) + 1e-8)
        print("couleurs décodées des prises :", np.round(col[0, :-1].cpu().numpy(), 2).tolist(), flush=True)
        jT, jA, _, _ = slot_ids(m, x0, x1, dev)
        print(f"jT={jT} jA={jA}  |  cg (but, décodé)={np.round(cg[0].cpu().numpy(),3)}"
              f"  vs goal vrai/512={np.round(gp[:2]/512.,3)}", flush=True)
        prev, cur = S0, S1
        for _ in range(2): nxt = m.step_a(cur, prev, torch.zeros(1, 2, device=dev)); prev, cur = cur, nxt
        mi, _ = m.imagine(cur)
    with torch.no_grad():
        mi_m, mi_r = m.imagine(cur); canvas = (mi_r * mi_m).sum(1)
    fig, ax = plt.subplots(1, 6, figsize=(18, 3))
    ax[0].imshow(g64[0].permute(1, 2, 0).cpu()); ax[0].set_title("image BUT")
    ax[1].imshow(soft_color_w(g64, "gris")[0].cpu()); ax[1].set_title("poids gris (cible)")
    ax[2].imshow(x1[0].permute(1, 2, 0).cpu()); ax[2].set_title("frame courante")
    ax[3].imshow(canvas[0].permute(1, 2, 0).cpu().clip(0, 1)); ax[3].set_title("canvas imaginé (+2)")
    ax[4].imshow(soft_color_w(canvas, "gris")[0].cpu()); ax[4].set_title("poids gris (canvas)")
    ax[5].imshow(soft_color_w(canvas, "bleu")[0].cpu()); ax[5].set_title("poids bleu (canvas)")
    for a_ in ax: a_.axis("off")
    out = "/content/pusht_plan_diag.png" if os.path.isdir("/content") else "pusht_plan_diag.png"
    plt.tight_layout(); plt.savefig(out); print(f"diag -> {out}", flush=True)
    env.close()

def diag_plans(m, hin, dev, scratch, seed=3000):
    """DIAG D pusht : 3 plans connus (pousser VERS le but / À L'ENVERS / S'ÉLOIGNER), notés par
    le coût imaginé ET par le vrai simulateur. Si le coût ne classe pas 'vers' devant les autres,
    l'imagination ne crédite pas les poussées dans l'espace couleur-canvas."""
    env = make_env(); obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    bp = np.array(info["block_pose"], np.float32)
    u = (gp[:2] - bp[:2]) / (np.linalg.norm(gp[:2] - bp[:2]) + 1e-8)
    # téléporter l'agent au point de poussée (derrière le bloc par rapport au but)
    pa = np.clip(bp[:2] - u * 60.0, 15, 497)
    obs, info = reset_faithful(env, np.array([pa[0], pa[1], bp[0], bp[1], bp[2]], np.float32))
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    obs, _, _, _, info = env.step(ag.copy()); f1 = obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
    # but (image -> stats)
    far = np.array([40.0, 40.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    cg, Cg = mask_stats(soft_color_w(to64(gob["pixels"], hin, dev), "gris")); cg, Cg = cg[0], Cg[0]
    Hp = 8
    plans = {"vers": np.repeat((u * 0.3)[None], Hp, 0).astype(np.float32),
             "envers": np.repeat((-u * 0.3)[None], Hp, 0).astype(np.float32),
             "s_eloigner": np.repeat((np.array([-u[1], u[0]]) * 0.5)[None], Hp, 0).astype(np.float32)}
    x0 = to64(f0, hin, dev); x1 = to64(f1, hin, dev)
    with torch.no_grad():
        _, _, S0 = m.peel(m.feats(x0)); _, _, S1 = m.peel(m.feats(x1), init=S0)
    st0 = np.array([ag[0], ag[1], *np.array(info["block_pose"], np.float32)], np.float32)
    for nom, A_ in plans.items():
        with torch.no_grad():
            prev, cur = S0, S1
            gs = []
            for h in range(Hp):
                nxt = m.step_a(cur, prev, torch.tensor(A_[h:h + 1]).to(dev)); prev, cur = cur, nxt
                if h >= Hp - 2:
                    mm, rr = m.imagine(cur); canvas = (rr * mm).sum(1)
                    cT, CT_, _ = t_stats(canvas)
                    gs.append(float(pose_cost(cT, CT_, cg, Cg)[0]))
        o2, i2 = reset_faithful(scratch, st0)
        a2 = np.array(o2["agent_pos"], np.float32)
        for h in range(Hp):
            o2, _, te, tr, i2 = scratch.step(np.clip(a2 + A_[h] * 256.0, 0, 512).astype(np.float32))
            a2 = np.array(o2["agent_pos"], np.float32)
            if te or tr: break
        bp2 = np.array(i2["block_pose"], np.float32)
        print(f"  {nom:11s} coût imaginé {np.mean(gs):.3f}  |  vraie dist bloc-but {np.linalg.norm(bp2[:2]-gp[:2])/512.:.3f}"
              f"  (départ {np.linalg.norm(bp[:2]-gp[:2])/512.:.3f})", flush=True)
    # volet DÉPART LOINTAIN : le plan-exploit 'se placer côté but sans toucher' bat-il encore ?
    pa2 = np.clip(bp[:2] - u * 180.0, 15, 497)
    obs, info = reset_faithful(env, np.array([pa2[0], pa2[1], bp[0], bp[1], bp[2]], np.float32))
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    obs, _, _, _, info = env.step(ag.copy()); f1 = obs["pixels"]
    x0 = to64(f0, hin, dev); x1 = to64(f1, hin, dev)
    with torch.no_grad():
        _, _, S0 = m.peel(m.feats(x0)); _, _, S1 = m.peel(m.feats(x1), init=S0)
    v = (gp[:2] - bp[:2]); v = v / (np.linalg.norm(v) + 1e-8)
    park = np.clip(bp[:2] + v * 80.0, 15, 497)                             # côté BUT, sans toucher
    plans2 = {"approche+pousse": np.repeat((u * 0.35)[None], Hp, 0).astype(np.float32),
              "parking_cote_but": np.repeat(((park - ag) / (256.0 * Hp) * Hp)[None] * 0 +
                                            ((park - ag) / 256.0 / Hp)[None], Hp, 0).astype(np.float32),
              "rester_loin": np.zeros((Hp, 2), np.float32)}
    print("  --- départ lointain ---", flush=True)
    for nom, A_ in plans2.items():
        with torch.no_grad():
            prev, cur = S0, S1; gs = []
            for h in range(Hp):
                nxt = m.step_a(cur, prev, torch.tensor(np.clip(A_[h:h+1], -0.8, 0.8)).to(dev)); prev, cur = cur, nxt
                if h >= Hp - 2:
                    mm, rr = m.imagine(cur); canvas = (rr * mm).sum(1)
                    cT, CT_, _ = t_stats(canvas)
                    gs.append(float(pose_cost(cT, CT_, cg, Cg)[0]))
        print(f"  {nom:17s} coût-but imaginé {np.mean(gs):.3f}", flush=True)
    env.close()

def diag_episode(m, hin, dev, scratch, seed=3000, steps=12):
    """Trace d'épisode MPC : à chaque replan, coût de l'élite CEM vs coût d'un plan-EXPERT
    injecté (aller derrière le bloc puis pousser vers le but), directions et progrès réels."""
    env = make_env(); obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    far = np.array([40.0, 40.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    g64 = to64(gob["pixels"], hin, dev)
    cg, Cg = mask_stats(soft_color_w(g64, "gris")); cg, Cg = cg[0], Cg[0]
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    obs, _, _, _, info = env.step(ag.copy()); f1 = obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
    mu = None; Hp = 4
    def eval_plan(x0, x1, A):
        with torch.no_grad():
            _, _, S0 = m.peel(m.feats(x0)); _, _, S1 = m.peel(m.feats(x1), init=S0)
            prev, cur, c = S0, S1, 0.
            for h in range(Hp):
                nxt = m.step_a(cur, prev, A[h:h + 1].to(dev)); prev, cur = cur, nxt
                if h >= Hp - 2:
                    mm, rr = m.imagine(cur); canvas = (rr * mm).sum(1)
                    cT, CT_, cA = t_stats(canvas)
                    c = c + float(pose_cost(cT, CT_, cg, Cg)[0]) + 0.25 * max(float((cA - cT).norm()) - 0.12, 0.)
        return c
    for t in range(steps):
        bp = np.array(info["block_pose"], np.float32)
        u = (gp[:2] - bp[:2]) / (np.linalg.norm(gp[:2] - bp[:2]) + 1e-8)
        push_pt = bp[:2] - u * 55.0
        x0 = to64(f0, hin, dev); x1 = to64(f1, hin, dev)
        plan = cem_plan(m, x0, x1, cg, Cg, Hp=Hp, pop=192, iters=3, dev=dev, mu0=mu)
        c_cem = eval_plan(x0, x1, plan)
        # plan expert : moitié des pas vers le point de poussée, moitié à pousser
        exp = np.zeros((Hp, 2), np.float32)
        for h in range(Hp):
            exp[h] = (push_pt - ag) / 256.0 / 3.0 if h < 4 else u * 0.3
        exp = np.clip(exp, -0.8, 0.8)
        c_exp = eval_plan(x0, x1, torch.tensor(exp))
        d = plan[0].cpu().numpy()
        to_block = (bp[:2] - ag) / (np.linalg.norm(bp[:2] - ag) + 1e-8)
        print(f"  pas {t:2d}  cov {info.get('coverage',0):.2f}  d(ag,bloc) {np.linalg.norm(ag-bp[:2]):3.0f}px"
              f"  |δ| {np.linalg.norm(d):.2f}  δ·versbloc {float(d @ to_block):+.2f}"
              f"  |  coût élite {c_cem:.3f}  vs EXPERT {c_exp:.3f}", flush=True)
        mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])
        obs, _, term, trunc, info = env.step(np.clip(ag + d * 256.0, 0, 512).astype(np.float32))
        f0, f1 = f1, obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
        if term or trunc: break
    env.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=""); p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--plan_h", type=int, default=4); p.add_argument("--plan_pop", type=int, default=192)
    p.add_argument("--plan_iters", type=int, default=3); p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--policies", type=str, default="mpc,oracle,random")
    p.add_argument("--diag", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = a.ckpt or default_path("pusht_wm.pt")
    ck = torch.load(ckpt, map_location=dev); sa = ck["args"]
    m = WM(sa["hin"], sa["K"], sa["D"], res=sa["hin"] // 2, slot_dim=sa["slot_dim"],
           dec_w=sa["dec_w"], iters=sa["iters"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    print(f"modèle chargé <- {ckpt} (hin {sa['hin']}, slot_dim {sa['slot_dim']})", flush=True)
    scratch = make_env(); scratch.reset(seed=0)
    if a.diag:
        diag(m, sa["hin"], dev, scratch)
        if a.diag >= 2: diag_plans(m, sa["hin"], dev, scratch)
        if a.diag >= 3: diag_episode(m, sa["hin"], dev, scratch)
        scratch.close(); return
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
