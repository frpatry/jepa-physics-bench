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

def _masks(img):
    """(agent, bloc) masques booléens. Source unique des seuils couleur."""
    if img.dtype == np.uint8: img = img.astype(np.float32) / 255
    R, G, Bl = img[..., 0], img[..., 1], img[..., 2]
    agent = (Bl > 0.8) & (R < 0.55) & (G < 0.75)
    green = (G > 0.75) & (R < 0.75) & (Bl < 0.75)
    block = (R > 0.35) & (R < 0.72) & (np.abs(R - G) < 0.12) & (Bl - R > 0.02) & (Bl - R < 0.30) & (~green)
    return agent, block

def pose_from_image(img, S=512):
    """img HxHx3 (float 0..1 ou uint8). -> (bx, by, bangle, ax, ay) en unités env [0,S], ou None."""
    H = img.shape[0]; agent, block = _masks(img)
    pa = _moment_pose(agent); pb = _moment_pose(block)
    if pa is None or pb is None: return None
    k = S / H
    return np.array([pb[0] * k, pb[1] * k, pb[2], pa[0] * k, pa[1] * k], np.float32)

# ---------- SONDE DE SURFACE : balayer les points de contact du T (idée utilisateur) ----------
def _smooth(m, it=3):
    m = m.astype(np.float32)
    for _ in range(it):
        s = m.copy(); s[1:] += m[:-1]; s[:-1] += m[1:]; s[:, 1:] += m[:, :-1]; s[:, :-1] += m[:, 1:]
        m = s / 5.0
    return m

def surface_points(img, spacing_px=6.0, S=512.0):
    """Depuis la PHOTO : contour du T (silhouette rendue) + normale SORTANTE par point, espacés
    ~spacing_px @S. Aucun apprentissage, aucune géométrie codée en dur — on lit les pixels."""
    _, mask = _masks(img); H = img.shape[0]; k = S / H
    er = mask.copy()
    er[1:] &= mask[:-1]; er[:-1] &= mask[1:]; er[:, 1:] &= mask[:, :-1]; er[:, :-1] &= mask[:, 1:]
    bnd = mask & ~er                                                      # pixels de bord
    sm = _smooth(mask); gy, gx = np.gradient(sm)                          # grad pointe vers l'INTÉRIEUR
    ys, xs = np.nonzero(bnd)
    if len(xs) < 5: return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    order = np.argsort(np.arctan2(ys - ys.mean(), xs - xs.mean()))        # tour du contour, espacement régulier
    pts, nls, last = [], [], None
    for i in order:
        x, y = xs[i] * k, ys[i] * k
        nx, ny = -gx[ys[i], xs[i]], -gy[ys[i], xs[i]]                     # normale sortante = -grad
        nn = math.hypot(nx, ny)
        if nn < 1e-6: continue
        nx, ny = nx / nn, ny / nn
        if last is not None and math.hypot(x - last[0], y - last[1]) < spacing_px: continue
        pts.append((x, y)); nls.append((nx, ny)); last = (x, y)
    return np.array(pts, np.float32), np.array(nls, np.float32)

def collect_probe(n_poses, spacing, gap, push, dev, seed=7):
    """Pour chaque pose de départ : garer l'agent, PHOTO -> points de contact ; puis, point par point,
    reset (T à la pose, agent devant le point) -> UNE poussée normale -> photo -> transition -> revenir.
    Retourne un buffer de transitions au même format que l'offline (depuis l'arrêt : blk_prev=blk)."""
    env = make_env(); rng = np.random.default_rng(seed)
    B0, AG0, AC, B1, AG1 = [], [], [], [], []
    for pi in range(n_poses):
        bx = rng.uniform(140, 372); by = rng.uniform(140, 372); ba = rng.uniform(0, 2 * np.pi)
        obs, info = reset_faithful(env, np.array([20.0, 20.0, bx, by, ba], np.float32))  # agent garé -> silhouette propre
        pts, nls = surface_points(obs["pixels"])
        for (px, py), (nx, ny) in zip(pts, nls):
            ax = px + nx * gap; ay = py + ny * gap                       # agent juste devant la surface
            if not (8 < ax < 504 and 8 < ay < 504): continue
            obs, info = reset_faithful(env, np.array([ax, ay, bx, by, ba], np.float32))
            b0 = np.array(info["block_pose"], np.float32); a0 = np.array(obs["agent_pos"], np.float32)
            tgt = np.clip([px - nx * push, py - ny * push], 0, 512).astype(np.float32)  # pousser DANS la surface
            obs, _, term, trunc, info = env.step(tgt)
            B0.append(b0); AG0.append(a0); AC.append(np.clip((tgt - a0) / 512.0, -1, 1))
            B1.append(np.array(info["block_pose"], np.float32)); AG1.append(np.array(obs["agent_pos"], np.float32))
        if pi % 20 == 0: print(f"  sonde pose {pi:3d}/{n_poses}  ({len(B0)} transitions)", flush=True)
    env.close()
    B0, AG0, B1, AG1 = map(lambda z: np.array(z, np.float32), (B0, AG0, B1, AG1)); AC = np.array(AC, np.float32)
    blk0, ag0 = to_state(B0, AG0); blk1, ag1 = to_state(B1, AG1)
    Tt = lambda z: torch.tensor(z, device=dev)
    # format buffer : [blk_prev, blk, ag, act, blk_next, ag_next] ; depuis l'arrêt -> blk_prev = blk
    return [Tt(blk0), Tt(blk0), Tt(ag0), Tt(AC), Tt(blk1), Tt(ag1)]

# ---------- BRIQUE 2 : dynamique apprise sur les poses ----------
class PoseDyn(nn.Module):
    """LEVIER 3 : géométrie RELATIVE (agent vu depuis le repère du bloc — le contact devient le même
    quel que soit l'angle) + LEVIER 3b : VITESSE (2 poses en entrée). Le bloc a de l'élan : sans sa
    vitesse, impossible de distinguer 'à l'arrêt' de 'glisse encore'. On passe donc la pose courante
    ET la précédente ; leur différence (en repère bloc) = la vitesse."""
    def __init__(s, hid=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(14, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                              nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 6))
    def forward(s, blk, blkp, ag, act):                                  # blk,blkp:(B,4) ag,act:(B,2)
        bx, by, si, co = blk[:, 0:1], blk[:, 1:2], blk[:, 2:3], blk[:, 3:4]
        def to_local(vx, vy): return torch.cat([vx * co + vy * si, -vx * si + vy * co], -1)
        rel = to_local(ag[:, 0:1] - bx, ag[:, 1:2] - by)                  # agent dans le repère du bloc
        actl = to_local(act[:, 0:1], act[:, 1:2])                         # action dans le repère du bloc
        vloc = to_local(blk[:, 0:1] - blkp[:, 0:1], blk[:, 1:2] - blkp[:, 1:2])  # vitesse pos (repère bloc)
        dang = torch.cat([si - blkp[:, 2:3], co - blkp[:, 3:4]], -1)       # vitesse angulaire (Δsin,Δcos)
        d = s.net(torch.cat([blk, act, rel, actl, vloc, dang], -1))
        nb = blk + d[:, :4]
        nb = torch.cat([nb[:, :2], nb[:, 2:4] / (nb[:, 2:4].norm(dim=-1, keepdim=True) + 1e-8)], -1)
        return nb, ag + d[:, 4:6]

def to_state(bp, ag, S=512.0):
    """poses env -> vecteurs normalisés. bp:(...,3) ag:(...,2)."""
    blk = np.concatenate([bp[..., :2] / S, np.sin(bp[..., 2:3]), np.cos(bp[..., 2:3])], -1)
    return blk.astype(np.float32), (ag / S).astype(np.float32)

# ---------- BRIQUE 3 : planification dans l'espace des poses ----------
def pose_cost(blk, gblk, w_ang=1.0):
    pos = (blk[:, :2] - gblk[:2]).norm(dim=-1)
    ang = (blk[:, 2:4] - gblk[2:4]).norm(dim=-1)                          # distance (sinθ,cosθ)
    return pos + w_ang * ang

def cem_pose(dyn, blk0, blkp0, ag0, gblk, Hp=7, pop=256, iters=3, elite=16, amax=0.5, dev="cpu",
             mu0=None, w_ang=1.0, w_app=0.1):
    with torch.no_grad():
        blk = blk0.expand(pop, -1).clone(); blkp = blkp0.expand(pop, -1).clone(); ag = ag0.expand(pop, -1).clone()
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.25, device=dev)
        for _ in range(iters):
            eps = torch.randn(pop, Hp, 2, device=dev)
            eps[: pop // 2] = torch.randn(pop // 2, 1, 2, device=dev)      # bruit corrélé (iCEM)
            A = (mu + sg * eps).clamp(-amax, amax)
            b, bp, a, cost = blk, blkp, ag, 0.
            for h in range(Hp):
                nb, a = dyn(b, bp, a, A[:, h]); bp = b; b = nb
                wt = (h + 1) / Hp                                          # coût DENSE : chaque pas compte, les tardifs plus
                cost = cost + wt * pose_cost(b, gblk, w_ang)               # -> pente même quand le but est LOIN (anti-dithering)
                cost = cost + wt * w_app * (a - b[:, :2]).norm(dim=-1)     # rester au contact pour pousser
            el = A[cost.argsort()[:elite]]
            mu = el.mean(0); sg = (el.std(0) + 1e-4).clamp_min(0.05)
    return mu.clamp(-amax, amax)

def seq_to_transitions(bp, ag, ac, dev):
    """Une trajectoire (poses+actions) -> transitions (blk_prev, blk, ag, act) -> (blk_next, ag_next)."""
    blk, ags = to_state(bp, ag)
    blk = torch.tensor(blk, device=dev); ags = torch.tensor(ags, device=dev); ac = torch.tensor(ac, device=dev)
    L = blk.shape[0]
    if L < 3: return None
    ii = torch.arange(1, L - 1)
    return blk[ii - 1], blk[ii], ags[ii], ac[ii], blk[ii + 1], ags[ii + 1]

def train_dyn(dyn, opt, buf, steps, bs, log_every=1000, eval_fn=None):
    Bp, Bc, Ag, Ac, Bn, An = buf; M = Bp.shape[0]
    for st in range(steps):
        idx = torch.randint(0, M, (bs,), device=Bp.device)
        nb, na = dyn(Bc[idx], Bp[idx], Ag[idx], Ac[idx])
        loss = ((nb - Bn[idx]) ** 2).sum(-1).mean() + ((na - An[idx]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if eval_fn and st % log_every == 0: eval_fn(st)

def collect_onpolicy(dyn, n_ep, dev, offset, w_ang, w_app, plan_h, max_steps=60):
    """Le PLANIFICATEUR joue avec le world model actuel ; on enregistre ses VRAIES trajectoires
    (y compris ses gestes fins d'endgame) -> transitions pour réentraîner (DAgger)."""
    parts = []
    for e in range(n_ep):
        r = run_episode(dyn, 20000 + e, dev, "mpc", max_steps=max_steps, plan_h=plan_h,
                        offset=offset, w_ang=w_ang, w_app=w_app, record_traj=True)
        tr = seq_to_transitions(r[1], r[2], r[3], dev)
        if tr is not None: parts.append(tr)
    if not parts: return None
    return [torch.cat([p[i] for p in parts], 0) for i in range(6)]

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

def run_episode(dyn, seed, dev, policy, max_steps=100, plan_h=4, scratch=None, offset=np.zeros(2),
                record=False, w_ang=1.0, w_app=0.1, record_traj=False):
    env = make_env(); rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    gblk = torch.tensor(np.concatenate([gp[:2] / 512, [np.sin(gp[2]), np.cos(gp[2])]]), dtype=torch.float32, device=dev)
    ag = np.array(obs["agent_pos"], np.float32); best = float(info.get("coverage", 0.0)); mu = None; blk_prev = None
    frames, covs = [], []; tbp, tag, tac = [], [], []                    # trajectoire VRAIE (pour on-policy)
    for _ in range(max_steps):
        if record: frames.append(obs["pixels"].copy()); covs.append(float(info.get("coverage", 0.0)))
        if policy == "mpc":
            pose = pose_from_image(obs["pixels"])                        # perception géométrique
            if pose is None: bp = np.array([*info["block_pose"]], np.float32)  # secours si masque perdu
            else:
                th = pose[2]; ox, oy = offset                            # centroïde -> référence BP (calibré)
                bx = pose[0] - (ox * np.cos(th) - oy * np.sin(th))
                by = pose[1] - (ox * np.sin(th) + oy * np.cos(th))
                bp = np.array([bx, by, th], np.float32)
            blk, agn = to_state(bp[None], ag[None]); blk = torch.tensor(blk, device=dev); agn = torch.tensor(agn, device=dev)
            if blk_prev is None: blk_prev = blk                          # 1er pas : vitesse nulle
            plan = cem_pose(dyn, blk, blk_prev, agn, gblk, Hp=plan_h, dev=dev, mu0=mu, w_ang=w_ang, w_app=w_app)
            blk_prev = blk
            delta = plan[0].cpu().numpy(); mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])
            act = np.clip(ag + delta * 512.0, 0, 512).astype(np.float32)
        elif policy == "oracle":                                         # ORACLE correct : replanifie chaque pas
            st = np.array([ag[0], ag[1], *np.array(info["block_pose"], np.float32)], np.float32)
            pl = oracle_plan(scratch, st, gp, rng, Hp=5, pop=40, iters=2); d0 = pl[0]
            act = np.clip(ag + d0 * 256.0, 0, 512).astype(np.float32)
        else:
            th = rng.uniform(0, 2 * np.pi); act = np.clip(ag + rng.uniform(0.2, 1.0) * 256 *
                                                          np.array([np.cos(th), np.sin(th)]), 0, 512).astype(np.float32)
        if record_traj:                                                  # VRAIE pose/action de CE pas
            tbp.append(np.array(info["block_pose"], np.float32)); tag.append(ag.copy())
            tac.append(np.clip((act - ag) / 512.0, -1, 1).astype(np.float32))
        obs, _, term, trunc, info = env.step(act); ag = np.array(obs["agent_pos"], np.float32)
        best = max(best, float(info.get("coverage", 0.0)))
        if info.get("is_success", False) or term or trunc: break
    if record: frames.append(obs["pixels"].copy()); covs.append(float(info.get("coverage", 0.0)))
    if record_traj: tbp.append(np.array(info["block_pose"], np.float32)); tag.append(ag.copy())  # pose finale
    env.close()
    if record_traj: return best, np.array(tbp), np.array(tag), np.array(tac)
    return (best, frames, covs) if record else best

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=""); p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--bs", type=int, default=128); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tasks", type=int, default=10); p.add_argument("--plan_h", type=int, default=7)
    p.add_argument("--policies", type=str, default="mpc,oracle,random"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--viz", type=int, default=-1)                         # >=0 : filmer l'épisode MPC de cette tâche
    p.add_argument("--w_ang", type=float, default=1.0)                    # poids de l'angle dans le coût (endgame rotation)
    p.add_argument("--w_app", type=float, default=0.1)                    # poids de l'approche
    p.add_argument("--dagger_rounds", type=int, default=0)               # rondes on-policy (0 = off)
    p.add_argument("--dagger_eps", type=int, default=20)                 # épisodes collectés par ronde
    p.add_argument("--dagger_steps", type=int, default=3000)             # réentraînement par ronde
    p.add_argument("--probe_poses", type=int, default=0)                 # sonde de surface : nb de poses de départ (0=off)
    p.add_argument("--probe_spacing", type=float, default=6.0)           # espacement des points de contact (px @512)
    p.add_argument("--probe_gap", type=float, default=18.0)              # distance agent<->surface au départ (px)
    p.add_argument("--probe_push", type=float, default=20.0)             # profondeur de la poussée fine (px)
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
    # buffer OFFLINE (jeu aléatoire) : transitions (blk_prev, blk, ag, act) -> (blk_next, ag_next)
    off_buf = [blk[:ntr, 0:T - 2].reshape(-1, 4), blk[:ntr, 1:T - 1].reshape(-1, 4), ag[:ntr, 1:T - 1].reshape(-1, 2),
               act[:ntr, 1:T - 1].reshape(-1, 2), blk[:ntr, 2:T].reshape(-1, 4), ag[:ntr, 2:T].reshape(-1, 2)]
    bv = blk[ntr:]
    # SONDE DE SURFACE (idée utilisateur) : balayage systématique de la réponse locale, mélangé à l'offline.
    pb_val = None
    if a.probe_poses > 0:
        print(f"--- sonde de surface : {a.probe_poses} poses, espacement {a.probe_spacing:.0f}px, "
              f"poussée {a.probe_push:.0f}px ---", flush=True)
        pb = collect_probe(a.probe_poses, a.probe_spacing, a.probe_gap, a.probe_push, dev)
        M = pb[0].shape[0]; nv = min(2000, M // 10)
        pb_val = [t[M - nv:] for t in pb]; pb_tr = [t[:M - nv] for t in pb]
        rep = max(1, off_buf[0].shape[0] // max(1, 2 * pb_tr[0].shape[0]))  # peser ~autant que l'offline
        off_buf = [torch.cat([off_buf[i], pb_tr[i].repeat(rep, *[1] * (pb_tr[i].dim() - 1))], 0) for i in range(6)]
        print(f"  {M} transitions de sonde ({nv} tenues à l'écart) ; ×{rep} mélangées à l'offline", flush=True)
    def dyn_eval(st):
        with torch.no_grad():
            nb, _ = dyn(bv[:, 1], bv[:, 0], ag[ntr:, 1], act[ntr:, 1])
            ep = (nb[:, :2] - bv[:, 2, :2]).norm(dim=-1).mean().item() * 512
            ec = (bv[:, 1, :2] - bv[:, 2, :2]).norm(dim=-1).mean().item() * 512
            msg = f"  step {st:5d}  bloc t+1 alÉatoire : imaginé {ep:.1f}px vs copie {ec:.1f}px"
            if pb_val is not None:                                        # réponse locale FINE (sonde)
                nbp, _ = dyn(pb_val[1], pb_val[0], pb_val[2], pb_val[3])
                ip = (nbp[:, :2] - pb_val[4][:, :2]).norm(dim=-1).mean().item() * 512
                cp = (pb_val[1][:, :2] - pb_val[4][:, :2]).norm(dim=-1).mean().item() * 512
                msg += f"  |  sonde fine : imaginé {ip:.1f}px vs copie {cp:.1f}px"
        print(msg, flush=True)
    print("--- dynamique ---", flush=True)
    train_dyn(dyn, opt, off_buf, a.steps, a.bs, eval_fn=dyn_eval)
    # DAgger : le planificateur joue -> on réentraîne sur SES gestes (endgame fin, hors distribution du jeu)
    buf = off_buf
    for r in range(a.dagger_rounds):
        print(f"--- DAgger ronde {r+1}/{a.dagger_rounds} : collecte on-policy ({a.dagger_eps} épisodes) ---", flush=True)
        op = collect_onpolicy(dyn, a.dagger_eps, dev, offset, a.w_ang, a.w_app, a.plan_h)
        if op is None: print("  (aucune trajectoire)"); continue
        # on répète les transitions on-policy pour qu'elles PÈSENT autant que l'offline (sinon noyées)
        rep = max(1, off_buf[0].shape[0] // (2 * op[0].shape[0]))
        buf = [torch.cat([buf[i], op[i].repeat(rep, *[1] * (op[i].dim() - 1))], 0) for i in range(6)]
        print(f"  {op[0].shape[0]} transitions on-policy (×{rep}) ; réentraînement", flush=True)
        train_dyn(dyn, opt, buf, a.dagger_steps, a.bs, eval_fn=dyn_eval)
    # VISUALISATION d'un épisode (diagnostic : où ça coince ?)
    if a.viz >= 0:
        best, frames, covs = run_episode(dyn, 3000 + a.viz, dev, "mpc", plan_h=a.plan_h, offset=offset, record=True, w_ang=a.w_ang, w_app=a.w_app)
        print(f"épisode viz (tâche {a.viz}) : coverage max {best:.2f} en {len(covs)} pas", flush=True)
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
            fig = plt.figure(figsize=(16, 5))
            for j, i in enumerate(idx):
                axf = fig.add_subplot(2, len(idx), j + 1); axf.imshow(frames[i])
                axf.set_title(f"pas {i}  cov {covs[i]:.2f}", fontsize=9); axf.axis("off")
            axc = fig.add_subplot(2, 1, 2)
            axc.plot(covs, lw=2); axc.axhline(0.95, ls="--", c="g", label="succès (0.95)")
            axc.axhline(best, ls=":", c="r", label=f"max atteint ({best:.2f})")
            axc.set_xlabel("pas"); axc.set_ylabel("coverage"); axc.legend(); axc.set_ylim(0, 1)
            axc.set_title("coverage dans le temps — monte puis plafonne ? oscille ? jamais proche ?")
            out = "/content/pusht_posewm_viz.png" if os.path.isdir("/content") else "pusht_posewm_viz.png"
            plt.tight_layout(); plt.savefig(out); print(f"figure -> {out}", flush=True)
        except Exception as e:
            print("plot skip:", str(e)[:70], flush=True)
        if a.tasks == 0:
            return
    # BRIQUE 3 : planification
    if a.tasks > 0:
        scratch = make_env(); scratch.reset(seed=0)
        pols = a.policies.split(","); sc = {q: [] for q in pols}
        for k in range(a.tasks):
            line = []
            for q in pols:
                cov = run_episode(dyn, 3000 + k, dev, q, plan_h=a.plan_h, scratch=scratch, offset=offset, w_ang=a.w_ang, w_app=a.w_app)
                sc[q].append(cov); line.append(f"{q} {cov:.2f}")
            print(f"  tâche {k:2d}  " + "  |  ".join(line), flush=True)
        print("\nTOUTES tâches : " + "  |  ".join(
            f"{q} {np.mean(sc[q]):.2f} (succès {sum(v>0.95 for v in sc[q])}/{a.tasks})" for q in pols), flush=True)
        # ÉVAL JUSTE : seulement les tâches FAISABLES (celles où l'oracle atteint >0.5)
        if "oracle" in sc and "mpc" in sc:
            feas = [k for k in range(a.tasks) if sc["oracle"][k] > 0.5]
            if feas:
                mo = np.mean([sc["mpc"][k] for k in feas]); oo = np.mean([sc["oracle"][k] for k in feas])
                rr = np.mean([sc["random"][k] for k in feas]) if "random" in sc else 0
                print(f"FAISABLES (oracle>0.5, {len(feas)}/{a.tasks} tâches) : "
                      f"MPC {mo:.2f}  |  oracle {oo:.2f}  |  aléatoire {rr:.2f}"
                      f"   [MPC/oracle = {100*mo/oo:.0f}% du plafond]", flush=True)
            else:
                print("FAISABLES : aucune tâche où l'oracle dépasse 0.5 (oracle trop faible ou tâches dures)", flush=True)
        scratch.close()

if __name__ == "__main__":
    main()
