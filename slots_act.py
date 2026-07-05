"""
SYSTEM-2 SUR LES SLOTS — toy-Push : agir via le futur imaginé (mini-protocole V-JEPA 2.1 / DINO-WM).

La boucle complète : perception object-centrée (slots.py) + dynamique relationnelle (slots_dyn.py)
+ ACTION. Un agent (disque blanc, contrôlé en vitesse) doit POUSSER un objet vers une cible.
Le but est spécifié par une IMAGE (comme V-JEPA 2.1 sur Franka et DINO-WM sur Push-T), la
planification par CEM dans l'espace des slots : on imagine les futurs de chaque séquence d'actions
candidate et on choisit celle dont le futur imaginé EXPLIQUE le mieux l'image but (NLL de mélange).

Physique du push : agent contrôlé en vitesse (masse infinie), objet avec friction (décélère seul),
contact = l'objet est chassé au contact exact + reçoit la composante normale de la vitesse relative.
Contact riche et permanent — plus dur que les chocs élastiques ponctuels de slots_dyn.

Mesures honnêtes :
  - entraînement : erreur position imaginé vs COPIE (+ sous-ensemble AVEC CONTACT)
  - planification : taux de succès MPC (objet amené à <0.7r de la cible) vs politique ALÉATOIRE
    sur les mêmes tâches. Le modèle n'a jamais vu de démonstration : données = actions aléatoires.

  python slots_act.py --n 5000 --H 48 --steps 20000 --bs 64 --plan_tasks 20
"""
import argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from slots import COLS, mixture_nll, match_error
from slots_dyn import DynModel

WHITE = np.array([1., 1., 1.], np.float32)                                 # l'agent (effecteur, comme Push-T)

def render(pa, po, col, H, r):
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    img = np.zeros((H, H, 3), np.float32)
    img[(xx - po[0]) ** 2 + (yy - po[1]) ** 2 < r * r] = col
    img[(xx - pa[0]) ** 2 + (yy - pa[1]) ** 2 < r * r] = WHITE             # agent dessiné dessus
    return img

def sim_step(pa, po, vo, a, r, fric=0.7):
    """Un pas de physique. Agent contrôlé en vitesse ; objet avec friction ; push au contact."""
    pa = np.clip(pa + a, r, 1 - r)
    vo = vo * fric
    po = po + vo
    dvec = po - pa; dist = float(np.linalg.norm(dvec))
    contact = dist < 2 * r
    if contact:
        nv = dvec / dist if dist > 1e-6 else np.array([1., 0.], np.float32)
        po = pa + nv * 2 * r                                               # chassé au contact exact
        s_ = float((a - vo) @ nv)
        if s_ > 0: vo = vo + s_ * nv                                       # reçoit la poussée normale
    for d in range(2):                                                     # murs INÉLASTIQUES (bloquent,
        if po[d] < r: po[d] = r; vo[d] = 0.                                # comme Push-T — pas de rebond
        if po[d] > 1 - r: po[d] = 1 - r; vo[d] = 0.                        # qui s'emballe contre l'agent)
    return pa, po, vo, contact

def gen_push(n, H, r, T=6, seed=0, amax=0.12):
    """Séquences d'interaction ALÉATOIRE (aucune démonstration) : actions en 2 segments persistants."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T, H, H, 3), np.float32); P = np.zeros((n, T, 2, 2), np.float32)
    A = np.zeros((n, T - 1, 2), np.float32); hit = np.zeros(n, bool)
    for i in range(n):
        col = COLS[rng.integers(0, len(COLS))]
        pa = rng.uniform(r, 1 - r, 2).astype(np.float32)
        for _ in range(50):
            po = rng.uniform(r, 1 - r, 2).astype(np.float32)
            if np.linalg.norm(po - pa) > 2.2 * r: break
        vo = np.zeros(2, np.float32)
        cut = rng.integers(1, T - 1)                                       # 2 segments d'action persistants
        for seg, (t0, t1) in enumerate([(0, cut), (cut, T - 1)]):
            sp = rng.uniform(0.2, 1.0) * amax
            if rng.uniform() < 0.5:                                        # exploration biaisée vers l'objet
                th = math.atan2(po[1] - pa[1], po[0] - pa[0]) + rng.uniform(-0.6, 0.6)  # (données de JEU,
            else:                                                          #  aucune démonstration de la tâche)
                th = rng.uniform(0, 2 * np.pi)
            A[i, t0:t1] = sp * np.array([np.cos(th), np.sin(th)], np.float32)
        for t in range(T):
            X[i, t] = render(pa, po, col, H, r); P[i, t, 0] = pa; P[i, t, 1] = po
            if t < T - 1:
                pa, po, vo, c = sim_step(pa, po, vo, A[i, t], r)
                if c: hit[i] = True
    return X, P, A, hit

class ActModel(DynModel):
    """slots_dyn + conditionnement sur l'action : chaque slot reçoit l'action (et doit apprendre
    que seul le slot-agent y répond, les autres via l'interaction — rien n'est étiqueté)."""
    def __init__(s, H, K, D=64, **kw):
        super().__init__(H, K, D, **kw)
        s.g_act = nn.Linear(2, D)
    def step_a(s, S, Sprev, a):                                            # a:(B,2)
        h = s.g_in(torch.cat([S, S - Sprev], -1)) + s.g_act(a).unsqueeze(1)
        att, _ = s.g_att(h, h, h)
        return S + s.g_out(torch.cat([h, att], -1))
    def rollout(s, x0, x1, A):                                             # A:(B,HOR,2), actions t1->t2, t2->t3...
        m0, r0, S0 = s.peel(s.feats(x0))
        m1, r1, S1 = s.peel(s.feats(x1), init=S0)
        outs, prev, cur = [], S0, S1
        for h in range(A.size(1)):
            nxt = s.step_a(cur, prev, A[:, h])
            outs.append(s.imagine(nxt))
            prev, cur = cur, nxt
        return (m0, r0), (m1, r1), outs

def cem_plan(m, x0, x1, gz, Hp=8, pop=192, iters=5, elite=16, amax=0.12, dev="cpu", mu0=None):
    """CEM dans l'ESPACE DES SLOTS : coût = distance de Chamfer entre les slots imaginés (2 derniers
    pas) et les slots du but. Leçon du run 1 : un coût pixel (NLL de l'image but) est PLAT tant que
    les blobs ne se recouvrent pas -> aucun gradient de progrès ; la distance entre latents de slots
    est LISSE (chaque cm de progrès de l'objet compte) — c'est le protocole DINO-WM/V-JEPA.
    On ne décode RIEN pendant la recherche : seule la dynamique latente est roulée (rapide).
    mu0 = warm-start (plan précédent décalé d'un pas)."""
    with torch.no_grad():
        _, _, S0 = m.peel(m.feats(x0)); _, _, S1 = m.peel(m.feats(x1), init=S0)
        S0 = S0.expand(pop, -1, -1); S1 = S1.expand(pop, -1, -1)
        gzr = gz.expand(pop, -1, -1)                                       # (pop,K-1,slot_dim)
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.06, device=dev)
        for _ in range(iters):
            A = (mu + sg * torch.randn(pop, Hp, 2, device=dev)).clamp(-amax, amax)
            prev, cur, cost = S0, S1, 0.
            for h in range(Hp):
                nxt = m.step_a(cur, prev, A[:, h]); prev, cur = cur, nxt
                if h >= Hp - 2:                                            # les 2 derniers pas comptent
                    d = torch.cdist(m.down(cur), gzr)                      # (pop,K-1,K-1)
                    cost = cost + d.min(-1).values.sum(-1) + d.min(-2).values.sum(-1)
            el = A[cost.argsort()[:elite]]
            mu, sg = el.mean(0), el.std(0) + 1e-4
    return mu.clamp(-amax, amax)

def run_episode(m, seed, H, r, dev, policy="mpc", max_steps=20, plan_h=8, amax=0.12,
                pop=192, iters=5):
    """Tâche : pousser l'objet à la cible (but = image rendue, encodée en SLOTS une fois).
    Succès si dist finale < 0.7r."""
    rng = np.random.default_rng(seed)
    col = COLS[rng.integers(0, len(COLS))]
    pa = rng.uniform(r, 1 - r, 2).astype(np.float32)
    for _ in range(50):
        po = rng.uniform(r, 1 - r, 2).astype(np.float32)
        if np.linalg.norm(po - pa) > 2.2 * r: break
    for _ in range(50):
        tg = rng.uniform(1.5 * r, 1 - 1.5 * r, 2).astype(np.float32)
        if np.linalg.norm(tg - po) > 0.25: break
    u = (tg - po) / (np.linalg.norm(tg - po) + 1e-8)
    goal_img = render(np.clip(tg - 2.05 * r * u, r, 1 - r), tg, col, H, r) # but : objet à la cible, agent derrière
    with torch.no_grad():                                                  # le but devient des SLOTS (une fois)
        gtens = torch.tensor(goal_img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        _, _, gS = m.peel(m.feats(gtens)); gz = m.down(gS)                 # (1,K-1,slot_dim)
    vo = np.zeros(2, np.float32); frames = [render(pa, po, col, H, r)]
    pa, po, vo, _ = sim_step(pa, po, vo, np.zeros(2, np.float32), r)       # 1 pas nul -> 2 frames de contexte
    frames.append(render(pa, po, col, H, r))
    best = float(np.linalg.norm(po - tg)); mu = None
    for _ in range(max_steps):
        if policy == "mpc":
            x0 = torch.tensor(frames[-2].transpose(2, 0, 1)).unsqueeze(0).to(dev)
            x1 = torch.tensor(frames[-1].transpose(2, 0, 1)).unsqueeze(0).to(dev)
            plan = cem_plan(m, x0, x1, gz, Hp=plan_h, pop=pop, iters=iters, amax=amax, dev=dev, mu0=mu)
            a = plan[0].cpu().numpy()
            mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])      # warm-start du prochain replan
        else:
            sp = rng.uniform(0.2, 1.0) * amax; th = rng.uniform(0, 2 * np.pi)
            a = sp * np.array([np.cos(th), np.sin(th)], np.float32)
        pa, po, vo, _ = sim_step(pa, po, vo, a.astype(np.float32), r)
        frames.append(render(pa, po, col, H, r))
        best = min(best, float(np.linalg.norm(po - tg)))
    return best < 0.7 * r, frames, goal_img, float(np.linalg.norm(po - tg))

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000); p.add_argument("--H", type=int, default=48)
    p.add_argument("--r", type=float, default=0.15); p.add_argument("--K", type=int, default=3)
    p.add_argument("--D", type=int, default=64); p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--iters", type=int, default=3); p.add_argument("--slot_dim", type=int, default=16)
    p.add_argument("--dec_w", type=int, default=32); p.add_argument("--sig", type=float, default=0.1)
    p.add_argument("--T", type=int, default=6); p.add_argument("--amax", type=float, default=0.12)
    p.add_argument("--plan_tasks", type=int, default=20); p.add_argument("--plan_h", type=int, default=8)
    p.add_argument("--plan_pop", type=int, default=192); p.add_argument("--plan_iters", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    HOR = a.T - 2
    X, P, A, _ = gen_push(a.n, a.H, a.r, T=a.T, seed=a.seed, amax=a.amax)
    Xt = torch.tensor(X.transpose(0, 1, 4, 2, 3)); At = torch.tensor(A)
    print(f"device={dev}  {a.n} séquences de {a.T} frames (agent blanc + objet, actions ALÉATOIRES)"
          f"  K={a.K}  seed={a.seed}  (voit t0,t1+actions -> IMAGINE t2..t{a.T-1})", flush=True)
    m = ActModel(a.H, a.K, a.D, slot_dim=a.slot_dim, dec_w=a.dec_w, iters=a.iters).to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    Xe, Pe, Ae, He = gen_push(400, a.H, a.r, T=a.T, seed=7, amax=a.amax)
    Xet = torch.tensor(Xe.transpose(0, 1, 4, 2, 3)).to(dev); Aet = torch.tensor(Ae).to(dev)
    Ce = np.full(400, 2, np.int64); hid = np.where(He)[0]
    print(f"éval : 400 séquences dont {len(hid)} avec contact agent-objet", flush=True)
    for st in range(a.steps):
        bi = np.random.randint(0, a.n, a.bs)
        x = Xt[bi].to(dev); acts = At[bi].to(dev)
        (m0, r0), (m1, r1), outs = m.rollout(x[:, 0], x[:, 1], acts[:, 1:])
        loss = mixture_nll(x[:, 0], r0, m0[:, :, 0], a.sig) + mixture_nll(x[:, 1], r1, m1[:, :, 0], a.sig)
        for h, (mh, rh) in enumerate(outs):
            loss = loss + mixture_nll(x[:, 2 + h], rh, mh[:, :, 0], a.sig)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if st % 500 == 0:
            with torch.no_grad():
                (tm0, _), (tm1, _), touts = m.rollout(Xet[:, 0], Xet[:, 1], Aet[:, 1:])
                err0 = match_error(tm0[:, :, 0], Pe[:, 0], a.H, Ce)
                eI = [match_error(touts[h][0][:, :, 0], Pe[:, 2 + h], a.H, Ce) for h in range(HOR)]
                eC = [match_error(tm1[:, :, 0], Pe[:, 2 + h], a.H, Ce) for h in range(HOR)]
                hI = [match_error(touts[h][0][hid][:, :, 0], Pe[hid, 2 + h], a.H, Ce[hid]) for h in range(HOR)]
                hC = [match_error(tm1[hid][:, :, 0], Pe[hid, 2 + h], a.H, Ce[hid]) for h in range(HOR)]
            imag = " ".join(f"t+{h+1} {eI[h]:.3f}/{eC[h]:.3f}" for h in range(HOR))
            cont = " ".join(f"t+{h+1} {hI[h]:.3f}/{hC[h]:.3f}" for h in range(HOR))
            print(f"  step {st:5d}  perception t0 {err0:.3f}  |  imaginé/copie : {imag}"
                  f"  |  AVEC CONTACT : {cont}", flush=True)
    # ===== PLANIFICATION (le test System-2) =====
    if a.plan_tasks > 0:
        m.eval()
        wins_m, wins_r, dists = 0, 0, []
        kw = dict(max_steps=a.max_steps, plan_h=a.plan_h, amax=a.amax, pop=a.plan_pop, iters=a.plan_iters)
        for k in range(a.plan_tasks):
            ok, frames, goal_img, d = run_episode(m, 1000 + k, a.H, a.r, dev, "mpc", **kw)
            wins_m += ok; dists.append(d)
            ok_r, _, _, _ = run_episode(m, 1000 + k, a.H, a.r, dev, "random", **kw)
            wins_r += ok_r
            print(f"  tâche {k:2d}  MPC {'OK ' if ok else 'ÉCHEC'} (dist finale {d:.3f})"
                  f"  |  aléatoire {'OK' if ok_r else 'échec'}", flush=True)
        print(f"\nSUCCÈS PLANIFICATION : MPC {wins_m}/{a.plan_tasks}  vs  aléatoire {wins_r}/{a.plan_tasks}"
              f"  (dist finale moyenne MPC {np.mean(dists):.3f}, succès si <{0.7*a.r:.3f})", flush=True)
        try:
            import matplotlib, os; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            ok, frames, goal_img, d = run_episode(m, 1000, a.H, a.r, dev, "mpc", **kw)
            show = frames[::3][:7]
            fig, ax = plt.subplots(1, len(show) + 1, figsize=(2 * (len(show) + 1), 2.4))
            ax[0].imshow(goal_img.clip(0, 1)); ax[0].set_title("BUT"); ax[0].axis("off")
            for j, fr in enumerate(show):
                ax[j + 1].imshow(fr.clip(0, 1)); ax[j + 1].set_title(f"pas {3*j}"); ax[j + 1].axis("off")
            out = "/content/slots_act.png" if os.path.isdir("/content") else "slots_act.png"
            plt.tight_layout(); plt.savefig(out)
            print(f"figure -> {out}  (épisode MPC : {'succès' if ok else 'échec'}, dist {d:.3f})", flush=True)
        except Exception as e:
            print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
