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

def render(pa, po, col, H, r, ra):
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    img = np.zeros((H, H, 3), np.float32)
    img[(xx - po[0]) ** 2 + (yy - po[1]) ** 2 < r * r] = col
    if pa is not None:
        img[(xx - pa[0]) ** 2 + (yy - pa[1]) ** 2 < ra * ra] = WHITE       # agent PETIT (comme Push-T :
    return img                                                             # l'effecteur passe partout)

def sim_step(pa, po, vo, a, r, ra, fric=0.7):
    """Un pas de physique. Agent (rayon ra < r) contrôlé en vitesse ; objet avec friction ;
    push au contact. Un agent PETIT peut passer entre l'objet et le mur (leçon oracle : avec
    ra=r, un objet plaqué au mur était imperdable -> oracle 6/10 ; avec ra=0.6r -> 10/10)."""
    pa = np.clip(pa + a, ra, 1 - ra)
    vo = vo * fric
    po = po + vo
    dvec = po - pa; dist = float(np.linalg.norm(dvec)); cd = r + ra
    contact = dist < cd
    if contact:
        nv = dvec / dist if dist > 1e-6 else np.array([1., 0.], np.float32)
        po = pa + nv * cd                                                  # chassé au contact exact
        s_ = float((a - vo) @ nv)
        if s_ > 0: vo = vo + s_ * nv                                       # reçoit la poussée normale
    for d in range(2):                                                     # murs INÉLASTIQUES (bloquent,
        if po[d] < r: po[d] = r; vo[d] = 0.                                # comme Push-T)
        if po[d] > 1 - r: po[d] = 1 - r; vo[d] = 0.
    return pa, po, vo, contact

def gen_push(n, H, r, ra, T=6, seed=0, amax=0.12):
    """Séquences d'interaction ALÉATOIRE (aucune démonstration) : actions en 2 segments persistants."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T, H, H, 3), np.float32); P = np.zeros((n, T, 2, 2), np.float32)
    A = np.zeros((n, T - 1, 2), np.float32); hit = np.zeros(n, bool)
    for i in range(n):
        col = COLS[rng.integers(0, len(COLS))]
        pa = rng.uniform(ra, 1 - ra, 2).astype(np.float32)
        for _ in range(50):
            po = rng.uniform(r, 1 - r, 2).astype(np.float32)
            if np.linalg.norm(po - pa) > 1.2 * (r + ra): break
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
            X[i, t] = render(pa, po, col, H, r, ra); P[i, t, 0] = pa; P[i, t, 1] = po
            if t < T - 1:
                pa, po, vo, c = sim_step(pa, po, vo, A[i, t], r, ra)
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
    """CEM dans l'ESPACE DES SLOTS. Leçons accumulées : (run 1) un coût pixel est PLAT sans
    recouvrement des blobs -> coût en latents de slots, lisse ; (run 2) imposer une position but à
    l'AGENT est toxique — le CEM le tire en ligne droite À TRAVERS l'objet (percutage, plaquage au
    mur). v3 : le but ne contient QUE l'objet (gz = 1 slot), coût one-sided = distance du slot
    imaginé le plus proche au slot-objet but. L'agent est LIBRE : aucun terme ne le concerne,
    c'est un moyen, pas une fin. Aucun décodage pendant la recherche (dynamique latente pure)."""
    with torch.no_grad():
        _, _, S0 = m.peel(m.feats(x0)); _, _, S1 = m.peel(m.feats(x1), init=S0)
        S0 = S0.expand(pop, -1, -1); S1 = S1.expand(pop, -1, -1)
        gzr = gz.expand(pop, -1, -1)                                       # (pop,1,slot_dim) : l'objet SEUL
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.06, device=dev)
        for _ in range(iters):
            eps = torch.randn(pop, Hp, 2, device=dev)                      # bruit CORRÉLÉ (iCEM) : moitié
            eps[: pop // 2] = torch.randn(pop // 2, 1, 2, device=dev)      # des candidats = trajectoires
            A = (mu + sg * eps).clamp(-amax, amax)                         # droites (explorent LOIN)
            prev, cur, cost = S0, S1, 0.
            for h in range(Hp):
                nxt = m.step_a(cur, prev, A[:, h]); prev, cur = cur, nxt
                if h >= Hp - 2:                                            # les 2 derniers pas comptent
                    z = m.down(cur)                                        # (pop,K-1,slot_dim)
                    d = torch.cdist(z, gzr)                                # (pop,K-1,1)
                    cost = cost + d.min(1).values.squeeze(-1)              # le slot le plus proche du but
                    cost = cost + 0.25 * (z[:, 0] - z[:, 1]).norm(dim=-1)  # approche : toucher est un
            el = A[cost.argsort()[:elite]]                                 # prérequis pour agir
            mu, sg = el.mean(0), el.std(0) + 1e-4
    return mu.clamp(-amax, amax)

def sim_step_batch(PA, PO, VO, A, r, ra, fric=0.7):
    """sim_step vectorisé (pop,...) — pour le planner ORACLE (diagnostic, état privilégié)."""
    PA = np.clip(PA + A, ra, 1 - ra)
    VO = VO * fric; PO = PO + VO
    D = PO - PA; dist = np.linalg.norm(D, axis=1, keepdims=True); cd = r + ra
    c = (dist < cd).squeeze(-1)
    nv = D / np.maximum(dist, 1e-6)
    PO = np.where(c[:, None], PA + nv * cd, PO)
    s_ = ((A - VO) * nv).sum(1, keepdims=True)
    VO = np.where((c & (s_.squeeze(-1) > 0))[:, None], VO + s_ * nv, VO)
    hit_lo = PO < r; hit_hi = PO > 1 - r
    VO = np.where(hit_lo | hit_hi, 0., VO); PO = np.clip(PO, r, 1 - r)
    return PA, PO, VO

def oracle_plan(pa, po, vo, tg, r, ra, rng, Hp=8, pop=192, iters=5, elite=16, amax=0.12, mu0=None):
    """BORNE SUPÉRIEURE : même CEM mais roulé dans le VRAI simulateur avec le VRAI coût
    (état privilégié). Sépare l'erreur de modèle de l'erreur de recherche : si l'oracle échoue
    aussi, c'est le search/horizon qui limite, pas le world model."""
    mu = np.zeros((Hp, 2), np.float32) if mu0 is None else mu0.copy()
    sg = np.full((Hp, 2), 0.06, np.float32)
    for _ in range(iters):
        eps = rng.standard_normal((pop, Hp, 2)).astype(np.float32)
        eps[: pop // 2] = rng.standard_normal((pop // 2, 1, 2)).astype(np.float32)  # bruit corrélé (iCEM)
        A = np.clip(mu + sg * eps, -amax, amax)
        PA = np.repeat(pa[None], pop, 0); PO = np.repeat(po[None], pop, 0); VO = np.repeat(vo[None], pop, 0)
        cost = np.zeros(pop, np.float32)
        for h in range(Hp):
            PA, PO, VO = sim_step_batch(PA, PO, VO, A[:, h], r, ra)
            if h >= Hp - 2:
                cost += np.linalg.norm(PO - tg[None], axis=1)
                cost += 0.25 * np.linalg.norm(PA - PO, axis=1)             # approche (même shaping que MPC)
        el = A[np.argsort(cost)[:elite]]; mu, sg = el.mean(0), el.std(0) + 1e-4
    return np.clip(mu, -amax, amax)

def run_episode(m, seed, H, r, ra, dev, policy="mpc", max_steps=25, plan_h=8, amax=0.12,
                pop=192, iters=5):
    """Tâche : pousser l'objet à la cible (but = image rendue, encodée en SLOTS une fois).
    Succès si dist finale < 0.7r."""
    rng = np.random.default_rng(seed)
    col = COLS[rng.integers(0, len(COLS))]
    pa = rng.uniform(ra, 1 - ra, 2).astype(np.float32)
    for _ in range(50):
        po = rng.uniform(r, 1 - r, 2).astype(np.float32)
        if np.linalg.norm(po - pa) > 1.2 * (r + ra): break
    for _ in range(50):
        tg = rng.uniform(1.5 * r, 1 - 1.5 * r, 2).astype(np.float32)
        if np.linalg.norm(tg - po) > 0.25: break
    goal_img = render(None, tg, col, H, r, ra)                             # but : l'OBJET SEUL à la cible
    with torch.no_grad():                                                  # (l'agent est libre — un moyen,
        gtens = torch.tensor(goal_img.transpose(2, 0, 1)).unsqueeze(0).to(dev)  # pas une fin)
        gm, _, gS = m.peel(m.feats(gtens))
        j = int(gm[0, :-1, 0].sum((-1, -2)).argmax())                      # la prise qui a capturé l'objet
        gz = m.down(gS)[:, j:j + 1]                                        # (1,1,slot_dim)
    vo = np.zeros(2, np.float32); frames = [render(pa, po, col, H, r, ra)]
    pa, po, vo, _ = sim_step(pa, po, vo, np.zeros(2, np.float32), r, ra)   # 1 pas nul -> 2 frames de contexte
    frames.append(render(pa, po, col, H, r, ra))
    best = float(np.linalg.norm(po - tg)); mu = None; mu_np = None
    for _ in range(max_steps):
        if policy == "mpc":
            x0 = torch.tensor(frames[-2].transpose(2, 0, 1)).unsqueeze(0).to(dev)
            x1 = torch.tensor(frames[-1].transpose(2, 0, 1)).unsqueeze(0).to(dev)
            plan = cem_plan(m, x0, x1, gz, Hp=plan_h, pop=pop, iters=iters, amax=amax, dev=dev, mu0=mu)
            a = plan[0].cpu().numpy()
            mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])      # warm-start du prochain replan
        elif policy == "oracle":                                           # borne sup : vrai sim, vrai coût
            plan = oracle_plan(pa, po, vo, tg, r, ra, rng, Hp=plan_h, pop=pop, iters=iters, amax=amax, mu0=mu_np)
            a = plan[0]; mu_np = np.concatenate([plan[1:], np.zeros((1, 2), np.float32)])
        else:
            sp = rng.uniform(0.2, 1.0) * amax; th = rng.uniform(0, 2 * np.pi)
            a = sp * np.array([np.cos(th), np.sin(th)], np.float32)
        pa, po, vo, _ = sim_step(pa, po, vo, a.astype(np.float32), r, ra)
        frames.append(render(pa, po, col, H, r, ra))
        best = min(best, float(np.linalg.norm(po - tg)))
    return best < 0.7 * r, frames, goal_img, float(np.linalg.norm(po - tg))

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000); p.add_argument("--H", type=int, default=48)
    p.add_argument("--r", type=float, default=0.15); p.add_argument("--ra", type=float, default=0.09)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--D", type=int, default=64); p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--iters", type=int, default=3); p.add_argument("--slot_dim", type=int, default=16)
    p.add_argument("--dec_w", type=int, default=32); p.add_argument("--sig", type=float, default=0.1)
    p.add_argument("--T", type=int, default=6); p.add_argument("--amax", type=float, default=0.12)
    p.add_argument("--plan_tasks", type=int, default=20); p.add_argument("--plan_h", type=int, default=8)
    p.add_argument("--plan_pop", type=int, default=192); p.add_argument("--plan_iters", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    HOR = a.T - 2
    X, P, A, _ = gen_push(a.n, a.H, a.r, a.ra, T=a.T, seed=a.seed, amax=a.amax)
    Xt = torch.tensor(X.transpose(0, 1, 4, 2, 3)); At = torch.tensor(A)
    print(f"device={dev}  {a.n} séquences de {a.T} frames (agent blanc + objet, actions ALÉATOIRES)"
          f"  K={a.K}  seed={a.seed}  (voit t0,t1+actions -> IMAGINE t2..t{a.T-1})", flush=True)
    m = ActModel(a.H, a.K, a.D, slot_dim=a.slot_dim, dec_w=a.dec_w, iters=a.iters).to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    Xe, Pe, Ae, He = gen_push(400, a.H, a.r, a.ra, T=a.T, seed=7, amax=a.amax)
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
        wins_m, wins_o, wins_r, dists = 0, 0, 0, []
        kw = dict(max_steps=a.max_steps, plan_h=a.plan_h, amax=a.amax, pop=a.plan_pop, iters=a.plan_iters)
        for k in range(a.plan_tasks):
            ok, frames, goal_img, d = run_episode(m, 1000 + k, a.H, a.r, a.ra, dev, "mpc", **kw)
            wins_m += ok; dists.append(d)
            ok_o, _, _, d_o = run_episode(m, 1000 + k, a.H, a.r, a.ra, dev, "oracle", **kw)
            wins_o += ok_o
            ok_r, _, _, _ = run_episode(m, 1000 + k, a.H, a.r, a.ra, dev, "random", **kw)
            wins_r += ok_r
            print(f"  tâche {k:2d}  MPC {'OK ' if ok else 'ÉCHEC'} (dist {d:.3f})"
                  f"  |  ORACLE {'OK ' if ok_o else 'échec'} (dist {d_o:.3f})"
                  f"  |  aléatoire {'OK' if ok_r else 'échec'}", flush=True)
        print(f"\nSUCCÈS PLANIFICATION : MPC {wins_m}/{a.plan_tasks}  |  ORACLE (borne sup) {wins_o}/{a.plan_tasks}"
              f"  |  aléatoire {wins_r}/{a.plan_tasks}"
              f"  (dist finale moyenne MPC {np.mean(dists):.3f}, succès si <{0.7*a.r:.3f})", flush=True)
        try:
            import matplotlib, os; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            ok, frames, goal_img, d = run_episode(m, 1000, a.H, a.r, a.ra, dev, "mpc", **kw)
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
