"""
Conduite — SYSTEME 2 : PLANNER MPC LATENT (le World Model imagine les futurs possibles
sous chaque action candidate et choisit la meilleure décision). Fidèle vision LeCun.

Pipeline (tout dans l'espace latent d'un World Model V-JEPA gelé) :
  1. PERCEPTION : V-JEPA (vjepa.py, prédicteur attentionnel + SIGReg, sans EMA) en SSL sur des
     fenêtres de W images ego-centrées -> encodeur GELÉ : fenêtre -> état latent z.
  2. DYNAMIQUE action-conditionnée  g(z, a) -> z'  : modèle de monde CONTROLABLE, appris en
     supervisé sur des rollouts du conducteur PERTURBÉ (oracle + bruit) pour couvrir les actions.
  3. DANGER  c(z) -> P(collision piéton imminente)  : tête apprise sur les latents.
  4. PLANNING (System-2) : à chaque pas, pour chaque action candidate, on DÉROULE g sur un horizon,
     on évalue Σ γ^k c(z^k) (danger imaginé) + coût progrès/confort, et on choisit l'argmin.

On compare en BOUCLE FERMÉE (collisions piéton, vitesse) :
  - naïf      : fonce toujours (jamais de frein)            -> System-0
  - réactif   : freine si c(z) > 0.5 sur la frame COURANTE  -> System-1 (pas de lookahead)
  - MPC       : imagine les futurs sous chaque action (g+c) -> System-2 (anticipe)

  python drive_plan.py --steps 2000 --n_train 600 --n_test 150
"""
import argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import drive2, drive2_model
from drive2_model import render_frame
from vjepa import VJEPA, temporal_mask

# ----------------------------------------------------------------- environnement (oracle + danger)
def oracle_accel(x, v, t, ev, peds, cleared, a, ignore_ped=False):
    """accel oracle (stop-and-go + cède au piéton). ignore_ped=True => navigation seule (croisière+stops,
    SANS freinage piéton) : c'est l'échafaudage commun ; la décision piéton est ce qu'on teste."""
    desired = a.cruise
    for i, e in enumerate(ev):
        d = e["x"] - x
        if d < -2.0: continue
        stop_at = None
        if e["type"] == "stop" and i not in cleared:
            stop_at = e["x"] - 1.0
        elif e["type"] == "cw" and not ignore_ped:
            py = peds[i][t] if i in peds else 3.0
            if -1.0 < py < 3.0: stop_at = e["x"] - 1.5         # piéton engagé
        if stop_at is not None:
            vmax = np.sqrt(max(0.0, 2 * a.a_brake * max(0.0, stop_at - x)))
            desired = min(desired, vmax)
    dv = np.clip(desired - v, -a.a_brake * a.dt, a.a_acc * a.dt)
    return dv / a.dt

def danger_label(x, t, ev, peds):
    """1 si un piéton est engagé sur/près de la route JUSTE devant l'ego (zone de collision imminente)."""
    for i, e in enumerate(ev):
        if e["type"] != "cw": continue
        d = e["x"] - x
        py = peds[i][t] if i in peds else 3.0
        if -1.5 < d < 14.0 and abs(py) < 2.0: return 1
    return 0

def peds_of(route): return {i: route["peds"][i] for i in route["peds"]}

def behavior_rollout(route, a, rng):
    """rollout du conducteur PERTURBÉ (oracle + bruit) pour couvrir l'espace des actions.
    Renvoie frames (T,obs), v (T,), actions (T,), danger (T,)."""
    ev = route["events"]; peds = peds_of(route); T = route["T"]
    x = 0.0; v = 0.0; cleared = set(); waited = {}
    frames, vs, acts, dngr = [], [], [], []
    for t in range(T):
        frames.append(render_frame(x, ev, {i: peds[i][t] for i in peds}))
        vs.append(v); dngr.append(danger_label(x, t, ev, peds))
        a_or = oracle_accel(x, v, t, ev, peds, cleared, a)
        a_cmd = float(np.clip(a_or + a.explore * rng.standard_normal(), -a.a_brake, a.a_acc))
        acts.append(a_cmd)
        v = max(0.0, v + a_cmd * a.dt); x += v * a.dt
        for i, e in enumerate(ev):                             # gestion attente aux stops
            if e["type"] == "stop" and i not in cleared and abs(x - (e["x"] - 1.0)) < 2.0 and v < 0.3:
                waited[i] = waited.get(i, 0) + 1
                if waited[i] >= 6: cleared.add(i)
    return np.stack(frames), np.array(vs, np.float32), np.array(acts, np.float32), np.array(dngr, np.int64)

def windows_of(frames, W):
    """(T,obs) -> (T,W,obs) : fenêtre glissante des W dernières frames (pad début)."""
    T = len(frames); pad = np.repeat(frames[:1], W - 1, 0)
    g = np.concatenate([pad, frames], 0)
    return np.stack([g[t:t + W] for t in range(T)])            # (T,W,obs)

# ----------------------------------------------------------------- dynamique latente + danger
class Dyn(nn.Module):
    """modèle de monde CONTROLABLE : g(z, a) -> z' (latent suivant sous l'action a)."""
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d))
    def forward(s, z, a): return z + s.net(torch.cat([z, a], -1))   # résiduel

class Danger(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 2))
    def forward(s, z): return s.net(z)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=135); p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--cruise", type=float, default=8.0); p.add_argument("--a_acc", type=float, default=2.5)
    p.add_argument("--a_brake", type=float, default=4.0); p.add_argument("--p_cross", type=float, default=0.6)
    p.add_argument("--G", type=int, default=16); p.add_argument("--W", type=int, default=10)
    p.add_argument("--d_model", type=int, default=128); p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--reg_w", type=float, default=1.0); p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--dyn_steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=128); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_train", type=int, default=600); p.add_argument("--n_test", type=int, default=150)
    p.add_argument("--explore", type=float, default=1.0, help="bruit d'action des rollouts (couverture dynamique)")
    p.add_argument("--stride", type=int, default=3, help="pas de la dynamique latente (frames) -> Δz action-sensible")
    p.add_argument("--horizon", type=int, default=4, help="horizon de planification (en pas de stride)")
    p.add_argument("--gamma", type=float, default=0.9); p.add_argument("--w_coll", type=float, default=6.0)
    p.add_argument("--w_prog", type=float, default=0.3); p.add_argument("--w_jerk", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def routes(n, a, seed):
    rng = np.random.default_rng(seed)
    env = argparse.Namespace(T=a.T, dt=a.dt, cruise=a.cruise, a_acc=a.a_acc, a_brake=a.a_brake,
                             ped_speed=1.4, p_cross=a.p_cross, wait=6)
    return [drive2.gen_route(rng, env) for _ in range(n)]

def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    drive2_model.G = a.G; obs = a.G * a.G; W = a.W; d = a.d_model
    tr_routes = routes(a.n_train, a, 1000 + a.seed); te_routes = routes(a.n_test, a, 99999)
    print(f"device={dev}  routes train={a.n_train} test={a.n_test}  G={a.G} W={W} T={a.T}", flush=True)

    # ---- 1) PERCEPTION : V-JEPA SSL sur fenêtres de W frames (masquage temporel = imaginer la suite) ----
    TRobs = torch.tensor(np.stack([drive2_model.route_obs(r) for r in tr_routes]))   # (n,T,obs) trajectoires oracle
    n, T = TRobs.shape[0], TRobs.shape[1]
    torch.manual_seed(a.seed)
    wm = VJEPA(obs, d, W, a.n_layer, a.n_head, a.reg_w, a.pred_layers).to(dev)
    opt = torch.optim.AdamW(wm.parameters(), a.lr)
    for st in range(a.steps):
        ri = torch.randint(0, n, (a.bs,)); si = torch.randint(0, T - W, (a.bs,))
        o = torch.stack([TRobs[ri[b], si[b]:si[b] + W] for b in range(a.bs)]).to(dev)   # (bs,W,obs)
        t0 = np.random.randint(1, W - 1); masks = [temporal_mask(a.bs, W, 1, t0, dev)]
        loss = wm(o, masks); opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0: print(f"  [percep] step {st} loss {loss.item():.3f}", flush=True)
    for prm in wm.parameters(): prm.requires_grad = False
    wm.eval()

    @torch.no_grad()
    def encode(win_np):                                        # (.,W,obs) numpy -> (.,d) latent gelé
        return wm.feat(torch.as_tensor(np.asarray(win_np), dtype=torch.float32, device=dev))

    # ---- 2+3) DONNÉES rollouts perturbés -> (z, a, danger, z') ----
    rng = np.random.default_rng(7); s = a.stride
    Z, A, D, Znext = [], [], [], []
    for r in tr_routes:
        fr, vv, ac, dg = behavior_rollout(r, a, rng)
        wins = windows_of(fr, W)                               # (T,W,obs)
        z = encode(wins).cpu().numpy()                         # (T,d)
        am = np.array([ac[t:t + s].mean() for t in range(len(ac) - s)], np.float32)  # action moyenne sur le stride
        Z.append(z[:-s]); Znext.append(z[s:]); A.append(am); D.append(dg[:-s])       # g(z_t,a)->z_{t+stride}
    Z = torch.tensor(np.concatenate(Z)).to(dev); Znext = torch.tensor(np.concatenate(Znext)).to(dev)
    A = torch.tensor(np.concatenate(A)).to(dev); D = torch.tensor(np.concatenate(D)).to(dev)
    anorm = (A / a.a_acc).unsqueeze(-1)                        # action normalisée
    print(f"\ntransitions dynamique : {len(Z)}  (danger+={int(D.sum())})", flush=True)

    dyn = Dyn(d).to(dev); cda = Danger(d).to(dev)
    od = torch.optim.Adam(list(dyn.parameters()) + list(cda.parameters()), 1e-3)
    pos = (D == 1).nonzero(as_tuple=True)[0]; neg = (D == 0).nonzero(as_tuple=True)[0]   # pour batch ÉQUILIBRÉ
    h = a.bs // 2
    for st in range(a.dyn_steps):
        bi = torch.randint(0, len(Z), (a.bs,), device=dev)
        lz = F.smooth_l1_loss(dyn(Z[bi], anorm[bi]), Znext[bi])      # g(z,a)->z' sur batch uniforme
        bd = torch.cat([pos[torch.randint(0, len(pos), (h,), device=dev)],   # danger : moitié + / moitié -
                        neg[torch.randint(0, len(neg), (h,), device=dev)]]) if len(pos) else bi
        lc = F.cross_entropy(cda(Z[bd]), D[bd])                       # c(z)->danger (calibré)
        od.zero_grad(); (lz + lc).backward(); od.step()
        if st % 400 == 0: print(f"  [dyn] step {st}  pred {lz.item():.3f}  danger {lc.item():.3f}", flush=True)
    dyn.eval(); cda.eval()
    with torch.no_grad():                                            # diagnostic : c(z) sépare-t-il danger / sûr ?
        print(f"  c(z) moyen : danger={cda(Z[pos]).softmax(-1)[:,1].mean():.2f}  sûr={cda(Z[neg]).softmax(-1)[:,1].mean():.2f}", flush=True)

    # ---- 4) DÉCISION TESTÉE = freinage piéton (la navigation croisière+stops est un échafaudage commun) ----
    # Une politique renvoie un NIVEAU DE FREIN piéton >=0 (0 = laisse rouler). On l'applique PAR-DESSUS la
    # navigation : acc = min(base_nav, -niveau). Seul ce niveau distingue System-0/1/2.
    BRAKES = [0.0, 2.0, a.a_brake]; V_MOVE = 0.5; s = a.stride

    @torch.no_grad()
    def decide_naive(z, v): return 0.0                        # ne freine JAMAIS pour les piétons
    @torch.no_grad()
    def decide_reactive(z, v):                                # freine si danger sur la frame COURANTE (System-1)
        return a.a_brake if cda(z).softmax(-1)[0, 1].item() > 0.5 else 0.0
    @torch.no_grad()
    def decide_mpc(z, v):                                     # imagine les futurs sous chaque frein (System-2)
        best, bestc = 0.0, 1e9
        for lvl in BRAKES:
            zc = z.clone(); vv = v; cost = 0.0; disc = 1.0
            acc = -lvl if lvl > 0 else min(a.a_acc, (a.cruise - vv) / a.dt)    # frein, ou suit la croisière
            an = torch.tensor([[acc / a.a_acc]], device=dev)
            for k in range(a.horizon):                        # déroule la dynamique latente par pas de stride
                for _ in range(s): vv = max(0.0, vv + acc * a.dt)
                zc = dyn(zc, an)
                pcoll = cda(zc).softmax(-1)[0, 1].item()
                cost += disc * a.w_coll * pcoll * (1.0 if vv > V_MOVE else 0.0)   # collision = danger imaginé ET on roule
                disc *= a.gamma
                if lvl == 0: acc = min(a.a_acc, (a.cruise - vv) / a.dt); an = torch.tensor([[acc / a.a_acc]], device=dev)
            cost += a.w_prog * max(0.0, a.cruise - vv) + a.w_jerk * lvl          # progrès + confort (frein = inconfort)
            if cost < bestc: bestc = cost; best = lvl
        return best

    def run(route, decide):
        ev = route["events"]; peds = peds_of(route)
        buf = [render_frame(0.0, ev, {i: peds[i][0] for i in peds})] * W
        x = 0.0; v = 0.0; hit = 0; speeds = []; cleared = set(); waited = {}
        for t in range(a.T):
            buf = buf[1:] + [render_frame(x, ev, {i: peds[i][t] for i in peds})]
            z = encode([np.stack(buf)])
            base = oracle_accel(x, v, t, ev, peds, cleared, a, ignore_ped=True)   # navigation commune (croisière+stops)
            lvl = decide(z, v)                                                    # décision piéton testée
            acc = float(np.clip(min(base, -lvl) if lvl > 0 else base, -a.a_brake, a.a_acc))
            v = max(0.0, v + acc * a.dt); x += v * a.dt; speeds.append(v)
            for i, e in enumerate(ev):                                            # stops (échafaudage) + collision
                if e["type"] == "stop" and i not in cleared and abs(x - (e["x"] - 1.0)) < 2.0 and v < 0.3:
                    waited[i] = waited.get(i, 0) + 1
                    if waited[i] >= 6: cleared.add(i)
                if e["type"] == "cw" and abs(x - e["x"]) < 2.0 and abs(peds[i][t]) < 1.0: hit = 1
        return hit, float(np.mean(speeds))

    print(f"\n=== BOUCLE FERMÉE ({a.n_test} routes test) : collisions piéton / vitesse moyenne ===", flush=True)
    print(f"    (navigation croisière+stops commune ; seule la décision de freinage piéton change)", flush=True)
    for name, dec in [("System-0 naïf (jamais de frein)", decide_naive),
                      ("System-1 réactif (danger frame courante)", decide_reactive),
                      ("System-2 MPC (imagine les futurs)", decide_mpc)]:
        res = [run(r, dec) for r in te_routes]
        coll = np.mean([h for h, _ in res]); spd = np.mean([s for _, s in res])
        print(f"  {name:42s} : collisions {coll:.0%}   vitesse {spd:.1f} m/s", flush=True)

if __name__ == "__main__":
    main()
