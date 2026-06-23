"""
Jouet de conduite 2D (top-down) — unites SI realistes.

Voiture-ego roule a ego_speed (m/s) vers un passage pieton (cw_x metres). Un pieton
traverse parfois (proba p_cross) a ped_speed (m/s), a un instant aleatoire = l'incertitude.
Ratios realistes : voiture ~5-6x plus rapide que le pieton ; distance de freinage = v^2/(2a).

Regle LeCun : l'observation (cote modele) sera une IMAGE ego-centree, pas l'etat verite-terrain.

  python drive.py --demo
"""
import argparse, json
import numpy as np

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=40, help="nb d'instants (dt=0.2s -> 8 s)")
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--ego_speed", type=float, default=8.0, help="m/s (~29 km/h)")
    p.add_argument("--ped_speed", type=float, default=1.4, help="m/s (pieton ~5.7x plus lent)")
    p.add_argument("--cw_x", type=float, default=38.0, help="position du passage (m), atteint vers t=24")
    p.add_argument("--road", type=float, default=2.5, help="demi-largeur de la route (m)")
    p.add_argument("--path_half", type=float, default=1.0, help="demi-largeur du couloir de collision (voiture+pieton, m)")
    p.add_argument("--p_cross", type=float, default=0.5)
    p.add_argument("--a_brake", type=float, default=4.0, help="deceleration de freinage (m/s^2)")
    p.add_argument("--demo", action="store_true"); p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def gen_episode(rng, a):
    T = a.T
    ego_x = a.ego_speed * a.dt * np.arange(T)                 # m
    crosses = bool(rng.random() < a.p_cross)
    t_step = int(rng.integers(3, T - 3))
    y0 = a.road + 0.5                                         # trottoir (au-dessus de la route)
    ped_y = np.full(T, y0)
    if crosses:
        for t in range(T):
            if t >= t_step:
                ped_y[t] = y0 - a.ped_speed * a.dt * (t - t_step)   # traverse vers le bas
    # COLLISION REELLE : voiture (voie y=0) et pieton au MEME point -> ego au passage ET pieton DANS LE COULOIR
    gap = 2.0; lane_y = 0.0
    conflict = False; t_conf = -1
    for t in range(T):
        if abs(ego_x[t] - a.cw_x) < gap and abs(ped_y[t] - lane_y) < a.path_half:
            conflict = True; t_conf = t; break
    return {"ego_x": [round(float(v), 2) for v in ego_x], "ped_x": round(float(a.cw_x), 2),
            "ped_y": [round(float(v), 2) for v in ped_y], "crosses": crosses, "t_step": t_step,
            "conflict": conflict, "t_conf": t_conf, "road": a.road, "lane_y": 0.0,
            "dt": a.dt, "ego_speed": a.ego_speed, "a_brake": a.a_brake}

def main():
    a = get_args()
    if a.demo:
        rng = np.random.default_rng(a.seed)
        eps = [gen_episode(rng, a) for _ in range(a.k)]
        bd = a.ego_speed ** 2 / (2 * a.a_brake)
        print(f"# {a.k} ep | T={a.T} dt={a.dt} | ego {a.ego_speed} m/s, pieton {a.ped_speed} m/s "
              f"(ratio {a.ego_speed/a.ped_speed:.1f}x) | dist freinage {bd:.1f} m | "
              f"passent={sum(e['crosses'] for e in eps)} conflits={sum(e['conflict'] for e in eps)}")
        print("DRIVE_DEMO " + json.dumps(eps))
    else:
        print("utilise --demo")

if __name__ == "__main__":
    main()
