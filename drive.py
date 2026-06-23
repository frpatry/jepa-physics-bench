"""
Jouet de conduite 2D (top-down) pour world model JEPA — PHASE 1 : l'environnement.

Monde : route horizontale (y in [-road, road]), voiture-ego dans la voie qui roule
vers la droite a vitesse constante (phase 1 : pas encore d'actor). Un PIETON a un
passage (x = cw_x) qui TRAVERSE parfois (proba p_cross) a un instant aleatoire =
l'INCERTITUDE irreductible (le "vent" de la conduite).

Regle LeCun : l'observation sera une IMAGE ego-centree (pixels), pas l'etat verite-
terrain. Phase 1 ne fait que generer + dumper les episodes pour les ANIMER et valider
le monde avant d'entrainer quoi que ce soit.

  python drive.py --demo            # dump K episodes (JSON) a animer
"""
import argparse, json
import numpy as np

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=24, help="nb d'instants par episode")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--ego_speed", type=float, default=0.5)
    p.add_argument("--cw_x", type=float, default=6.0, help="position du passage pieton (ego l'atteint vers T/2)")
    p.add_argument("--road", type=float, default=0.9, help="demi-largeur de la route")
    p.add_argument("--p_cross", type=float, default=0.6, help="proba que le pieton traverse")
    p.add_argument("--cross_dur", type=int, default=6, help="duree de traversee (instants)")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--k", type=int, default=6, help="nb d'episodes pour --demo")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def gen_episode(rng, a):
    """Un episode. Ego roule a vitesse constante (phase 1). Pieton traverse peut-etre."""
    T = a.T
    ego_x = a.ego_speed * a.dt * np.arange(T)                 # ego part de 0, roule a droite
    crosses = bool(rng.random() < a.p_cross)
    t_step = int(rng.integers(4, T - a.cross_dur - 1))        # quand le pieton s'engage
    ped_y = np.full(T, 1.3)                                   # trottoir haut (hors route)
    if crosses:
        for t in range(T):
            if t >= t_step:
                frac = min(1.0, (t - t_step) / a.cross_dur)
                ped_y[t] = 1.3 - frac * 2.6                   # de +1.3 (haut) a -1.3 (bas)
    # CONFLIT (verite-terrain, pour l'eval seulement, JAMAIS donne au modele) :
    # ego au passage en meme temps que le pieton sur la route.
    gap = 0.7
    conflict = False; t_conf = -1
    for t in range(T):
        if abs(ego_x[t] - a.cw_x) < gap and abs(ped_y[t]) < a.road:
            conflict = True; t_conf = t; break
    return {"ego_x": [round(float(v), 2) for v in ego_x],
            "ped_x": round(float(a.cw_x), 2),
            "ped_y": [round(float(v), 2) for v in ped_y],
            "crosses": crosses, "t_step": t_step,
            "conflict": conflict, "t_conf": t_conf,
            "road": a.road, "lane_y": -0.45}

def main():
    a = get_args()
    if a.demo:
        rng = np.random.default_rng(a.seed)
        eps = [gen_episode(rng, a) for _ in range(a.k)]
        nconf = sum(e["conflict"] for e in eps); ncross = sum(e["crosses"] for e in eps)
        print(f"# {a.k} episodes | T={a.T} | passent={ncross} | conflits={nconf} "
              f"| ego_speed={a.ego_speed} cw_x={a.cw_x}")
        print("DRIVE_DEMO " + json.dumps(eps))
    else:
        print("rien a faire (utilise --demo). Phase 1 = generer + voir le monde.")

if __name__ == "__main__":
    main()
