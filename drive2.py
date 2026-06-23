"""
Conduite PHASE 4 : route longue + sequence d'evenements (stop-and-go).

Evenements le long de la route :
  - STOP : arret obligatoire (toujours), attendre, puis repartir.
  - PASSAGE PIETON : arret CONDITIONNEL (seulement si un pieton traverse), puis repartir.

Phase 4a (ce fichier) : l'environnement + un conducteur ORACLE scripte (qui connait la
scene) pour GENERER et VOIR le monde. La regle LeCun (world model + actor appris depuis
l'observation) viendra ensuite remplacer l'oracle.

  python drive2.py --demo
"""
import argparse, json, math
import numpy as np

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=135, help="instants (dt=0.2 -> 27 s)")
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--cruise", type=float, default=8.0, help="vitesse de croisiere m/s")
    p.add_argument("--a_acc", type=float, default=2.5); p.add_argument("--a_brake", type=float, default=4.0)
    p.add_argument("--ped_speed", type=float, default=1.4)
    p.add_argument("--p_cross", type=float, default=0.6, help="proba qu'un pieton traverse a un passage")
    p.add_argument("--wait", type=int, default=6, help="instants d'attente a un stop")
    p.add_argument("--demo", action="store_true"); p.add_argument("--k", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

LAYOUT = [(25.0, "stop"), (55.0, "cw"), (85.0, "stop"), (115.0, "cw")]   # positions (m) et types

def gen_route(rng, a):
    T, dt = a.T, a.dt
    ev = []
    for x, ty in LAYOUT:
        e = {"x": x, "type": ty}
        if ty == "cw":
            e["crosses"] = bool(rng.random() < a.p_cross)
            n_before = sum(1 for xx, tt in LAYOUT if tt == "stop" and xx < x)
            arr = int((x / a.cruise) / dt + n_before * (a.wait + 9))   # arrivee nominale (cruise + stops)
            e["t_step"] = int(np.clip(arr + rng.integers(-14, 5), 0, T - 8)) if e["crosses"] else -1
        ev.append(e)
    def ped_y(e, t):                                          # position y du pieton d'un passage
        if e["type"] != "cw" or not e["crosses"] or t < e["t_step"]:
            return 3.0
        return 3.0 - a.ped_speed * dt * (t - e["t_step"])
    # ----- conducteur ORACLE (connait la scene) : stop-and-go -----
    x, v = 0.0, 0.0; cleared = set(); waited = {}
    xs, vs, stopped_flag = [], [], []
    for t in range(T):
        xs.append(round(x, 2)); vs.append(round(v, 2))
        desired = a.cruise
        for i, e in enumerate(ev):
            d = e["x"] - x
            if d < -2.0: continue                            # deja passe
            stop_at = None
            if e["type"] == "stop" and i not in cleared:
                stop_at = e["x"] - 1.0                        # s'arreter avant le stop
            elif e["type"] == "cw":
                py = ped_y(e, t)
                if e.get("crosses") and -1.0 < py < 3.0:     # pieton engage, pas encore degage
                    stop_at = e["x"] - 1.5
            if stop_at is not None:
                dd = stop_at - x
                vmax = math.sqrt(max(0.0, 2 * a.a_brake * max(0.0, dd)))   # vitesse qui permet l'arret
                desired = min(desired, vmax)
        dv = max(-a.a_brake * dt, min(a.a_acc * dt, desired - v))
        v = max(0.0, v + dv); x += v * dt
        # gestion de l'attente aux stops
        for i, e in enumerate(ev):
            if e["type"] == "stop" and i not in cleared and abs(x - (e["x"] - 1.0)) < 2.0 and v < 0.3:
                waited[i] = waited.get(i, 0) + 1
                if waited[i] >= a.wait: cleared.add(i)
        stopped_flag.append(v < 0.3)
    peds = {i: [round(ped_y(e, t), 2) for t in range(T)] for i, e in enumerate(ev) if e["type"] == "cw"}
    return {"ego_x": xs, "ego_v": vs, "events": [{"x": e["x"], "type": e["type"]} for e in ev],
            "peds": peds, "T": T, "cruise": a.cruise}

def main():
    a = get_args()
    if a.demo:
        rng = np.random.default_rng(a.seed)
        r = gen_route(rng, a)
        stops = sum(1 for e in r["events"] if e["type"] == "stop")
        cws = sum(1 for e in r["events"] if e["type"] == "cw")
        crossed = sum(1 for i, e in enumerate(r["events"]) if e["type"] == "cw" and min(r["peds"][i]) < 0)
        print(f"# route {LAYOUT[-1][0]:.0f} m, T={a.T} | {stops} stops, {cws} passages ({crossed} avec pieton) "
              f"| v final {r['ego_v'][-1]} | x final {r['ego_x'][-1]}")
        print("DRIVE2_DUMP " + json.dumps(r))
    else:
        print("utilise --demo")

if __name__ == "__main__":
    main()
