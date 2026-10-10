"""
ANIMATIONS des épisodes de planification du bébé (enregistrés par `av_plan0.py --dump_n N`, ligne « DUMP_JSON … »).
Rejoue chaque épisode dans la même physique (Push0) avec les gestes choisis par le bébé, et superpose :
  - la CIBLE (croix verte : point pour la main, objet à toucher, ou cible de l'objet) ;
  - ce que le bébé IMAGINE à la fin de son plan (cercle jaune = sa main imaginée, carré rouge = l'objet imaginé) ;
  - les trajets réels (main en blanc, objet en orange).

  python anim_plan0.py anim_all.txt          # -> anim_<tâche>.gif
"""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from av_plan0 import Push0, VMAX

TITRES = {"main": "Amener SA MAIN sur la croix (pièce vide)", "main_loin": "Amener sa main sur une croix lointaine",
          "toucher": "TOUCHER l'objet", "objet": "POUSSER l'objet sur la croix", "bouger": "Faire BOUGER l'objet",
          "direction": "Pousser l'objet dans une direction"}

def replay(task, ep, steps, plan_c):
    env = Push0(ep, task=task); rng = np.random.default_rng(30_000 + ep)
    for _ in range(plan_c): env.step(np.clip(rng.normal(0, 0.03, 2), -VMAX, VMAX))
    for s in steps: env.step(np.asarray(s["act"], np.float32))
    return env

def main():
    lines = [l for l in open(sys.argv[1]) if l.startswith("DUMP_JSON")]
    for l in lines:
        D = json.loads(l[len("DUMP_JSON "):]); task, eps = D["task"], D["episodes"]; n = len(eps)
        envs = [replay(task, e["ep"], e["steps"], D["plan_c"]) for e in eps]; T_ = envs[0].t + 1
        cols = 3; rows = -(-n // cols); fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows)); axs = np.array(axs).reshape(-1)
        def draw(f):
            for k, ax in enumerate(axs):
                ax.clear(); ax.set_xticks([]); ax.set_yticks([])
                if k >= n: ax.axis("off"); continue
                env, st = envs[k], eps[k]["steps"]
                ax.imshow(env.X[f], extent=(0, 1, 1, 0), interpolation="nearest")
                ax.plot(env.HAND[:f + 1, 0], env.HAND[:f + 1, 1], "-", c="w", lw=1, alpha=0.7)
                if task not in ("main", "main_loin"): ax.plot(env.POS[:f + 1, 0], env.POS[:f + 1, 1], "-", c="orange", lw=1, alpha=0.8)
                gx, gy = (env.P0 if task in ("toucher", "bouger", "direction") else env.g)
                ax.plot(gx, gy, "+", c="lime", ms=14, mew=2.5)
                if task == "direction": ax.arrow(env.P0[0], env.P0[1], 0.15 * env.u[0], 0.15 * env.u[1], color="lime", width=0.008)
                g_ = [s for s in st if s["t"] <= f]
                if g_ and f >= D["plan_c"]:
                    gh = g_[-1]["ghost"]
                    ax.plot(gh[2], gh[3], "o", ms=13, mfc="none", mec="yellow", mew=2)                       # main IMAGINÉE (fin du plan)
                    if task not in ("main", "main_loin"): ax.plot(gh[0], gh[1], "s", ms=12, mfc="none", mec="red", mew=2)   # objet IMAGINÉ
                phase = "il gigote" if f < D["plan_c"] else "il PLANIFIE"
                ax.set_title(f"épisode {eps[k]['ep']} · t={f} · {phase}", fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(1, 0)
            fig.suptitle(f"{TITRES.get(task, task)}  —  jaune : où il IMAGINE sa main"
                         + (" · rouge : où il IMAGINE l'objet" if task not in ("main", "main_loin") else "") + " (à la fin de son plan)", fontsize=9)
        anim = FuncAnimation(fig, draw, frames=T_, interval=400)
        out = f"anim_{task}.gif"; anim.save(out, writer=PillowWriter(fps=2.5)); plt.close(fig); print("->", out)

if __name__ == "__main__":
    main()
