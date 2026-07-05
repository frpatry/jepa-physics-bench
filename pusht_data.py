"""
PUSH-T — BLOC 1 : collecte de données de JEU depuis gym-pusht (LeRobot).

Objectif final ([[cl-pusht-recon]]) : la recette slots complète sur le vrai Push-T, face aux
chiffres publiés de DINO-WM (SR 0.90, but par image, ≤25 pas). Ici : les données.

Politique de jeu (aucune démonstration de la tâche) : cibles de position aléatoires persistantes,
biaisées vers le bloc 1 fois sur 2 (comme le toy-Push — leçon : sans biais, le contact est trop
rare et le signal d'interaction est noyé). L'env attend des actions ABSOLUES (position cible de
l'agent dans [0,512]²) ; on stocke l'absolu + la position de l'agent, le delta normalisé
(a − agent)/256 sera calculé à l'entraînement (l'action locale est ce que g doit consommer).

Format de sortie (npz compressé) :
  X  (n,T,96,96,3) uint8   frames
  A  (n,T-1,2)     float32 actions absolues [0,512]
  AG (n,T,2)       float32 position agent (px, py) [0,512]
  BP (n,T,3)       float32 pose du bloc (x, y, angle) — pour métriques honnêtes et oracle
  GP (3,)          float32 pose but (constante de l'env)
  CT (n,T)         bool    contact agent-bloc à ce pas
  CV (n,T)         float32 coverage (recouvrement bloc/cible officiel de l'env)

  pip install gym-pusht 'pymunk<7'
  python pusht_data.py --n 3000 --T 6
"""
import argparse, os
import numpy as np

def default_out():
    if os.path.isdir("/content/drive/MyDrive"): return "/content/drive/MyDrive/pusht_data.npz"
    if os.path.isdir("/content"): return "/content/pusht_data.npz"
    return "pusht_data.npz"

def play_policy(rng, agent, block, hold_left, target):
    """Cible aléatoire persistante, biaisée vers le bloc 1/2 (données de jeu, zéro démonstration)."""
    if hold_left <= 0:
        if rng.uniform() < 0.5:
            direction = block[:2] - agent
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            ang = np.arctan2(direction[1], direction[0]) + rng.uniform(-0.6, 0.6)
        else:
            ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(60, 200)
        target = np.clip(agent + dist * np.array([np.cos(ang), np.sin(ang)]), 10, 502)
        hold_left = int(rng.integers(2, 5))
    return target, hold_left - 1

def collect(n, T, seed=0, warmup_max=25):
    import gymnasium as gym, gym_pusht  # noqa: F401
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T, 96, 96, 3), np.uint8); A = np.zeros((n, T - 1, 2), np.float32)
    AG = np.zeros((n, T, 2), np.float32); BP = np.zeros((n, T, 3), np.float32)
    CT = np.zeros((n, T), bool); CV = np.zeros((n, T), np.float32); GP = None
    i = 0
    while i < n:
        obs, info = env.reset(seed=int(rng.integers(1 << 30)))
        if GP is None: GP = np.array(info["goal_pose"], np.float32)
        agent = np.array(obs["agent_pos"], np.float32)
        block = np.array(info["block_pose"], np.float32)
        target, hold = agent.copy(), 0
        for _ in range(int(rng.integers(0, warmup_max))):                  # diversifier l'état de départ
            target, hold = play_policy(rng, agent, block, hold, target)
            obs, _, term, trunc, info = env.step(target.astype(np.float32))
            agent = np.array(obs["agent_pos"], np.float32); block = np.array(info["block_pose"], np.float32)
            if term or trunc: break
        ok = True
        for t in range(T):
            X[i, t] = obs["pixels"]; AG[i, t] = agent; BP[i, t] = block
            CT[i, t] = info.get("n_contacts", 0) > 0; CV[i, t] = float(info.get("coverage", 0.0))
            if t < T - 1:
                target, hold = play_policy(rng, agent, block, hold, target)
                A[i, t] = target
                obs, _, term, trunc, info = env.step(target.astype(np.float32))
                agent = np.array(obs["agent_pos"], np.float32); block = np.array(info["block_pose"], np.float32)
                if term or trunc: ok = False; break                        # épisode fini avant la fin
        if ok: i += 1
        if i and i % 200 == 0: print(f"  {i}/{n} séquences", flush=True)
    env.close()
    return X, A, AG, BP, np.asarray(GP), CT, CV

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000); p.add_argument("--T", type=int, default=6)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", type=str, default="")
    a = p.parse_args()
    out = a.out or default_out()
    print(f"collecte gym-pusht : {a.n} séquences de {a.T} frames (politique de JEU, aucune démo)", flush=True)
    X, A, AG, BP, GP, CT, CV = collect(a.n, a.T, a.seed)
    hit = CT.any(1)
    print(f"contact agent-bloc : {100 * hit.mean():.0f}% des séquences  |  coverage moyen {CV.mean():.3f}"
          f"  |  bloc déplacé (>2px) : {100 * (np.linalg.norm(BP[:, -1, :2] - BP[:, 0, :2], axis=1) > 2).mean():.0f}%",
          flush=True)
    np.savez_compressed(out, X=X, A=A, AG=AG, BP=BP, GP=GP, CT=CT, CV=CV)
    print(f"données -> {out}  ({os.path.getsize(out) / 1e6:.0f} Mo)", flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ids = list(np.where(hit)[0])[:2] + list(np.where(~hit)[0])[:1]
        fig, ax = plt.subplots(len(ids), a.T, figsize=(2 * a.T, 2.2 * len(ids)))
        for r_, i in enumerate(ids):
            for t in range(a.T):
                ax[r_, t].imshow(X[i, t]); ax[r_, t].axis("off")
                if r_ == 0: ax[r_, t].set_title(f"t{t}")
        fig.suptitle("séquences de jeu (2 avec contact, 1 sans)")
        fout = "/content/pusht_data.png" if os.path.isdir("/content") else "pusht_data.png"
        plt.tight_layout(); plt.savefig(fout); print(f"figure -> {fout}", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
