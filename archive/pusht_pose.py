"""
PUSH-T — SONDE DE POSE : lire (x, y, angle) du T DIRECTEMENT depuis les slots, sans repeindre.

Virage (discussion utilisateur) : le crayon (décodeur) ne sait faire que des taches floues, et un
indice radial ne sauverait que les cercles (un T n'est pas radial — c'est pour ça qu'il ressemblait
à une boule). Mais la VISION marche : les slots localisent au demi-pixel (position 0.021). L'info
d'orientation est presque sûrement dedans aussi (sinon le crayon ne dessinerait pas le T à la bonne
pose, floue ou pas). Donc on ARRÊTE de dessiner : on lit les nombres directement dans le slot.

Émergent ? Oui : le world model reste 100% non-supervisé (reconstruction). La sonde n'est qu'un
LECTEUR de ce qui a émergé — exactement les sondes linéaires/attentives de DINO / V-JEPA 2.1.

Sonde = pooling attentif sur les K-1 slots-objets -> MLP -> (x, y, sinθ, cosθ). Entraînée sur les
états de PERCEPTION *et* les états POST-g (imaginés) — car le planner lira la pose d'états imaginés.

Question tranchée, un chiffre : ERREUR D'ANGLE du T.
  - petite -> on planifie sur (x,y,angle) lus directement, le canvas flou n'est plus dans la boucle.
  - grande -> les slots ne capturent pas l'orientation = limite de fond à connaître.

  python pusht_pose.py --steps 8000 --bs 32
"""
import argparse, math, os
import numpy as np, torch, torch.nn as nn
from slots_pusht import WM, default_path, load_data, to_batch

class PoseProbe(nn.Module):
    """Pooling attentif (une requête apprise lit les K-1 slots) -> (x, y, sinθ, cosθ)."""
    def __init__(s, D, hid=128):
        super().__init__()
        s.q = nn.Parameter(torch.randn(1, 1, D) * 0.5)
        s.att = nn.MultiheadAttention(D, 4, batch_first=True)
        s.mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, hid), nn.ReLU(),
                              nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 4))
    def forward(s, S):                                                    # S:(B,K-1,D)
        q = s.q.expand(S.size(0), -1, -1)
        pooled, _ = s.att(q, S, S)                                        # (B,1,D)
        out = s.mlp(pooled[:, 0])
        xy = torch.sigmoid(out[:, :2])                                    # position dans [0,1]
        ang = out[:, 2:] / (out[:, 2:].norm(dim=-1, keepdim=True) + 1e-8)  # (sinθ,cosθ) normalisé
        return xy, ang

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=""); p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--out", type=str, default=""); p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = a.data or default_path("pusht_data.npz"); ckpt = a.ckpt or default_path("pusht_wm.pt")
    out = a.out or default_path("pusht_pose.pt")
    ck = torch.load(ckpt, map_location=dev); sa = ck["args"]
    m = WM(sa["hin"], sa["K"], sa["D"], res=sa["hin"] // 2, slot_dim=sa["slot_dim"],
           dec_w=sa["dec_w"], iters=sa["iters"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    for q in m.parameters(): q.requires_grad_(False)
    H = sa["hin"]
    X, dA, P, CT = load_data(data)
    d = np.load(data); BP = d["BP"]                                       # (n,T,3) : x,y,angle du bloc
    n, T = X.shape[0], X.shape[1]; HOR = T - 2
    ne = min(300, n // 10); ntr = n - ne
    probe = PoseProbe(sa["D"]).to(dev)
    opt = torch.optim.Adam(probe.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    print(f"device={dev}  sonde de pose (x,y,angle) sur modèle GELÉ ({ckpt})  train {ntr} / éval {ne}", flush=True)

    def slot_states(bi):
        """États-slots (perception t1 + rollout post-g) et poses vraies alignées, pour un batch."""
        x = to_batch(X, bi, dev, H); acts = torch.tensor(dA[bi]).to(dev)
        with torch.no_grad():
            _, _, S0 = m.peel(m.feats(x[:, 0])); _, _, S1 = m.peel(m.feats(x[:, 1]), init=S0)
            states, prev, cur = [S1], S0, S1
            for h in range(HOR):
                nxt = m.step_a(cur, prev, acts[:, 1 + h]); states.append(nxt); prev, cur = cur, nxt
        # poses vraies aux frames 1..T-1
        pos = torch.tensor(BP[bi, 1:, :2] / 512.0, dtype=torch.float32, device=dev)     # (B,T-1,2)
        ang = torch.tensor(BP[bi, 1:, 2], dtype=torch.float32, device=dev)              # (B,T-1)
        sc = torch.stack([torch.sin(ang), torch.cos(ang)], -1)                          # (B,T-1,2)
        return states, pos, sc

    ev = np.arange(ntr, n)
    for st in range(a.steps):
        bi = np.random.randint(0, ntr, a.bs)
        states, pos, sc = slot_states(bi)
        loss = 0.
        for k, S in enumerate(states):                                   # chaque état -> sa pose
            xy, ang = probe(S.detach())
            loss = loss + ((xy - pos[:, k]) ** 2).sum(-1).mean() + 0.5 * ((ang - sc[:, k]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step(); sched.step()
        if st % 500 == 0:
            with torch.no_grad():
                states, pos, sc = slot_states(ev)
                pe, ae = [], []
                for k, S in enumerate(states):
                    xy, ang = probe(S)
                    pe.append((xy - pos[:, k]).norm(dim=-1))
                    cosd = (ang * sc[:, k]).sum(-1).clamp(-1, 1)          # cos de l'écart d'angle
                    ae.append(torch.acos(cosd) * 180 / math.pi)
                pe = torch.cat(pe).mean().item(); ae = torch.cat(ae)
                # baseline angle : prédire l'angle MOYEN (niveau 'aucune info')
                mang = torch.atan2(sc[..., 0].mean(), sc[..., 1].mean())
                base = torch.acos((torch.stack([torch.sin(mang), torch.cos(mang)]) * sc).sum(-1).clamp(-1, 1)).mean() * 180 / math.pi
            print(f"  step {st:5d}  pos {pe:.3f} (px≈{pe*512:.0f})  |  ANGLE méd {ae.median():.0f}° moy {ae.mean():.0f}°"
                  f"  (baseline 'aucune info' {base:.0f}°)", flush=True)
    torch.save({"model": probe.state_dict(), "wm_args": sa}, out)
    print(f"sonde sauvegardée -> {out}", flush=True)
    # figure : angle vrai vs prédit sur l'éval (nuage) — voir si ça suit la diagonale
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with torch.no_grad():
            states, pos, sc = slot_states(ev)
            xy, ang = probe(states[0])                                    # perception t1
            at = torch.atan2(sc[:, 0, 0], sc[:, 0, 1]) * 180 / math.pi
            ap = torch.atan2(ang[:, 0], ang[:, 1]) * 180 / math.pi
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].scatter(at.cpu(), ap.cpu(), s=8, alpha=0.5); ax[0].plot([-180, 180], [-180, 180], "r--")
        ax[0].set_xlabel("angle VRAI (°)"); ax[0].set_ylabel("angle LU par la sonde (°)")
        ax[0].set_title("angle : sur la diagonale = lisible")
        ax[1].scatter(pos[:, 0, 0].cpu(), xy[:, 0].cpu(), s=8, alpha=0.5, label="x")
        ax[1].scatter(pos[:, 0, 1].cpu(), xy[:, 1].cpu(), s=8, alpha=0.5, label="y")
        ax[1].plot([0, 1], [0, 1], "r--"); ax[1].legend(); ax[1].set_title("position : vrai vs lu")
        ax[1].set_xlabel("vrai"); ax[1].set_ylabel("lu")
        fo = "/content/pusht_pose.png" if os.path.isdir("/content") else "pusht_pose.png"
        plt.tight_layout(); plt.savefig(fo); print(f"figure -> {fo}", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:70], flush=True)

if __name__ == "__main__":
    main()
