"""
PUSH-T — PEINTRE DE LECTURE : décodeur riche entraîné APRÈS COUP sur le checkpoint GELÉ.

Diagnostic de fond (discussion avec l'utilisateur) : le flou est constant depuis le début du
projet — c'est la signature de l'économie anti-triche (décodeur faible + goulot) qui a rendu
l'ÉMERGENCE possible. Tous les consommateurs précédents étaient tolérants au flou (centroïdes,
comparaisons relatives) ; l'optimiseur adversarial du planner est le premier qui se nourrit des
ambiguïtés absolues (marge honnête ~0.079 vs illusions ~0.05).

Sortie de la tension : DÉCOUPLER le peintre d'entraînement (faible, gardé tel quel — c'est lui
qui force l'émergence) du peintre de lecture (riche, entraîné représentation GELÉE — son gradient
ne remonte jamais dans le modèle, il ne peut pas corrompre l'économie). Il lit le slot COMPLET
(64d, le goulot de 12 était une contrainte d'entraînement, pas de lecture) et il est calibré sur
les états POST-g (leçon DIAG PIPELINE : c'est là que le planner décode).

Verdict rendu par l'entraînement lui-même : si l'erreur de lecture couleur (centroïde gris vs
pose vraie du bloc) chute nettement sous celle du peintre faible, l'information ÉTAIT dans les
slots (problème pictural, résolu) ; sinon la limite est représentationnelle (le goulot a jeté
la netteté) — dans les deux cas on sait.

  python pusht_readout.py --steps 10000 --bs 32
"""
import argparse, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from slots import PosEmbed
from slots_pusht import WM, default_path, load_data, to_batch, wmix, motion_w

GRIS = torch.tensor([0.55, 0.58, 0.62])

class ReadOut(nn.Module):
    def __init__(s, H, D=64, w=256):
        super().__init__(); s.H = H
        s.pe = PosEmbed(D, H)                                              # position ADDITIVE (test contrôlé :
        s.dec = nn.Sequential(nn.Conv2d(D, w, 1), nn.ReLU(),               # additif large > FiLM, la largeur
                              nn.Conv2d(w, w, 1), nn.ReLU(),               # était le goulot pictural)
                              nn.Conv2d(w, 4, 1))
        s.bg = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.95)                 # fond blanc appris
    def imagine(s, S):
        """Même signature/compositing séquentiel que WM.imagine, mais lit le slot COMPLET (64d)
        avec un peintre large. S:(B,K-1,D) -> masks (B,K,1,H,H), rgbs (B,K,3,H,H)."""
        B, Km1, D = S.shape; H = s.H
        x = S.reshape(B * Km1, D, 1, 1).expand(-1, -1, H, H).permute(0, 2, 3, 1)
        x = s.pe(x).permute(0, 3, 1, 2)
        out = s.dec(x)
        rgb = out[:, :3].reshape(B, Km1, 3, H, H); alog = out[:, 3:4].reshape(B, Km1, 1, H, H)
        scope = torch.ones(B, 1, H, H, device=S.device)
        masks, rgbs = [], []
        for j in range(Km1):
            mj = torch.sigmoid(alog[:, j])
            masks.append(scope * mj); rgbs.append(rgb[:, j]); scope = scope * (1 - mj)
        masks.append(scope); rgbs.append(s.bg.expand(B, 3, H, H))
        return torch.stack(masks, 1), torch.stack(rgbs, 1)

def gray_centroid_err(canvas, bp_true, tau=0.15):
    """Erreur de LECTURE couleur : centroïde des pixels gris du canvas vs pose vraie du bloc.
    C'est exactement ce que le planner consomme."""
    BLANC = torch.tensor([0.97, 0.97, 0.97]); BLEU = torch.tensor([0.35, 0.50, 0.90])
    VERT = torch.tensor([0.70, 0.90, 0.70])
    refs = torch.stack([BLANC, GRIS, BLEU, VERT]).to(canvas.device)
    d2 = ((canvas.unsqueeze(1) - refs.reshape(1, 4, 3, 1, 1)) ** 2).sum(2)
    w = torch.softmax(-d2 / (2 * tau * tau), dim=1)
    wg = w[:, 1] * (1 - w[:, 0])
    H = canvas.shape[-1]
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) / H
    gx = torch.tensor(xx).to(canvas.device); gy = torch.tensor(yy).to(canvas.device)
    sm = wg.sum((-1, -2)) + 1e-8
    cx = (wg * gx).sum((-1, -2)) / sm; cy = (wg * gy).sum((-1, -2)) / sm
    c = torch.stack([cx, cy], -1).cpu().numpy()
    return float(np.linalg.norm(c - bp_true, axis=-1).mean())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=""); p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--out", type=str, default=""); p.add_argument("--w", type=int, default=256)
    p.add_argument("--steps", type=int, default=10000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--sig", type=float, default=0.03)
    p.add_argument("--wmotion", type=float, default=30.0); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = a.data or default_path("pusht_data.npz"); ckpt = a.ckpt or default_path("pusht_wm.pt")
    out = a.out or default_path("pusht_ro.pt")
    ck = torch.load(ckpt, map_location=dev); sa = ck["args"]
    m = WM(sa["hin"], sa["K"], sa["D"], res=sa["hin"] // 2, slot_dim=sa["slot_dim"],
           dec_w=sa["dec_w"], iters=sa["iters"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    for q in m.parameters(): q.requires_grad_(False)                       # GELÉ — gradient coupé
    H = sa["hin"]
    X, dA, P, CT = load_data(data)
    n, T = X.shape[0], X.shape[1]; HOR = T - 2
    ne = min(150, n // 10); ntr = n - ne
    ro = ReadOut(H, sa["D"], a.w).to(dev)
    opt = torch.optim.Adam(ro.parameters(), a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, st / warm) *
                                              0.5 * (1 + math.cos(math.pi * max(0, st - warm) / max(1, a.steps - warm))))
    print(f"device={dev}  peintre de lecture w={a.w} sig={a.sig} sur checkpoint GELÉ ({ckpt})", flush=True)
    xe = to_batch(X, np.arange(ntr, n), dev, H); ae = torch.tensor(dA[ntr:]).to(dev); Pe = P[ntr:]
    for st in range(a.steps):
        bi = np.random.randint(0, ntr, a.bs)
        x = to_batch(X, bi, dev, H); acts = torch.tensor(dA[bi]).to(dev)
        with torch.no_grad():                                              # états gelés : S1 + rollout post-g
            _, _, S0 = m.peel(m.feats(x[:, 0]))
            _, _, S1 = m.peel(m.feats(x[:, 1]), init=S0)
            states, prev, cur = [S1], S0, S1
            for h in range(HOR):
                nxt = m.step_a(cur, prev, acts[:, 1 + h]); states.append(nxt); prev, cur = cur, nxt
        loss = 0.
        for k, S in enumerate(states):                                     # cibles : frames 1..T-1
            mm, rr = ro.imagine(S.detach())
            wq = motion_w(x[:, 1 + k], x[:, k], a.wmotion)
            loss = loss + wmix(x[:, 1 + k], rr, mm[:, :, 0], wq, a.sig)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ro.parameters(), 1.0)
        opt.step(); sched.step()
        if st % 500 == 0:
            with torch.no_grad():
                _, _, S0e = m.peel(m.feats(xe[:, 0])); _, _, S1e = m.peel(m.feats(xe[:, 1]), init=S0e)
                prev, cur = S0e, S1e
                for h in range(2): nxt = m.step_a(cur, prev, ae[:, 1 + h]); prev, cur = cur, nxt
                mmr, rrr = ro.imagine(cur); can_r = (rrr * mmr).sum(1)
                mmw, rrw = m.imagine(cur); can_w = (rrw * mmw).sum(1)
                eR = gray_centroid_err(can_r, Pe[:, 3, 1]); eW = gray_centroid_err(can_w, Pe[:, 3, 1])
            print(f"  step {st:5d}  lecture-T post-g(+2) : peintre RICHE {eR:.3f}  vs faible {eW:.3f}"
                  f"  (pose vraie ; plus bas = mieux)", flush=True)
    torch.save({"model": ro.state_dict(), "args": vars(a), "wm_args": sa}, out)
    print(f"peintre de lecture sauvegardé -> {out}", flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ids = np.arange(ntr, ntr + 3)
        xv = to_batch(X, ids, dev, H); av = torch.tensor(dA[ids]).to(dev)
        with torch.no_grad():
            _, _, S0v = m.peel(m.feats(xv[:, 0])); _, _, S1v = m.peel(m.feats(xv[:, 1]), init=S0v)
            prev, cur = S0v, S1v
            for h in range(2): nxt = m.step_a(cur, prev, av[:, 1 + h]); prev, cur = cur, nxt
            mmr, rrr = ro.imagine(cur); can_r = (rrr * mmr).sum(1)
            mmw, rrw = m.imagine(cur); can_w = (rrw * mmw).sum(1)
        fig, ax = plt.subplots(3, 3, figsize=(9, 9))
        for i in range(3):
            ax[i, 0].imshow(X[ids[i], 3]); ax[i, 0].set_title("vraie t3" if i == 0 else "")
            ax[i, 1].imshow(can_w[i].permute(1, 2, 0).cpu().clip(0, 1)); ax[i, 1].set_title("peintre faible" if i == 0 else "")
            ax[i, 2].imshow(can_r[i].permute(1, 2, 0).cpu().clip(0, 1)); ax[i, 2].set_title("peintre RICHE" if i == 0 else "")
            for j in range(3): ax[i, j].axis("off")
        fo = "/content/pusht_readout.png" if os.path.isdir("/content") else "pusht_readout.png"
        plt.tight_layout(); plt.savefig(fo); print(f"figure -> {fo}", flush=True)
    except Exception as e:
        print("plot skip:", str(e)[:60], flush=True)

if __name__ == "__main__":
    main()
