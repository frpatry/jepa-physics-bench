"""
PUSH-T — WORLD MODEL JEPA + SIGReg OFFICIEL (LeJEPA, Balestriero & LeCun).

But : refaire le test Push-T (protocole DINO-WM, but par image, ≤N pas, succès = coverage
officiel >0.95) mais avec un VRAI world model JEPA anti-collapse par le SIGReg du package
`lejepa` — PAS de slots, PAS de reconstruction, PAS de décodeur. On prédit des LATENTS.

Choix (validés avec l'utilisateur) :
  ARCHI = JEPA-WM CONJOINT. Un seul modèle entraîné bout-à-bout :
    • Encodeur E (ViT par frame, tokens de patch)         -> z_t = E(frame_t)   (B,npf,d)
    • Dynamique G conditionnée par l'ACTION (attentionnelle) : (z_{t-1}, z_t, a_t) -> ẑ_{t+1}
    • Cible = le latent de la frame future (PAS de stop-grad, PAS d'EMA — thèse LeJEPA).
    • Anti-collapse = SIGReg officiel sur les sorties de E (le SEUL rempart : c'est le test).
  PLANIF = CEM PUREMENT LATENT (DINO-WM) : encoder l'image-but -> z_but ; dérouler G sous CEM ;
    coût = distance latente (smooth-L1) entre ẑ_H et z_but.
      --cost latent          (A, défaut) : distance sur TOUS les tokens (fidèle DINO-WM).
      --cost latent_noagent  (B)        : on jette les tokens dominés par l'agent (bleu) côté
                                          image-but avant de comparer (isole le signal du T).

Réutilise : vjepa.py (sigreg officiel, Encoder ViT) ; pusht_data.py (npz de jeu) ;
pusht_plan.py (env, reset_faithful, oracle_plan borne-sup, scaling d'action, image-but).

  python pusht_jepa.py --steps 30000 --bs 32           # entraîne + sauvegarde (Drive si monté)
  python pusht_jepa.py --load 1 --tasks 10             # recharge + planifie (MPC/oracle/aléa)
  python pusht_jepa.py --resume 1 --steps 20000        # reprend l'entraînement
"""
import argparse, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vjepa import sigreg, Encoder                       # SIGReg officiel + encodeur-contexte ViT
from pusht_plan import make_env, reset_faithful, to64, oracle_plan

# ----------------------------------------------------------- tokenisation torch (frame CHW -> tokens)
def patchify_t(x, P):
    """x:(B,3,H,H) -> (B, npf, P*P*3). Ordre (nP_r,nP_c,P_r,P_c,C) cohérent avec vjepa.patchify."""
    B, C, H, _ = x.shape; nP = H // P
    x = x[:, :, :nP * P, :nP * P].reshape(B, C, nP, P, nP, P)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, nP * nP, P * P * C)
    return x

def frames_to_tokens(Xb, hin, P, dev):
    """Xb:(B,T,96,96,3) uint8 (tensor CPU/GPU) -> tokens (B,T,npf,obs) float sur dev."""
    B, T = Xb.shape[:2]
    x = Xb.to(dev).permute(0, 1, 4, 2, 3).float().div(255.0).reshape(B * T, 3, Xb.shape[2], Xb.shape[3])
    if hin != x.shape[-1]: x = F.interpolate(x, size=hin, mode="area")
    tok = patchify_t(x, P)                              # (B*T, npf, obs)
    return tok.reshape(B, T, tok.shape[1], tok.shape[2])

def img_to_tokens(x_chw, P):
    """x_chw:(B,3,hin,hin) déjà normalisé (sortie to64) -> tokens (B,npf,obs)."""
    return patchify_t(x_chw, P)

def agent_token_mask(tok, P, thr=0.06):
    """tok:(B,npf,obs). Renvoie (npf,) bool = tokens dont le patch est dominé par le BLEU agent
    ([0.35,0.50,0.90]) -> à EXCLURE dans le coût 'latent_noagent'. Moyenne couleur du patch."""
    B, npf, obs = tok.shape
    c = tok.reshape(B, npf, P, P, 3).mean((2, 3))       # (B,npf,3) couleur moyenne par token
    blue = (c[..., 2] - c[..., 0] > 0.10) & (c[..., 2] - c[..., 1] > 0.10) & (c[..., 2] > 0.6)
    return blue.any(0)                                  # (npf,) : token agent dans AU MOINS un échantillon

# ----------------------------------------------------------- dynamique conditionnée par l'action
class Dynamics(nn.Module):
    """Prédicteur attentionnel : contexte = tokens de 2 frames (t-1,t) + un token d'ACTION ;
    requêtes = npf mask-tokens positionnels de la frame t+1 -> latents prédits par attention.
    Rôles distincts (prev / cur / action / query) pour que l'attention sépare les sources."""
    def __init__(s, d, npf, nl, nh):
        super().__init__()
        s.npf = npf
        s.sp = nn.Embedding(npf, d)                     # position spatiale (partagée ctx + requêtes)
        s.role = nn.Embedding(4, d)                     # 0=prev 1=cur 2=action 3=query
        s.act = nn.Linear(2, d)                         # action locale (delta normalisé) -> token
        s.mask_token = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(layer, nl); s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, d)

    def forward(s, z_prev, z_cur, a):
        """z_prev,z_cur:(B,npf,d) ; a:(B,2) -> ẑ_next:(B,npf,d)."""
        B = z_cur.size(0); dev = z_cur.device
        pos = s.sp(torch.arange(s.npf, device=dev)).unsqueeze(0)          # (1,npf,d)
        prev = z_prev + pos + s.role.weight[0]
        cur = z_cur + pos + s.role.weight[1]
        atok = (s.act(a) + s.role.weight[2]).unsqueeze(1)                 # (B,1,d)
        q = (s.mask_token + s.role.weight[3]).unsqueeze(0) + pos          # (1,npf,d)
        q = q.expand(B, -1, -1)
        x = s.tr(torch.cat([prev, cur, atok, q], dim=1))
        return s.head(s.ln(x[:, -s.npf:]))                               # lit les npf requêtes

# ----------------------------------------------------------- world model JEPA complet
class JEPAWM(nn.Module):
    def __init__(s, obs, d, npf, nl=4, nh=4, dyn_layers=4, rw=1.0):
        super().__init__()
        s.enc = Encoder(obs, d, npf, nl, nh)            # ViT-contexte de vjepa (ntok=npf, une frame)
        s.dyn = Dynamics(d, npf, dyn_layers, nh)
        s.npf, s.rw, s.d = npf, rw, d

    def encode(s, tok):
        """tok:(B,npf,obs) -> z:(B,npf,d)."""
        B = tok.size(0)
        idx = torch.arange(s.npf, device=tok.device).expand(B, -1)
        return s.enc(tok, idx)

    def encode_clip(s, tok):
        """tok:(B,T,npf,obs) -> z:(B,T,npf,d)."""
        B, T = tok.shape[:2]
        z = s.encode(tok.reshape(B * T, s.npf, -1))
        return z.reshape(B, T, s.npf, s.d)

    def rollout(s, z_prev, z_cur, A):
        """z_prev,z_cur:(B,npf,d) ; A:(B,H,2) -> latents prédits empilés (B,H,npf,d) (boucle fermée)."""
        outs, prev, cur = [], z_prev, z_cur
        for h in range(A.size(1)):
            nxt = s.dyn(prev, cur, A[:, h]); prev, cur = cur, nxt; outs.append(nxt)
        return torch.stack(outs, 1)

    def forward(s, tok, A, w_tf=1.0, stopgrad=True):
        """tok:(B,T,npf,obs) clip ; A:(B,T-1,2) actions locales. Prédit les frames 2..T-1 depuis
        (frame0,frame1). Perte = boucle-fermée + teacher-forced 1-pas + SIGReg.
        stopgrad=True : la CIBLE de prédiction est détachée (cible stable, recette DINO-WM) — SIGReg
        reste sur z avec gradient (seul anti-collapse). stopgrad=False = fidèle LeJEPA (cible mouvante)."""
        B, T = tok.shape[:2]
        z = s.encode_clip(tok)                          # (B,T,npf,d) — SIGReg dessus (gradient)
        zt = z.detach() if stopgrad else z              # CIBLE de prédiction (détachée si stopgrad)
        # action a[t] = transition t->t+1 ; prédire frame k (k>=2) utilise a[k-1].
        # --- boucle fermée depuis (z0,z1), actions a[1..T-2] -> frames 2..T-1 ---
        cl = z.new_zeros(())
        if T >= 3:
            pred = s.rollout(z[:, 0], z[:, 1], A[:, 1:T - 1])            # (B,T-2,npf,d) ; contexte AVEC grad
            cl = F.smooth_l1_loss(pred, zt[:, 2:T])
        # --- teacher-forced 1-pas : G(z[t-1],z[t],a[t]) vs z[t+1], t=1..T-2 ---
        tf = z.new_zeros(())
        if w_tf > 0 and T >= 3:
            for t in range(1, T - 1):
                p = s.dyn(z[:, t - 1], z[:, t], A[:, t])
                tf = tf + F.smooth_l1_loss(p, zt[:, t + 1])
            tf = tf / (T - 2)
        reg = sigreg(z.reshape(-1, s.d))
        return cl + w_tf * tf + s.rw * reg, cl.detach(), tf.detach(), reg.detach()

# ----------------------------------------------------------- données
def default_path(name):
    if os.path.isdir("/content/drive/MyDrive"): return f"/content/drive/MyDrive/{name}"
    if os.path.isdir("/content"): return f"/content/{name}"
    return name

def load_data(path):
    d = np.load(path)
    X = torch.from_numpy(d["X"])                        # (n,T,96,96,3) uint8
    A = torch.from_numpy(d["A"].astype(np.float32))     # (n,T-1,2) actions absolues [0,512]
    AG = torch.from_numpy(d["AG"].astype(np.float32))   # (n,T,2) position agent
    a = (A - AG[:, :-1]) / 256.0                        # delta local normalisé (ce que G consomme)
    return X, a

# ----------------------------------------------------------- planification CEM latente (DINO-WM)
def cem_latent(m, z0, z1, zg, keep, Hp=4, pop=192, iters=3, elite=16, amax=0.8, dev="cpu", mu0=None):
    """z0,z1:(npf,d) contexte ; zg:(npf,d) but ; keep:(npf,) bool tokens gardés dans le coût.
    Candidats iCEM (moitié constants, moitié 2 segments), coût = smooth-L1 latent au but (dernier pas)."""
    with torch.no_grad():
        Z0 = z0.unsqueeze(0).expand(pop, -1, -1); Z1 = z1.unsqueeze(0).expand(pop, -1, -1)
        ZG = zg.unsqueeze(0)
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.30, device=dev)
        for _ in range(iters):
            e1 = torch.randn(pop, 1, 2, device=dev).expand(-1, Hp, -1).clone()            # constant
            e2 = torch.randn(pop, 2, 2, device=dev).repeat_interleave((Hp + 1) // 2, 1)[:, :Hp]  # 2 segments
            eps = torch.where((torch.arange(pop, device=dev) % 2 == 0).reshape(-1, 1, 1), e1, e2)
            A = (mu + sg * eps).clamp(-amax, amax)
            pred = m.rollout(Z0, Z1, A)                 # (pop,Hp,npf,d)
            zf = pred[:, -1]                            # latent final (pop,npf,d)
            diff = F.smooth_l1_loss(zf, ZG.expand(pop, -1, -1), reduction="none").mean(-1)  # (pop,npf)
            cost = (diff * keep).sum(-1) / keep.sum().clamp_min(1)                            # tokens gardés
            el = A[cost.argsort()[:elite]]
            mu = el.mean(0); sg = (el.std(0) + 1e-4).clamp_min(0.08)                          # plancher iCEM
    return mu.clamp(-amax, amax)

# ----------------------------------------------------------- un épisode d'évaluation
def run_episode(m, hin, P, seed, dev, policy, cost_mode, scratch, max_steps=100, plan_h=4, pop=192, iters=3,
                oracle_h=10, oracle_pop=64, oracle_iters=3, oracle_exec=5):
    env = make_env(); rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    far = np.array([40.0, 40.0], np.float32) if np.linalg.norm(gp[:2] - 40) > 120 else np.array([470.0, 470.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    gtok = img_to_tokens(to64(gob["pixels"], hin, dev), P)
    with torch.no_grad(): zg = m.encode(gtok)[0]        # (npf,d) latent BUT
    keep = torch.ones(m.npf, device=dev)
    if cost_mode == "latent_noagent":                   # B : jeter les tokens agent (bleu) du BUT
        keep = (~agent_token_mask(gtok, P)).float().to(dev)
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    obs, _, _, _, info = env.step(ag.copy()); f1 = obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
    best_cov, success, mu, oq = float(info.get("coverage", 0.0)), False, None, []
    for _ in range(max_steps):
        if policy == "mpc":
            with torch.no_grad():
                z0 = m.encode(img_to_tokens(to64(f0, hin, dev), P))[0]
                z1 = m.encode(img_to_tokens(to64(f1, hin, dev), P))[0]
            plan = cem_latent(m, z0, z1, zg, keep, Hp=plan_h, pop=pop, iters=iters, dev=dev, mu0=mu)
            delta = plan[0].cpu().numpy(); mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])
        elif policy == "oracle":
            if not len(oq):                                 # re-planifie tous les oracle_exec pas (vrai sim = lent)
                state = np.array([ag[0], ag[1], *np.array(info["block_pose"], np.float32)], np.float32)
                oq = list(oracle_plan(scratch, state, gp, rng, Hp=oracle_h, pop=oracle_pop, iters=oracle_iters)[:oracle_exec])
            delta = oq.pop(0)
        else:
            th = rng.uniform(0, 2 * np.pi)
            delta = rng.uniform(0.2, 1.0) * 0.8 * np.array([np.cos(th), np.sin(th)], np.float32)
        act = np.clip(ag + delta * 256.0, 0, 512).astype(np.float32)
        obs, _, term, trunc, info = env.step(act)
        f0, f1 = f1, obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
        best_cov = max(best_cov, float(info.get("coverage", 0.0)))
        if info.get("is_success", False) or term: success = True; break
        if trunc: break
    env.close()
    return success, best_cov

# ----------------------------------------------------------- CLI
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=""); p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--steps", type=int, default=30000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hin", type=int, default=64); p.add_argument("--patch", type=int, default=8)
    p.add_argument("--d", type=int, default=128); p.add_argument("--nl", type=int, default=4)
    p.add_argument("--nh", type=int, default=4); p.add_argument("--dyn_layers", type=int, default=4)
    p.add_argument("--rw", type=float, default=1.0, help="poids SIGReg (1.0 sur images, cf. drive_model)")
    p.add_argument("--w_tf", type=float, default=1.0, help="poids teacher-forced 1-pas (stabilise)")
    p.add_argument("--stopgrad", type=int, default=1, help="1 = cible de prédiction détachée (cible stable, recette DINO-WM) ; 0 = fidèle LeJEPA (cible mouvante, échoue)")
    p.add_argument("--load", type=int, default=0); p.add_argument("--resume", type=int, default=0)
    p.add_argument("--tasks", type=int, default=0, help=">0 : évalue MPC/oracle/aléa sur ce nb de tâches")
    p.add_argument("--cost", type=str, default="latent", choices=["latent", "latent_noagent"])
    p.add_argument("--plan_h", type=int, default=4); p.add_argument("--plan_pop", type=int, default=192)
    p.add_argument("--plan_iters", type=int, default=3); p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--oracle_h", type=int, default=10, help="horizon oracle (vrai sim, non bridé par le modèle)")
    p.add_argument("--oracle_pop", type=int, default=64); p.add_argument("--oracle_iters", type=int, default=3)
    p.add_argument("--oracle_exec", type=int, default=5, help="actions exécutées par plan oracle (re-planif tous les N pas ; vrai sim lent)")
    p.add_argument("--policies", type=str, default="mpc,oracle,random", help="sous-ensemble à évaluer, ex. 'oracle' seul")
    p.add_argument("--diag", type=int, default=0, help="1 = diagnostic du plancher cl (copy vs moyenne vs modèle)")
    p.add_argument("--task_seed", type=int, default=9000, help="graine des tâches de test (tenues à l'écart)")
    return p.parse_args()

def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = a.data or default_path("pusht_data.npz")
    ckpt = a.ckpt or default_path("pusht_jepa.pt")
    P = a.patch; npf = (a.hin // P) ** 2; obs = P * P * 3

    if (a.load or a.resume) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location=dev); sa = blob["args"]
        P = sa["patch"]; npf = (sa["hin"] // P) ** 2; obs = P * P * 3
        m = JEPAWM(obs, sa["d"], npf, sa["nl"], sa["nh"], sa["dyn_layers"], sa["rw"]).to(dev)
        m.load_state_dict(blob["model"]); a.hin = sa["hin"]
        m.rw = a.rw                                     # rw = KNOB d'entraînement (pas d'archi) -> valeur CLI, pas celle gelée du checkpoint
        print(f"checkpoint chargé : {ckpt}  (hin {a.hin}, d {sa['d']}, npf {npf})", flush=True)
    else:
        m = JEPAWM(obs, a.d, npf, a.nl, a.nh, a.dyn_layers, a.rw).to(dev)

    if not a.load:                                      # --load 1 = planifier sans réentraîner
        X, act = load_data(data); n, T = X.shape[0], X.shape[1]
        print(f"données {tuple(X.shape)}  |  {'reprise' if a.resume else 'entraînement'} "
              f"{a.steps} pas, bs {a.bs}, SIGReg rw {m.rw}, stopgrad {bool(a.stopgrad)}, dev {dev}", flush=True)
        opt = torch.optim.Adam(m.parameters(), a.lr)
        rng = np.random.default_rng(0)
        for st in range(a.steps):
            ids = torch.from_numpy(rng.integers(0, n, a.bs))
            tok = frames_to_tokens(X[ids], a.hin, P, dev)               # (bs,T,npf,obs)
            A = act[ids].to(dev)                                        # (bs,T-1,2)
            loss, cl, tf, reg = m(tok, A, w_tf=a.w_tf, stopgrad=bool(a.stopgrad))
            opt.zero_grad(); loss.backward(); opt.step()
            if st % 500 == 0:
                print(f"  step {st:6d}  loss {loss.item():.4f}  (cl {cl.item():.4f}  tf {tf.item():.4f}"
                      f"  sigreg {reg.item():.4f})", flush=True)
        torch.save({"model": m.state_dict(),
                    "args": {"hin": a.hin, "patch": P, "d": a.d, "nl": a.nl, "nh": a.nh,
                             "dyn_layers": a.dyn_layers, "rw": a.rw}}, ckpt)
        print(f"modèle -> {ckpt}", flush=True)

    if a.diag:                                          # DIAGNOSTIC : d'où vient le plancher cl ≈ 0.38 ?
        m.eval(); X, act = load_data(data); n, T = X.shape[0], X.shape[1]
        ids = torch.from_numpy(np.random.default_rng(1).integers(0, n, min(256, n)))
        with torch.no_grad():
            z = m.encode_clip(frames_to_tokens(X[ids], a.hin, P, dev)); A = act[ids].to(dev)
            marg = F.smooth_l1_loss(z, z.mean((0, 1, 2), keepdim=True).expand_as(z)).item()  # prédire la moyenne
            copy = F.smooth_l1_loss(z[:, 1:T - 1], z[:, 2:T]).item()                          # prédire = frame précédente
            pred = m.rollout(z[:, 0], z[:, 1], A[:, 1:T - 1])
            model = F.smooth_l1_loss(pred, z[:, 2:T]).item()                                  # le vrai modèle (boucle fermée)
            zt = z[:, 1:T - 1].reshape(-1, m.d); zt1 = z[:, 2:T].reshape(-1, m.d)
            cos = F.cosine_similarity(zt, zt1, -1).mean().item()                              # similarité temporelle
            std = z.reshape(-1, m.d).std(0).mean().item()
        print(f"\n=== DIAGNOSTIC (batch {len(ids)}) : d'où vient le plancher de cl ? ===", flush=True)
        print(f"  prédire la MOYENNE globale (baseline trivial haut) : {marg:.4f}", flush=True)
        print(f"  COPIER la frame précédente (z_t -> z_t+1)          : {copy:.4f}", flush=True)
        print(f"  le MODÈLE appris (dynamique, boucle fermée)        : {model:.4f}", flush=True)
        print(f"  cos(z_t, z_t+1) temporel : {cos:.3f}  |  std moyen par dim : {std:.3f}", flush=True)
        print("  lecture : copy≈marg -> encodeur BROUILLE le temps (fondamental, encodeur gelé requis) ;", flush=True)
        print("            copy<<marg mais model>>copy -> le PRÉDICTEUR échoue (réparable).", flush=True)

    if a.tasks > 0:
        m.eval(); scratch = make_env()
        print(f"\n=== PLANIFICATION (protocole DINO-WM, coût {a.cost}) — {a.tasks} tâches, ≤{a.max_steps} pas ===", flush=True)
        for policy in [p_ for p_ in a.policies.split(",") if p_]:
            succ, covs = [], []
            for k in range(a.tasks):
                s_, c_ = run_episode(m, a.hin, P, a.task_seed + k, dev, policy, a.cost, scratch,
                                     max_steps=a.max_steps, plan_h=a.plan_h, pop=a.plan_pop, iters=a.plan_iters,
                                     oracle_h=a.oracle_h, oracle_pop=a.oracle_pop, oracle_iters=a.oracle_iters,
                                     oracle_exec=a.oracle_exec)
                succ.append(s_); covs.append(c_)
            print(f"  {policy:7s}  SR {np.mean(succ):.2f}  coverage moyen {np.mean(covs):.3f}"
                  f"  (max {np.max(covs):.2f})", flush=True)
        scratch.close()
        print("repère DINO-WM (encodeur DINO gelé + dynamique) : SR 0.90", flush=True)

if __name__ == "__main__":
    main()
