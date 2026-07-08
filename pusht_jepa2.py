"""
PUSH-T — WORLD MODEL JEPA 2 ÉTAPES : encodeur I-JEPA+SIGReg GELÉ, puis dynamique (forme DINO-WM).

Pourquoi (résultat de pusht_jepa.py) : le JEPA CONJOINT (encodeur + dynamique appris ensemble) +
SIGReg ÉCHOUE — la prédiction planche à cl≈0.38 quel que soit rw/stopgrad, MPC = aléatoire. Diag :
COPIER la frame voisine (0.60) est PIRE que prédire la moyenne (0.42), cos(z_t,z_{t+1})=0.20. SIGReg
gaussianise parfaitement la marginale MAIS n'impose aucune CONTINUITÉ → l'encodeur envoie des frames
quasi identiques vers des latents quasi indépendants (anti-lisse). Leçon : SIGReg = régularisateur
anti-collapse, PAS un apprenant de représentation ; dans le vrai LeJEPA il accompagne une tâche
prédictive MASQUÉE (I-JEPA) qui, elle, crée la continuité.

La parade (vision LeCun / DINO-WM) : une TÊTE DE PERCEPTION GELÉE définit un espace latent stable,
le prédicteur imagine dedans.
  ÉTAPE 1 (enc) : pré-entraîner un encodeur en I-JEPA MASQUÉ + SIGReg sur les frames Push-T (réutilise
                  VJEPA de vjepa.py : masquage spatial par blocs, prédicteur attentionnel) -> GELER.
  ÉTAPE 2 (dyn) : dynamique action-conditionnée (Dynamics de pusht_jepa.py) apprise sur les latents
                  GELÉS. Cible stable + lisse -> cl doit enfin descendre. AUCUN SIGReg ici.
  ÉVAL         : planif CEM latente (protocole DINO-WM), réutilise run_episode/cem_latent de pusht_jepa.

  python pusht_jepa2.py --stage enc --enc_steps 30000        # étape 1 : encodeur SSL, puis GELÉ
  python pusht_jepa2.py --stage dyn --dyn_steps 20000        # étape 2 : dynamique sur latents gelés
  python pusht_jepa2.py --stage eval --tasks 10              # planif MPC/oracle/aléa
"""
import argparse, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vjepa import VJEPA, sigreg, tube_masks
from pusht_jepa import (Dynamics, frames_to_tokens, img_to_tokens, cem_latent,
                        run_episode, load_data, default_path, agent_token_mask)
from pusht_plan import make_env, reset_faithful, to64

# ----------------------------------------------------------- world model : encodeur GELÉ + dynamique
class WM2(nn.Module):
    """Interface compatible run_episode/cem_latent (encode / rollout / npf). Encodeur VJEPA GELÉ."""
    def __init__(s, enc, d, npf, dyn_layers=4, nh=4):
        super().__init__()
        s.encoder = enc
        for p in s.encoder.parameters(): p.requires_grad_(False)
        s.encoder.eval()                                # gelé : perception figée, on imagine dedans
        s.dyn = Dynamics(d, npf, dyn_layers, nh)
        s.npf, s.d = npf, d

    def encode(s, tok):                                 # (B,npf,obs) -> (B,npf,d) — VJEPA.tokens est @no_grad
        return s.encoder.tokens(tok)

    def encode_clip(s, tok):                            # (B,T,npf,obs) -> (B,T,npf,d)
        B, T = tok.shape[:2]
        return s.encoder.tokens(tok.reshape(B * T, s.npf, -1)).reshape(B, T, s.npf, s.d)

    def rollout(s, z_prev, z_cur, A):                   # identique à pusht_jepa (boucle fermée)
        outs, prev, cur = [], z_prev, z_cur
        for h in range(A.size(1)):
            nxt = s.dyn(prev, cur, A[:, h]); prev, cur = cur, nxt; outs.append(nxt)
        return torch.stack(outs, 1)

    def dyn_loss(s, tok, A, w_tf=1.0):
        """Cible = latents GELÉS (déjà sans grad). Perte = boucle-fermée + teacher-forced 1-pas.
        Le gradient ne passe QUE par la dynamique (encodeur figé)."""
        B, T = tok.shape[:2]
        z = s.encode_clip(tok)                          # (B,T,npf,d) — figé
        cl = z.new_zeros(())
        if T >= 3:
            cl = F.smooth_l1_loss(s.rollout(z[:, 0], z[:, 1], A[:, 1:T - 1]), z[:, 2:T])
        tf = z.new_zeros(())
        if w_tf > 0 and T >= 3:
            for t in range(1, T - 1):
                tf = tf + F.smooth_l1_loss(s.dyn(z[:, t - 1], z[:, t], A[:, t]), z[:, t + 1])
            tf = tf / (T - 2)
        return cl + w_tf * tf, cl.detach(), tf.detach()

# ----------------------------------------------------------- smoothness des latents (le test qui prédit le succès)
@torch.no_grad()
def latent_smoothness(z, d):
    """z:(B,T,npf,d). copy = prédire z_t+1 par z_t ; marg = prédire la moyenne. copy<<marg => lisse."""
    T = z.shape[1]
    marg = F.smooth_l1_loss(z, z.mean((0, 1, 2), keepdim=True).expand_as(z)).item()
    copy = F.smooth_l1_loss(z[:, 1:T - 1], z[:, 2:T]).item()
    zt, zt1 = z[:, 1:T - 1].reshape(-1, d), z[:, 2:T].reshape(-1, d)
    cos = F.cosine_similarity(zt, zt1, -1).mean().item()
    return marg, copy, cos

# ----------------------------------------------------------- alignement coût-modèle vs vrai progrès
def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))

@torch.no_grad()
def cost_alignment(m, hin, P, dev, scratch, seed, cost_mode, K=96, Hp=4, amax=0.8, frameskip=1):
    """Depuis un état réel, tire K séquences d'actions au hasard ; compare le COÛT LATENT prédit par
    le modèle au VRAI résultat (coverage et distance bloc-but) en simulant ces mêmes actions.
    r_bd > 0 attendu (coût haut <-> bloc loin du but) ; r_cov < 0 (coût haut <-> peu de coverage)."""
    env = make_env(); rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    gp = np.array(info["goal_pose"], np.float32)
    far = np.array([40.0, 40.0], np.float32) if np.linalg.norm(gp[:2] - 40) > 120 else np.array([470.0, 470.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    gtok = img_to_tokens(to64(gob["pixels"], hin, dev), P); zg = m.encode(gtok)[0]
    keep = torch.ones(m.npf, device=dev)
    if cost_mode == "latent_noagent": keep = (~agent_token_mask(gtok, P)).float().to(dev)
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    obs, _, _, _, info = env.step(ag.copy()); f1 = obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
    bp0 = np.array(info["block_pose"], np.float32)
    z0 = m.encode(img_to_tokens(to64(f0, hin, dev), P))[0]
    z1 = m.encode(img_to_tokens(to64(f1, hin, dev), P))[0]
    A = rng.uniform(-amax, amax, (K, Hp, 2)).astype(np.float32)
    Z0 = z0.unsqueeze(0).expand(K, -1, -1); Z1 = z1.unsqueeze(0).expand(K, -1, -1)
    zf = m.rollout(Z0, Z1, torch.tensor(A, device=dev))[:, -1]                # (K,npf,d)
    diff = F.smooth_l1_loss(zf, zg.unsqueeze(0).expand(K, -1, -1), reduction="none").mean(-1)
    cost_model = ((diff * keep).sum(-1) / keep.sum().clamp_min(1)).cpu().numpy()
    true_cov = np.zeros(K, np.float32); true_bd = np.zeros(K, np.float32)
    ff = np.zeros((K, 96, 96, 3), np.uint8)                              # frame réellement atteinte (pour coût VRAI-latent)
    state0 = np.array([ag[0], ag[1], *bp0], np.float32)
    for k in range(K):
        obs2, info2 = reset_faithful(scratch, state0); agk = np.array(obs2["agent_pos"], np.float32); stop = False
        for h in range(Hp):
            act = np.clip(agk + A[k, h] * 256.0, 0, 512).astype(np.float32)
            for _ in range(frameskip):                                   # action maintenue (comme l'entraînement)
                obs2, _, term, trunc, info2 = scratch.step(act); agk = np.array(obs2["agent_pos"], np.float32)
                if term or trunc: stop = True; break
            if stop: break
        true_cov[k] = float(info2.get("coverage", 0.0)); ff[k] = obs2["pixels"]
        true_bd[k] = np.linalg.norm(np.array(info2["block_pose"], np.float32)[:2] - gp[:2])
    env.close()
    # coût VRAI-latent : encoder la frame réellement atteinte -> teste la MÉTRIQUE de l'encodeur seule
    zt = m.encode_clip(frames_to_tokens(torch.from_numpy(ff).unsqueeze(1), hin, P, dev))[:, 0]
    difft = F.smooth_l1_loss(zt, zg.unsqueeze(0).expand(K, -1, -1), reduction="none").mean(-1)
    cost_true = ((difft * keep).sum(-1) / keep.sum().clamp_min(1)).cpu().numpy()
    return (_pearson(cost_model, true_bd), _pearson(cost_true, true_bd), _pearson(cost_model, true_cov),
            float(true_bd.std()), float(cost_model.std()))

# ----------------------------------------------------------- CLI
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="eval", choices=["enc", "dyn", "eval", "probe"])
    p.add_argument("--data", type=str, default="")
    p.add_argument("--enc_ckpt", type=str, default=""); p.add_argument("--dyn_ckpt", type=str, default="")
    p.add_argument("--enc_steps", type=int, default=30000); p.add_argument("--dyn_steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hin", type=int, default=64); p.add_argument("--patch", type=int, default=8)
    p.add_argument("--d", type=int, default=128); p.add_argument("--nl", type=int, default=4)
    p.add_argument("--nh", type=int, default=4); p.add_argument("--pred_layers", type=int, default=2)
    p.add_argument("--dyn_layers", type=int, default=4)
    p.add_argument("--rw", type=float, default=1.0, help="poids SIGReg à l'étape 1 (encodeur)")
    p.add_argument("--mask_ratio", type=float, default=0.6); p.add_argument("--n_masks", type=int, default=4)
    p.add_argument("--w_tf", type=float, default=1.0)
    # éval
    p.add_argument("--tasks", type=int, default=0); p.add_argument("--cost", type=str, default="latent",
                   choices=["latent", "latent_noagent"])
    p.add_argument("--plan_h", type=int, default=4); p.add_argument("--plan_pop", type=int, default=192)
    p.add_argument("--plan_iters", type=int, default=3); p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--oracle_h", type=int, default=10); p.add_argument("--oracle_pop", type=int, default=64)
    p.add_argument("--oracle_iters", type=int, default=3); p.add_argument("--oracle_exec", type=int, default=5)
    p.add_argument("--frameskip", type=int, default=1, help="pas d'env par action (doit matcher les données d'entraînement)")
    p.add_argument("--task_seed", type=int, default=9000)
    p.add_argument("--policies", type=str, default="mpc,oracle,random")
    p.add_argument("--probe_k", type=int, default=96, help="séquences tirées / tâche pour --stage probe")
    p.add_argument("--probe_seeds", type=int, default=5, help="nb de tâches pour --stage probe")
    return p.parse_args()

def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = a.data or default_path("pusht_data.npz")
    enc_ckpt = a.enc_ckpt or default_path("pusht_enc.pt")
    dyn_ckpt = a.dyn_ckpt or default_path("pusht_dyn.pt")
    P = a.patch; npf = (a.hin // P) ** 2; obs = P * P * 3; nP = a.hin // P

    # ===================== ÉTAPE 1 : encodeur I-JEPA masqué + SIGReg, puis GELÉ =====================
    if a.stage == "enc":
        X, _ = load_data(data); n, T = X.shape[0], X.shape[1]
        Xf = X.reshape(n * T, X.shape[2], X.shape[3], 3)                     # frames à plat (I-JEPA per-frame)
        enc = VJEPA(obs, a.d, npf, a.nl, a.nh, a.rw, a.pred_layers).to(dev)
        opt = torch.optim.Adam(enc.parameters(), a.lr); rng = np.random.default_rng(0)
        print(f"ÉTAPE 1 (encodeur SSL) : {Xf.shape[0]} frames, I-JEPA masqué (ratio {a.mask_ratio}, "
              f"{a.n_masks} masques) + SIGReg rw {a.rw}, {a.enc_steps} pas, dev {dev}", flush=True)
        for st in range(a.enc_steps):
            ids = torch.from_numpy(rng.integers(0, Xf.shape[0], a.bs))
            o = frames_to_tokens(Xf[ids].unsqueeze(1), a.hin, P, dev)[:, 0]  # (bs,npf,obs)
            masks = [mm.to(dev) for mm in tube_masks(a.bs, 1, nP, a.mask_ratio, a.n_masks, rng)]
            loss = enc(o, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            if st % 500 == 0:
                with torch.no_grad(): rg = sigreg(enc.tokens(o).reshape(-1, a.d)).item()
                print(f"  step {st:6d}  loss {loss.item():.4f}  (sigreg latents {rg:.4f})", flush=True)
        torch.save({"model": enc.state_dict(),
                    "args": {"hin": a.hin, "patch": P, "d": a.d, "nl": a.nl, "nh": a.nh,
                             "pred_layers": a.pred_layers, "rw": a.rw}}, enc_ckpt)
        print(f"encodeur (à GELER) -> {enc_ckpt}", flush=True)
        return

    # charge l'encodeur gelé (étapes dyn + eval)
    if not os.path.exists(enc_ckpt):
        raise SystemExit(f"encodeur introuvable : {enc_ckpt} — lance d'abord --stage enc")
    eb = torch.load(enc_ckpt, map_location=dev); ea = eb["args"]
    a.hin = ea["hin"]; P = ea["patch"]; npf = (a.hin // P) ** 2; obs = P * P * 3
    enc = VJEPA(obs, ea["d"], npf, ea["nl"], ea["nh"], ea["rw"], ea["pred_layers"]).to(dev)
    enc.load_state_dict(eb["model"])
    m = WM2(enc, ea["d"], npf, a.dyn_layers, a.nh).to(dev)
    print(f"encodeur GELÉ chargé : {enc_ckpt}  (hin {a.hin}, d {ea['d']}, npf {npf})", flush=True)

    # ===================== ÉTAPE 2 : dynamique sur latents GELÉS =====================
    if a.stage == "dyn":
        X, act = load_data(data); n, T = X.shape[0], X.shape[1]
        # sanity : l'encodeur gelé est-il temporellement LISSE ? (prédit le succès de l'étape 2)
        ids0 = torch.from_numpy(np.random.default_rng(1).integers(0, n, min(256, n)))
        with torch.no_grad():
            z0 = m.encode_clip(frames_to_tokens(X[ids0], a.hin, P, dev))
            marg, copy, cos = latent_smoothness(z0, m.d)
        print(f"SMOOTHNESS de l'encodeur gelé : moyenne {marg:.4f} | COPY {copy:.4f} | cos {cos:.3f}"
              f"  ({'LISSE ✓ (copy<<moyenne)' if copy < 0.8 * marg else 'ANTI-LISSE ✗ — étape 2 vouée à échouer'})", flush=True)
        opt = torch.optim.Adam(m.dyn.parameters(), a.lr); rng = np.random.default_rng(0)
        print(f"ÉTAPE 2 (dynamique) : {a.dyn_steps} pas, bs {a.bs}, latents GELÉS, dev {dev}", flush=True)
        for st in range(a.dyn_steps):
            ids = torch.from_numpy(rng.integers(0, n, a.bs))
            tok = frames_to_tokens(X[ids], a.hin, P, dev); A = act[ids].to(dev)
            loss, cl, tf = m.dyn_loss(tok, A, w_tf=a.w_tf)
            opt.zero_grad(); loss.backward(); opt.step()
            if st % 500 == 0:
                print(f"  step {st:6d}  loss {loss.item():.4f}  (cl {cl.item():.4f}  tf {tf.item():.4f})", flush=True)
        torch.save({"dyn": m.dyn.state_dict(), "dyn_layers": a.dyn_layers}, dyn_ckpt)
        print(f"dynamique -> {dyn_ckpt}", flush=True)
        return

    # ===================== ÉVAL / PROBE : chargent la dynamique =====================
    if not os.path.exists(dyn_ckpt):
        raise SystemExit(f"dynamique introuvable : {dyn_ckpt} — lance d'abord --stage dyn")
    db = torch.load(dyn_ckpt, map_location=dev); m.dyn.load_state_dict(db["dyn"]); m.eval()
    print(f"dynamique chargée : {dyn_ckpt}", flush=True)

    if a.stage == "probe":                              # le coût latent est-il corrélé au vrai progrès ?
        scratch = make_env()
        print(f"\n=== PROBE alignement coût-modèle vs vrai (coût {a.cost}, {a.probe_k} séq/tâche) ===", flush=True)
        rbd, rtrue = [], []
        for k in range(a.probe_seeds):
            r_bd, r_true, r_cov, bdstd, cstd = cost_alignment(m, a.hin, P, dev, scratch, a.task_seed + k, a.cost,
                                                              K=a.probe_k, Hp=a.plan_h, frameskip=a.frameskip)
            rbd.append(r_bd); rtrue.append(r_true)
            print(f"  tâche {k}: r(MODÈLE, dist) {r_bd:+.2f}  |  r(VRAI-latent, dist) {r_true:+.2f}"
                  f"  (σ dist {bdstd:.0f}px, σ coût-modèle {cstd:.3f})", flush=True)
        scratch.close()
        print(f"  MOYENNE : r(MODÈLE) {np.mean(rbd):+.2f}  |  r(VRAI-latent) {np.mean(rtrue):+.2f}", flush=True)
        print("  lecture : r(VRAI-latent) élevé mais r(MODÈLE)~0 -> DYNAMIQUE sous-réactive (pondérer le mouvement) ;", flush=True)
        print("            r(VRAI-latent)~0 aussi -> la MÉTRIQUE de l'encodeur ne suit pas la position du bloc (autre readout).", flush=True)
        return

    if a.tasks > 0:
        scratch = make_env()
        print(f"\n=== PLANIFICATION (2 étapes, coût {a.cost}) — {a.tasks} tâches, ≤{a.max_steps} pas ===", flush=True)
        for policy in [p_ for p_ in a.policies.split(",") if p_]:
            succ, covs = [], []
            for k in range(a.tasks):
                s_, c_ = run_episode(m, a.hin, P, a.task_seed + k, dev, policy, a.cost, scratch,
                                     max_steps=a.max_steps, plan_h=a.plan_h, pop=a.plan_pop, iters=a.plan_iters,
                                     oracle_h=a.oracle_h, oracle_pop=a.oracle_pop, oracle_iters=a.oracle_iters,
                                     oracle_exec=a.oracle_exec, frameskip=a.frameskip)
                succ.append(s_); covs.append(c_)
            print(f"  {policy:7s}  SR {np.mean(succ):.2f}  coverage moyen {np.mean(covs):.3f}"
                  f"  (max {np.max(covs):.2f})", flush=True)
        scratch.close()
        print("repère DINO-WM (encodeur DINO gelé + dynamique) : SR 0.90", flush=True)

if __name__ == "__main__":
    main()
