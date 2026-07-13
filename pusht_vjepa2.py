"""
PUSH-T × V-JEPA 2 GELÉ (Meta) — ABLATION : notre encodeur faible est-il LE problème ?

On garde tout le pipeline (sonde pose, plus tard dynamique + planif) et on remplace SEULEMENT notre
petit encodeur maison par le VRAI V-JEPA 2 pré-entraîné et gelé (ViT-L, 1M h de vidéo). Test :
  - la SONDE POSE se met-elle à décoder la pose du bloc depuis les features V-JEPA 2 ?
    OUI  -> c'était bien notre encodeur (« Chose 1 » trop faible) ; on branche la dynamique dessus.
    NON  -> le problème est ailleurs (sonde, tâche, ou la pose n'est pas dans les patches) — pas l'encodeur.

V-JEPA 2 = « Chose 1 » (world model général gelé) ; notre dynamique+planif Push-T = « Chose 2 »
(exactement la séparation V-JEPA 2 -> V-JEPA 2-AC de Meta).

  pip install -U transformers
  python pusht_vjepa2.py --stage pose_probe            # le test décisif (features V-JEPA 2 -> pose)
"""
import argparse, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pusht_jepa2 import PoseReadout
from pusht_jepa import default_path, Dynamics
from pusht_plan import make_env, reset_faithful, oracle_plan

# ----------------------------------------------------------- V-JEPA 2 gelé : frame Push-T -> tokens
_VJ = {"model": None, "proc": None}
def load_vjepa2(name, dev):
    if _VJ["model"] is None:
        from transformers import AutoModel, AutoVideoProcessor
        _VJ["proc"] = AutoVideoProcessor.from_pretrained(name)
        _VJ["model"] = AutoModel.from_pretrained(name, attn_implementation="sdpa").to(dev).eval()
        for p in _VJ["model"].parameters(): p.requires_grad_(False)
    return _VJ["model"], _VJ["proc"]

@torch.no_grad()
def encode_frames(frames, name, dev, clip_len=2, bs=16, verbose=True):
    """frames:(N,96,96,3) uint8 -> features V-JEPA 2 (N, ntok, d) fp16 CPU. Chaque frame -> clip
    [frame×clip_len] (tubelet_size=2) -> grille de patches 16×16 = 256 tokens de dim 1024."""
    model, proc = load_vjepa2(name, dev)
    out = []
    for i in range(0, len(frames), bs):
        batch = frames[i:i + bs]
        pv = []
        for f in batch:
            clip = np.repeat(np.asarray(f)[None], clip_len, 0)               # (clip_len,96,96,3)
            pv.append(proc(list(clip), return_tensors="pt")["pixel_values_videos"])
        pv = torch.cat(pv, 0).to(dev)
        z = model(pixel_values_videos=pv, skip_predictor=True).last_hidden_state
        out.append(z.float().cpu().half())
        if verbose and (i // bs) % 20 == 0: print(f"  encodage V-JEPA 2 : {i + len(batch)}/{len(frames)}", flush=True)
    return torch.cat(out, 0)

def enc_gpu(frames, a, dev):
    """(n,96,96,3) uint8 -> (n, ntok, d) float sur dev (encodage V-JEPA 2 en direct, silencieux)."""
    return encode_frames(np.asarray(frames), a.model, dev, a.clip_len, a.enc_bs, verbose=False).float().to(dev)

# ----------------------------------------------------------- dynamique + coût pose (planif façon V-JEPA 2-AC)
def rollout(dyn, zp, zc, A, ckpt=False):
    """ckpt=True : gradient checkpointing (recalcule au backward) -> mémoire ~1 passe au lieu de Hp."""
    from torch.utils.checkpoint import checkpoint
    outs, prev, cur = [], zp, zc
    for h in range(A.size(1)):
        nxt = (checkpoint(dyn, prev, cur, A[:, h], use_reentrant=False)
               if (ckpt and torch.is_grad_enabled()) else dyn(prev, cur, A[:, h]))
        prev, cur = cur, nxt; outs.append(nxt)
    return torch.stack(outs, 1)

def pose_cost(ph, pg, w_ang):
    """ph:(...,4) pose imaginée [x,y,sin,cos], pg:(4,) pose but. Position + angle (via sin/cos)."""
    pos = (ph[..., :2] - pg[:2]).norm(dim=-1)
    dsin = ph[..., 2] * pg[3] - ph[..., 3] * pg[2]; dcos = ph[..., 3] * pg[3] + ph[..., 2] * pg[2]
    ang = torch.atan2(dsin.abs(), dcos) / math.pi                            # |Δangle| ∈ [0,1]
    return pos + w_ang * ang

def cem_vj2(dyn, ro, z0, z1, pg, a, dev, mu0=None):
    Hp, pop, iters, elite, amax = a.plan_h, a.plan_pop, a.plan_iters, 16, 0.8
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        Z0 = z0.unsqueeze(0).expand(pop, -1, -1); Z1 = z1.unsqueeze(0).expand(pop, -1, -1)
        mu = torch.zeros(Hp, 2, device=dev) if mu0 is None else mu0.clone()
        sg = torch.full((Hp, 2), 0.30, device=dev)
        for _ in range(iters):
            e1 = torch.randn(pop, 1, 2, device=dev).expand(-1, Hp, -1).clone()
            e2 = torch.randn(pop, 2, 2, device=dev).repeat_interleave((Hp + 1) // 2, 1)[:, :Hp]
            eps = torch.where((torch.arange(pop, device=dev) % 2 == 0).reshape(-1, 1, 1), e1, e2)
            A = (mu + sg * eps).clamp(-amax, amax)
            zf = rollout(dyn, Z0, Z1, A)[:, -1]                              # (pop,npf,d) latent imaginé final
            cost = pose_cost(ro(zf), pg, a.w_ang)                           # coût en espace POSE (readout=critic)
            el = A[cost.argsort()[:elite]]; mu = el.mean(0); sg = (el.std(0) + 1e-4).clamp_min(0.08)
    return mu.clamp(-amax, amax)

def run_episode_vj2(a, dev, dyn, ro, seed, policy, scratch):
    env = make_env(); rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed); gp = np.array(info["goal_pose"], np.float32)
    far = np.array([40.0, 40.0], np.float32) if np.linalg.norm(gp[:2] - 40) > 120 else np.array([470.0, 470.0], np.float32)
    gob, _ = reset_faithful(scratch, np.array([far[0], far[1], gp[0], gp[1], gp[2]], np.float32))
    with torch.no_grad(): pg = ro(enc_gpu([gob["pixels"]], a, dev))[0]       # pose-but LUE de l'image-but (readout)
    ag = np.array(obs["agent_pos"], np.float32); f0 = obs["pixels"]
    for _ in range(a.frameskip): obs, _, _, _, info = env.step(ag.copy())    # 1 transition frameskip -> contexte (f0,f1)
    f1 = obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
    best, success, mu = float(info.get("coverage", 0.0)), False, None
    for _ in range(a.max_steps):
        if policy == "mpc":
            with torch.no_grad(): z0 = enc_gpu([f0], a, dev)[0]; z1 = enc_gpu([f1], a, dev)[0]
            plan = cem_vj2(dyn, ro, z0, z1, pg, a, dev, mu)
            delta = plan[0].cpu().numpy(); mu = torch.cat([plan[1:], torch.zeros(1, 2, device=dev)])
        elif policy == "oracle":
            state = np.array([ag[0], ag[1], *np.array(info["block_pose"], np.float32)], np.float32)
            delta = oracle_plan(scratch, state, gp, rng, Hp=a.plan_h, pop=48, iters=3)[0]
        else:
            th = rng.uniform(0, 2 * np.pi); delta = rng.uniform(0.2, 1.0) * 0.8 * np.array([np.cos(th), np.sin(th)], np.float32)
        act = np.clip(ag + delta * 256.0, 0, 512).astype(np.float32); done = False
        for _ in range(a.frameskip):
            obs, _, term, trunc, info = env.step(act); best = max(best, float(info.get("coverage", 0.0)))
            if info.get("is_success", False) or term: done = True; break
            if trunc: break
        f0, f1 = f1, obs["pixels"]; ag = np.array(obs["agent_pos"], np.float32)
        if done: success = True; break
        if trunc: break
    env.close()
    return success, best

# ----------------------------------------------------------- ÉTAPES dyn / readout / eval
def stage_dyn(a, dev):
    feat_path = a.cache or default_path("pusht_vj2feat.npy")
    meta = np.load(feat_path.replace(".npy", "_meta.npz"))
    feat = np.load(feat_path, mmap_mode="r"); N, T, npf, d = feat.shape       # MEMMAP (pas de RAM 9.4Go)
    A = torch.from_numpy(meta["A"].astype(np.float32)); AG = torch.from_numpy(meta["AG"].astype(np.float32))
    act = (A - AG[:, :-1]) / 256.0
    rng = np.random.default_rng(0)
    amp = (dev == "cuda")                                                     # precision mixte bf16 (mémoire /2)
    def load(ids): return torch.from_numpy(np.ascontiguousarray(feat[ids])).float().to(dev)
    with torch.no_grad():
        z = load(rng.integers(0, N, 128))
        copy = F.smooth_l1_loss(z[:, 1:T - 1], z[:, 2:T]).item()
        marg = F.smooth_l1_loss(z, z.mean((0, 1, 2), keepdim=True).expand_as(z)).item()
    print(f"features V-JEPA 2 : copy {copy:.4f}  |  moyenne {marg:.4f}  (la dynamique doit battre copy)", flush=True)
    dyn = Dynamics(d, npf, a.dyn_layers, a.nh).to(dev); opt = torch.optim.Adam(dyn.parameters(), a.lr)
    print(f"ÉTAPE dyn : {a.dyn_steps} pas, d {d}, npf {npf}, bs {a.bs}, bf16 {amp}, dev {dev}", flush=True)
    for st in range(a.dyn_steps):
        ids = rng.integers(0, N, a.bs)
        z = load(ids); Ai = act[torch.from_numpy(ids)].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss = F.smooth_l1_loss(rollout(dyn, z[:, 0], z[:, 1], Ai[:, 1:T - 1], ckpt=True), z[:, 2:T])
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(dyn.parameters(), a.clip); opt.step()
        if st % 500 == 0: print(f"  step {st:5d}  cl {loss.item():.4f}  (copy {copy:.3f})", flush=True)
    torch.save({"dyn": dyn.state_dict(), "d": d, "npf": npf, "dyn_layers": a.dyn_layers, "nh": a.nh},
               a.dyn_ckpt or default_path("pusht_vj2dyn.pt"))
    print(f"dynamique -> {a.dyn_ckpt or default_path('pusht_vj2dyn.pt')}", flush=True)

def stage_readout(a, dev):
    feat_path = a.cache or default_path("pusht_vj2feat.npy")
    meta = np.load(feat_path.replace(".npy", "_meta.npz"))
    feat = torch.from_numpy(np.load(feat_path)); N, T, npf, d = feat.shape
    featf = feat.reshape(N * T, npf, d); BP = torch.from_numpy(meta["BP"].astype(np.float32)).reshape(N * T, 3)
    ntr = int(0.9 * N * T); rng = np.random.default_rng(1)
    ro = PoseReadout(d).to(dev); opt = torch.optim.Adam(ro.parameters(), 1e-3)
    print(f"ÉTAPE readout (critic) : {a.readout_steps} pas sur {N * T} frames en cache", flush=True)
    for st in range(a.readout_steps):
        ids = torch.from_numpy(rng.integers(0, ntr, a.bs))
        z = featf[ids].float().to(dev); tg = BP[ids].to(dev)
        ang = tg[:, 2]; sc = torch.stack([ang.sin(), ang.cos()], -1)
        pred = ro(z); loss = F.mse_loss(pred[:, :2], tg[:, :2] / 512.0) + F.mse_loss(pred[:, 2:], sc)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0: print(f"  step {st:5d}  loss {loss.item():.4f}", flush=True)
    ro.eval(); pe, ae, nb = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(ntr, N * T, a.bs):
            z = featf[i:i + a.bs].float().to(dev); tg = BP[i:i + a.bs].to(dev); pr = ro(z)
            pe += (pr[:, :2] * 512.0 - tg[:, :2]).norm(dim=-1).sum().item()
            da = torch.atan2(pr[:, 2], pr[:, 3]) - tg[:, 2]
            ae += torch.atan2(da.sin(), da.cos()).abs().sum().item() * 180.0 / math.pi; nb += len(z)
    print(f"readout (test) : POSITION {pe / nb:.1f} px  |  ANGLE {ae / nb:.1f}°", flush=True)
    torch.save({"ro": ro.state_dict(), "d": d}, a.readout_ckpt or default_path("pusht_vj2ro.pt"))
    print(f"readout -> {a.readout_ckpt or default_path('pusht_vj2ro.pt')}", flush=True)

def stage_eval(a, dev):
    db = torch.load(a.dyn_ckpt or default_path("pusht_vj2dyn.pt"), map_location=dev)
    dyn = Dynamics(db["d"], db["npf"], db["dyn_layers"], db["nh"]).to(dev); dyn.load_state_dict(db["dyn"]); dyn.eval()
    rb = torch.load(a.readout_ckpt or default_path("pusht_vj2ro.pt"), map_location=dev)
    ro = PoseReadout(rb["d"]).to(dev); ro.load_state_dict(rb["ro"]); ro.eval()
    scratch = make_env()
    print(f"\n=== PLANIF V-JEPA 2-AC (coût POSE, frameskip {a.frameskip}) — {a.tasks} tâches, ≤{a.max_steps} pas ===", flush=True)
    for policy in [p_ for p_ in a.policies.split(",") if p_]:
        succ, covs = [], []
        for k in range(a.tasks):
            s_, c_ = run_episode_vj2(a, dev, dyn, ro, a.task_seed + k, policy, scratch)
            succ.append(s_); covs.append(c_)
        print(f"  {policy:7s}  SR {np.mean(succ):.2f}  coverage moyen {np.mean(covs):.3f}  (max {np.max(covs):.2f})", flush=True)
    scratch.close()
    print("repère DINO-WM (DINOv2 gelé + dynamique latente) : SR 0.90", flush=True)

# ----------------------------------------------------------- CACHE : features V-JEPA 2 des données Push-T
def cache_features(a, dev, data):
    """Précalcule les features V-JEPA 2 (lentes, ViT-L) une fois -> memmap fp16 sur Drive, pour
    entraîner la dynamique/le readout ensuite sans réencoder. + méta (actions, poses, but)."""
    dd = np.load(data); X = dd["X"]; n, T = X.shape[:2]; N = min(a.n_cache, n)
    feat_path = a.cache or default_path("pusht_vj2feat.npy")
    meta_path = feat_path.replace(".npy", "_meta.npz")
    z0 = encode_frames(X[0, :1], a.model, dev, a.clip_len, 1)                 # infère (ntok, d)
    ntok, d = z0.shape[1], z0.shape[2]
    feat = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float16, shape=(N, T, ntok, d))
    print(f"CACHE -> {feat_path}  shape {(N, T, ntok, d)}  (~{N * T * ntok * d * 2 / 1e9:.1f} Go fp16)", flush=True)
    flat = feat.reshape(N * T, ntok, d); Xf = X[:N].reshape(N * T, 96, 96, 3)
    CH = a.enc_bs * 8
    for i in range(0, N * T, CH):
        z = encode_frames(Xf[i:i + CH], a.model, dev, a.clip_len, a.enc_bs)
        flat[i:i + len(z)] = z.numpy(); print(f"  {min(i + CH, N * T)}/{N * T} frames en cache", flush=True)
    feat.flush()
    np.savez(meta_path, A=dd["A"][:N], AG=dd["AG"][:N], BP=dd["BP"][:N], GP=dd["GP"], N=N, T=T)
    print(f"META -> {meta_path}  |  cache prêt (encodeur = {a.model})", flush=True)

# ----------------------------------------------------------- CLI
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="pose_probe",
                   choices=["pose_probe", "cache", "dyn", "readout", "eval"])
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--data", type=str, default="")
    p.add_argument("--cache", type=str, default="", help="chemin du memmap de features (défaut Drive)")
    p.add_argument("--n_cache", type=int, default=3000, help="séquences à mettre en cache (fp16 ~3.1Mo/frame)")
    p.add_argument("--n_probe", type=int, default=6000, help="frames Push-T échantillonnées pour la sonde")
    p.add_argument("--probe_steps", type=int, default=6000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--enc_bs", type=int, default=16, help="batch d'encodage V-JEPA 2 (ViT-L, mémoire GPU)")
    p.add_argument("--clip_len", type=int, default=2)
    # dynamique / readout / éval
    p.add_argument("--dyn_ckpt", type=str, default=""); p.add_argument("--readout_ckpt", type=str, default="")
    p.add_argument("--dyn_steps", type=int, default=20000); p.add_argument("--readout_steps", type=int, default=6000)
    p.add_argument("--dyn_layers", type=int, default=4); p.add_argument("--nh", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--tasks", type=int, default=10); p.add_argument("--policies", type=str, default="mpc,random")
    p.add_argument("--plan_h", type=int, default=4); p.add_argument("--plan_pop", type=int, default=96)
    p.add_argument("--plan_iters", type=int, default=3); p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--frameskip", type=int, default=5); p.add_argument("--w_ang", type=float, default=0.5)
    p.add_argument("--task_seed", type=int, default=9000)
    return p.parse_args()

def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = a.data or default_path("pusht_data.npz")
    if a.stage == "cache":   cache_features(a, dev, data); return
    if a.stage == "dyn":     stage_dyn(a, dev); return
    if a.stage == "readout": stage_readout(a, dev); return
    if a.stage == "eval":    stage_eval(a, dev); return
    dd = np.load(data)
    X = dd["X"].reshape(-1, 96, 96, 3); BP = dd["BP"].reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X), min(a.n_probe, len(X)), replace=False)
    Xs = X[sel]; BPs = torch.from_numpy(BP[sel])
    print(f"V-JEPA 2 gelé « {a.model} » : encodage de {len(Xs)} frames Push-T…", flush=True)
    feats = encode_frames(Xs, a.model, dev, clip_len=a.clip_len, bs=a.enc_bs)   # (N, ntok, d) fp16 CPU
    N, ntok, d = feats.shape
    print(f"features : {tuple(feats.shape)}  (ntok {ntok}, d {d})", flush=True)

    ntr = int(0.85 * N)
    probe = PoseReadout(d).to(dev); opt = torch.optim.Adam(probe.parameters(), 1e-3)
    print(f"SONDE POSE sur features V-JEPA 2 : {a.probe_steps} pas", flush=True)
    for st in range(a.probe_steps):
        ids = torch.from_numpy(rng.integers(0, ntr, a.bs))
        z = feats[ids].float().to(dev); tgt = BPs[ids].to(dev)
        ang = tgt[:, 2]; sc = torch.stack([ang.sin(), ang.cos()], -1)
        pred = probe(z)
        loss = F.mse_loss(pred[:, :2], tgt[:, :2] / 512.0) + F.mse_loss(pred[:, 2:], sc)
        if not torch.isfinite(loss): print(f"  step {st}: loss non-finie, skip", flush=True); continue
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0: print(f"  step {st:5d}  loss {loss.item():.4f}", flush=True)

    probe.eval(); pe, ae, nb = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(ntr, N, a.bs):
            z = feats[i:i + a.bs].float().to(dev); tg = BPs[i:i + a.bs].to(dev); pr = probe(z)
            pe += (pr[:, :2] * 512.0 - tg[:, :2]).norm(dim=-1).sum().item()
            da = torch.atan2(pr[:, 2], pr[:, 3]) - tg[:, 2]
            ae += torch.atan2(da.sin(), da.cos()).abs().sum().item() * 180.0 / math.pi; nb += len(z)
    base_px = (BPs[ntr:, :2] - BPs[:ntr, :2].mean(0)).norm(dim=-1).mean().item()
    print(f"\n=== SONDE POSE / V-JEPA 2 gelé (test {nb} frames) : POSITION {pe / nb:.1f} px  |  ANGLE {ae / nb:.1f}°"
          f"  (baseline sans info ~{base_px:.0f} px, 90°  ;  notre encodeur maison : 85px/79°)", flush=True)
    print("  lecture : position ~qq px / angle bas -> c'était bien NOTRE encodeur (Chose 1) le problème ;", flush=True)
    print("            toujours grosse -> le problème est AILLEURS (sonde/tâche/pose pas dans les patches).", flush=True)

if __name__ == "__main__":
    main()
