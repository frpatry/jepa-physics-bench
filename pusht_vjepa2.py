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
from pusht_jepa import default_path

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
def encode_frames(frames, name, dev, clip_len=2, bs=16):
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
        if (i // bs) % 20 == 0: print(f"  encodage V-JEPA 2 : {i + len(batch)}/{len(frames)}", flush=True)
    return torch.cat(out, 0)

# ----------------------------------------------------------- CLI
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="pose_probe", choices=["pose_probe"])
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--data", type=str, default="")
    p.add_argument("--n_probe", type=int, default=6000, help="frames Push-T échantillonnées pour la sonde")
    p.add_argument("--probe_steps", type=int, default=6000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--enc_bs", type=int, default=16, help="batch d'encodage V-JEPA 2 (ViT-L, mémoire GPU)")
    p.add_argument("--clip_len", type=int, default=2)
    return p.parse_args()

def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = a.data or default_path("pusht_data.npz")
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
