"""
TEST CONDUITE : un world model V-JEPA détecte-t-il le DANGER routier (collision imminente) ?

Dataset Nexar (dashcam réel, collision vs conduite normale). Label = time_of_event>0.
On compare :
  #2 (GRAAL)  : encodeur V-JEPA entraîné UNIQUEMENT sur UCF101 (maquillage/sport), GELÉ,
                appliqué aux dashcams -> sonde danger. (transfert cross-domaine)
  #1 (réf)    : encodeur V-JEPA entraîné sur les dashcams elles-mêmes -> sonde danger.
  hasard 0.50.

Archi = vjepa.py (V-JEPA FIDELE : encodeur-contexte sur visibles, prédicteur attentionnel,
masquage par tubelets, multi-masque, SIGReg sans EMA). Résolution montée à H=96 (12x12 patches/frame)
pour que le modèle voie des distinctions fines (piéton/voiture lointaine), pas 32x32.

On évite torchcodec (cast video decode=False) et on décode en OpenCV.

  python driving_transfer.py --n_nexar 400 --ucf_source full --ucf_nclass 50 --ucf_per 40
  (si OOM : baisser --bs, ou --H 64)
"""
import argparse
from argparse import Namespace
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2
from vjepa import patchify, VJEPA, probe, attentive_probe, tube_masks
from lejepa_video import load_frames, get_data

def load_nexar(n, T, H):
    from huggingface_hub import list_repo_files, hf_hub_download
    repo = "nexar-ai/nexar_collision_prediction"
    files = list_repo_files(repo, repo_type="dataset")
    pos = [f for f in files if "positive/" in f and f.endswith(".mp4")][:n // 2]
    neg = [f for f in files if "negative/" in f and f.endswith(".mp4")][:n // 2]
    X, Y = [], []
    for lbl, lst in [(1, pos), (0, neg)]:
        for f in lst:
            try:
                p = hf_hub_download(repo, f, repo_type="dataset")
                fr = load_frames(p, T, H)
                if fr is not None: X.append(fr); Y.append(lbl)
            except Exception as e:
                print("skip", f, str(e)[:60])
    X = np.asarray(X, np.float32); Y = np.array(Y)
    print(f"Nexar : {int((Y == 1).sum())} collisions + {int((Y == 0).sum())} normales", flush=True)
    return X, Y

def train_wm(X, ntok, obs, nP, a, dev, tag):
    """SSL V-JEPA : masquage par tubelets, multi-masque, SIGReg. Encodeur gelé ensuite pour la sonde."""
    torch.manual_seed(0)
    m = VJEPA(obs, a.d_model, ntok, a.n_layer, a.n_head, a.reg_w, a.pred_layers).to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr); Xt = torch.tensor(X); bs = min(a.bs, len(X))
    rng = np.random.default_rng(0)
    for st in range(a.steps):
        bi = np.random.randint(0, len(X), bs); o = Xt[bi].to(dev)
        masks = [mk.to(dev) for mk in tube_masks(bs, a.T, nP, a.mask_ratio, a.n_mask, rng)]
        loss = m(o, masks); opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0: print(f"  [{tag}] step {st} loss {loss.item():.3f}", flush=True)
    return m

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16); p.add_argument("--H", type=int, default=96)
    p.add_argument("--patch", type=int, default=8); p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4); p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--pred_layers", type=int, default=2, help="profondeur du prédicteur attentionnel")
    p.add_argument("--n_mask", type=int, default=2, help="nb de masques-cibles par clip (multi-masque)")
    p.add_argument("--reg_w", type=float, default=1.0); p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=16); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--n_nexar", type=int, default=400)
    p.add_argument("--ucf_source", default="full"); p.add_argument("--ucf_nclass", type=int, default=50)
    p.add_argument("--ucf_per", type=int, default=40)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    nP = a.H // a.patch
    # --- données dashcam (Nexar) ---
    Xn, Yn = load_nexar(a.n_nexar, a.T, a.H); Xn = patchify(Xn, a.patch)
    ntok, obs = Xn.shape[1], Xn.shape[2]
    print(f"résolution H={a.H} patch={a.patch} -> grille {nP}x{nP}, {ntok} tokens/clip, dim {obs}", flush=True)
    Xnt = torch.tensor(Xn); Ynt = torch.tensor(Yn)
    g = torch.Generator().manual_seed(1); pm = torch.randperm(len(Xn), generator=g)
    nt = int(0.7 * len(Xn)); tr, te = pm[:nt].numpy(), pm[nt:].numpy()

    def probe_mlp(Xtr, ytr, Xte, yte):                        # sonde plus profonde (non-linéaire)
        clf = nn.Sequential(nn.Linear(Xtr.size(1), 256), nn.GELU(), nn.Linear(256, 2)).to(dev)
        opt = torch.optim.Adam(clf.parameters(), 1e-3)
        for _ in range(600):
            opt.zero_grad(); F.cross_entropy(clf(Xtr), ytr).backward(); opt.step()
        with torch.no_grad(): return (clf(Xte).argmax(-1) == yte).float().mean().item()
    @torch.no_grad()
    def feats(model, idx): return torch.cat([model.feat(Xnt[torch.tensor(idx[i:i+32])].to(dev)) for i in range(0, len(idx), 32)])
    @torch.no_grad()
    def token_feats(model, idx):                              # latents PAR TOKEN (CPU) pour la sonde attentive
        return torch.cat([model.tokens(Xnt[torch.tensor(idx[i:i+16])].to(dev)).cpu() for i in range(0, len(idx), 16)])
    def run_probe(model):
        Ftr, Fte = feats(model, tr), feats(model, te)                          # poolé (moyenne)
        lin = probe(Ftr, Ynt[tr].to(dev), Fte, Ynt[te].to(dev), dev, 2)
        mlp = probe_mlp(Ftr, Ynt[tr].to(dev), Fte, Ynt[te].to(dev))
        Ttr, Tte = token_feats(model, tr), token_feats(model, te)              # par-token (non diluant)
        att = attentive_probe(Ttr, Ynt[tr], Tte, Ynt[te], dev, 2)
        return lin, mlp, att

    # --- #2 GRAAL : encodeur UCF101 (générique), gelé, sur dashcam ---
    uargs = Namespace(T=a.T, H=a.H, patch=a.patch, source=a.ucf_source, nclass=a.ucf_nclass, per_class=a.ucf_per)
    Xu, _, _ = get_data(uargs); Xu = patchify(Xu, a.patch)
    print(f"\nUCF101 générique : {len(Xu)} vidéos, entraînement de l'encodeur V-JEPA…", flush=True)
    enc_ucf = train_wm(Xu, ntok, obs, nP, a, dev, "UCF→generic")
    lin2, mlp2, att2 = run_probe(enc_ucf)

    # --- #1 réf : encodeur entraîné sur dashcam ---
    print(f"\nEncodeur V-JEPA in-domaine sur dashcam…", flush=True)
    enc_nex = train_wm(Xn[tr], ntok, obs, nP, a, dev, "dashcam")
    lin1, mlp1, att1 = run_probe(enc_nex)

    print(f"\n=== DÉTECTER LE DANGER ROUTIER (collision imminente, {len(te)} test) ===", flush=True)
    print(f"  H={a.H} grille {nP}x{nP} | n_mask={a.n_mask} | mean-pool dilue à haute résolution, l'attentif non", flush=True)
    print(f"  {'':38s}  lin.   MLP    attentif", flush=True)
    print(f"  #2 GRAAL  UCF101 (maquillage/sport) gelé : {lin2:.2f}   {mlp2:.2f}   {att2:.2f}", flush=True)
    print(f"  #1 réf    entraîné sur dashcam           : {lin1:.2f}   {mlp1:.2f}   {att1:.2f}", flush=True)
    print(f"  hasard : 0.50", flush=True)

if __name__ == "__main__":
    main()
