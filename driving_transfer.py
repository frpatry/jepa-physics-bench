"""
TEST CONDUITE : un world model LeJEPA détecte-t-il le DANGER routier (collision imminente) ?

Dataset Nexar (dashcam réel, collision vs conduite normale). Label = time_of_event>0.
On compare :
  #2 (GRAAL)  : encodeur LeJEPA entraîné UNIQUEMENT sur UCF101 (maquillage/sport), GELÉ,
                appliqué aux dashcams -> sonde danger. (transfert cross-domaine)
  #1 (réf)    : encodeur LeJEPA entraîné sur les dashcams elles-mêmes -> sonde danger.
  hasard 0.50.

On évite torchcodec (cast video decode=False) et on décode en OpenCV.

  python driving_transfer.py --n_nexar 400 --ucf_source full --ucf_nclass 50 --ucf_per 40
"""
import argparse, os, tempfile
from argparse import Namespace
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2
from lejepa_video import patchify, WM, probe, load_frames, get_data

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

def train_wm(X, ntok, obs, a, dev, tag):
    torch.manual_seed(0); m = WM(obs, a.d_model, ntok, a.n_layer, a.n_head, a.reg_w).to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr); Xt = torch.tensor(X); bs = min(a.bs, len(X))
    for st in range(a.steps):
        bi = np.random.randint(0, len(X), bs); o = Xt[bi].to(dev)
        loss = m(o, a.mask_ratio); opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0: print(f"  [{tag}] step {st} loss {loss.item():.3f}", flush=True)
    return m

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16); p.add_argument("--H", type=int, default=32)
    p.add_argument("--patch", type=int, default=8); p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4); p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--reg_w", type=float, default=1.0); p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--n_nexar", type=int, default=400)
    p.add_argument("--ucf_source", default="full"); p.add_argument("--ucf_nclass", type=int, default=50)
    p.add_argument("--ucf_per", type=int, default=40)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    # --- données dashcam (Nexar) ---
    Xn, Yn = load_nexar(a.n_nexar, a.T, a.H); Xn = patchify(Xn, a.patch)
    ntok, obs = Xn.shape[1], Xn.shape[2]
    Xnt = torch.tensor(Xn); Ynt = torch.tensor(Yn)
    g = torch.Generator().manual_seed(1); pm = torch.randperm(len(Xn), generator=g)
    nt = int(0.7 * len(Xn)); tr, te = pm[:nt].numpy(), pm[nt:].numpy()

    @torch.no_grad()
    def feats(model, idx): return torch.cat([model.feat(Xnt[torch.tensor(idx[i:i+32])].to(dev)) for i in range(0, len(idx), 32)])
    def run_probe(model):
        return probe(feats(model, tr), Ynt[tr].to(dev), feats(model, te), Ynt[te].to(dev), dev, 2)

    # --- #2 GRAAL : encodeur UCF101 (générique), gelé, sur dashcam ---
    uargs = Namespace(T=a.T, H=a.H, patch=a.patch, source=a.ucf_source, nclass=a.ucf_nclass, per_class=a.ucf_per)
    Xu, _, _ = get_data(uargs); Xu = patchify(Xu, a.patch)
    print(f"\nUCF101 générique : {len(Xu)} vidéos, entraînement de l'encodeur…", flush=True)
    enc_ucf = train_wm(Xu, ntok, obs, a, dev, "UCF→generic")
    acc2 = run_probe(enc_ucf)

    # --- #1 réf : encodeur entraîné sur dashcam ---
    print(f"\nEncodeur in-domaine sur dashcam…", flush=True)
    enc_nex = train_wm(Xn[tr], ntok, obs, a, dev, "dashcam")
    acc1 = run_probe(enc_nex)

    print(f"\n=== DÉTECTER LE DANGER ROUTIER (collision imminente, {len(te)} test) ===", flush=True)
    print(f"  #2 GRAAL  UCF101 (maquillage/sport) gelé -> dashcam : {acc2:.2f}", flush=True)
    print(f"  #1 réf    entraîné sur dashcam            -> dashcam : {acc1:.2f}", flush=True)
    print(f"  hasard : 0.50", flush=True)

if __name__ == "__main__":
    main()
