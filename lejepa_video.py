"""
LeJEPA (SIGReg, SANS EMA/stop-grad) entraine sur de la VRAIE video (UCF101-subset).

Fidele LeJEPA + structure spatiale (PATCHIFY, facon V-JEPA) : chaque frame -> patches =
tokens spatio-temporels ; on masque des tokens, on predit leurs latents, anti-collapse SIGReg.
Donnees REELLES : UCF101-subset (actions). Sonde action vs pixels bruts vs hasard.

  python lejepa_video.py --steps 2000 --nclass 10 --per_class 60
"""
import argparse, os, glob, tarfile
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2
from huggingface_hub import hf_hub_download

_LE = None
def sigreg(z):
    global _LE
    if _LE is None:
        import lejepa
        _LE = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(t_max=3, n_points=17), num_slices=1024)
        try: _LE = _LE.to(z.device)
        except Exception: pass
    return _LE(z)

def load_frames(path, T, H):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (H, H)))
    cap.release()
    if len(fr) < 2: return None
    a = np.asarray(fr, np.float32) / 255.0
    idx = np.linspace(0, len(a) - 1, T).astype(int)
    return a[idx]                                              # (T, H, H, 3)

def get_data(a):
    p = hf_hub_download("sayakpaul/ucf101-subset", "UCF101_subset.tar.gz", repo_type="dataset")
    root = "/content/ucf_data"
    if not os.path.exists(root):
        os.makedirs(root); tarfile.open(p).extractall(root)
    vids = glob.glob(root + "/**/*.avi", recursive=True)
    byc = defaultdict(list)
    for v in vids: byc[os.path.basename(os.path.dirname(v))].append(v)
    classes = sorted(byc)[:a.nclass]
    X, Y = [], []
    for ci, c in enumerate(classes):
        for v in byc[c][:a.per_class]:
            fr = load_frames(v, a.T, a.H)
            if fr is not None: X.append(fr); Y.append(ci)
    return np.asarray(X, np.float32), np.asarray(Y), classes

def patchify(X, P):
    """(n,T,H,H,3) -> (n, T*nP*nP, P*P*3) : tokens spatio-temporels."""
    n, T, H, _, C = X.shape; nP = H // P
    X = X[:, :, :nP * P, :nP * P, :].reshape(n, T, nP, P, nP, P, C)
    X = X.transpose(0, 1, 2, 4, 3, 5, 6).reshape(n, T * nP * nP, P * P * C)
    return X

class Enc(nn.Module):
    def __init__(s, obs, d, ntok, nl, nh):
        super().__init__()
        s.emb = nn.Linear(obs, d); s.pos = nn.Embedding(ntok, d); s.mtok = nn.Parameter(torch.zeros(d))
        l = nn.TransformerEncoderLayer(d, nh, d * 2, batch_first=True, activation="gelu", dropout=0.0)
        s.tr = nn.TransformerEncoder(l, nl); s.ln = nn.LayerNorm(d)
    def forward(s, o, m=None):
        e = s.emb(o)
        if m is not None: e = torch.where(m.unsqueeze(-1), s.mtok, e)
        x = e + s.pos(torch.arange(o.size(1), device=o.device))
        return s.ln(s.tr(x))

class WM(nn.Module):
    def __init__(s, obs, d, ntok, nl, nh, rw):
        super().__init__(); s.enc = Enc(obs, d, ntok, nl, nh); s.rw = rw
        s.pred = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(s, o, mr):
        B, N, _ = o.shape; m = torch.rand(B, N, device=o.device) < mr
        z = s.enc(o, None); p = s.pred(s.enc(o, m))
        pr = F.smooth_l1_loss(p[m], z[m]) if m.any() else (p - z).pow(2).mean()
        return pr + s.rw * sigreg(z.reshape(-1, z.size(-1)))
    @torch.no_grad()
    def feat(s, o): return s.enc(o).mean(1)

def probe(Xtr, ytr, Xte, yte, dev, nc):
    clf = nn.Linear(Xtr.size(1), nc).to(dev); opt = torch.optim.Adam(clf.parameters(), 1e-2)
    for _ in range(500):
        opt.zero_grad(); F.cross_entropy(clf(Xtr), ytr).backward(); opt.step()
    with torch.no_grad(): return (clf(Xte).argmax(-1) == yte).float().mean().item()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16); p.add_argument("--H", type=int, default=32)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--nclass", type=int, default=10); p.add_argument("--per_class", type=int, default=60)
    p.add_argument("--d_model", type=int, default=256); p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4); p.add_argument("--reg_w", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--mask_ratio", type=float, default=0.5)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y, classes = get_data(a); nc = len(classes)
    Xp = patchify(X, a.patch); ntok = Xp.shape[1]; obs = Xp.shape[2]
    print(f"\n{len(X)} vraies videos UCF101 | {nc} classes {classes}", flush=True)
    print(f"chance {1/nc:.0%} | {ntok} tokens/video ({a.T} frames x {(a.H//a.patch)**2} patches) dim {obs}", flush=True)
    Xt = torch.tensor(Xp); Yt = torch.tensor(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(X), generator=g)
    ntr = int(0.8 * len(X)); tr, te = perm[:ntr], perm[ntr:]
    torch.manual_seed(0); m = WM(obs, a.d_model, ntok, a.n_layer, a.n_head, a.reg_w).to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr); bs = min(a.bs, len(tr))
    for st in range(a.steps):
        bi = tr[torch.randint(0, len(tr), (bs,))]; o = Xt[bi].to(dev)
        loss = m(o, a.mask_ratio); opt.zero_grad(); loss.backward(); opt.step()
        if st % 200 == 0: print(f"  [LeJEPA] step {st} loss {loss.item():.3f}", flush=True)
    @torch.no_grad()
    def feats(idx): return torch.cat([m.feat(Xt[idx[i:i+32]].to(dev)) for i in range(0, len(idx), 32)])
    acc = probe(feats(tr), Yt[tr].to(dev), feats(te), Yt[te].to(dev), dev, nc)
    rtr = Xt[tr].reshape(len(tr), -1).to(dev); rte = Xt[te].reshape(len(te), -1).to(dev)
    racc = probe(rtr, Yt[tr].to(dev), rte, Yt[te].to(dev), dev, nc)
    print(f"\n=== LeJEPA (SIGReg) sur VRAIE video UCF101 — sonde action ({len(te)} test) ===", flush=True)
    print(f"  LeJEPA features : {acc:.2f}", flush=True)
    print(f"  pixels bruts    : {racc:.2f}", flush=True)
    print(f"  hasard          : {1/nc:.2f}", flush=True)

if __name__ == "__main__":
    main()
