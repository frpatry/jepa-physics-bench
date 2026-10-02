"""
AV + V-JEPA 2 — ÉTAPE 0 : le « cortex visuel » de Meta (V-JEPA 2 gelé, pré-entraîné sans étiquettes
sur ~1M h de vidéo) VOIT-IL notre monde de disques ?

Avant de brancher le son sur les latents V-JEPA 2 (étape 1), on vérifie par sondes gelées que ses
latents contiennent ce dont le liage son<->disque a besoin : positions des disques, instants de choc,
et (bonus) le ratio de masses lisible dans la dynamique d'un choc.

Clip = les 16 frames d'une séquence (32 px agrandies en 256 par le processeur officiel) ->
V-JEPA 2 ViT-L -> 8 pas temporels (tubelet 2) × 16×16 patches × 1024 -> moyenne spatiale --pool
(4 -> grille 4×4 = 128 tokens/clip). Comparaison : MÊMES sondes sur les PIXELS BRUTS (patches 8×8
des 2 frames de chaque pas temporel) = ce que la sonde tire sans aucun encodeur.

Sondes (jeu tenu à l'écart) : positions des 2 disques (R², par pas temporel), impact par pas
(exactitude équilibrée), log-masse et ratio de masses (R², séquences avec choc).
Matériau non sondé ici : invisible par construction (aucun intérêt en vision seule).

  python av_vjepa2.py --n 3000            # Colab GPU (~15-30 min)
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from av_jepa import gen_world
from vjepa import AttentiveProbe, patchify

def load_vjepa2(name, dev):
    from transformers import AutoModel, AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation="sdpa").to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, proc

@torch.no_grad()
def encode_vjepa2(X, a, dev):
    """X:(n,T,H,H,3) float [0,1] -> (n, T/2, g*g, 1024) fp16 CPU, g = 16/pool."""
    model, proc = load_vjepa2(a.model, dev)
    X8 = (X * 255).round().astype(np.uint8); out = []; t0 = time.time()
    for i in range(0, len(X8), a.enc_bs):
        vids = [list(v) for v in X8[i:i + a.enc_bs]]
        pv = proc(videos=vids, return_tensors="pt")["pixel_values_videos"].to(dev)
        if i == 0: print(f"  entrée V-JEPA 2 : {tuple(pv.shape)}", flush=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            z = model(pixel_values_videos=pv, skip_predictor=True).last_hidden_state   # (b, Tt*16*16, 1024)
        b, n, d = z.shape; Tt = n // 256
        z = z.float().view(b, Tt, 16, 16, d).permute(0, 1, 4, 2, 3).reshape(b * Tt, d, 16, 16)
        z = F.avg_pool2d(z, a.pool).reshape(b, Tt, d, -1).transpose(2, 3)               # (b, Tt, g*g, d)
        out.append(z.half().cpu())
        if (i // a.enc_bs) % 25 == 0: print(f"  encodage {i + len(vids)}/{len(X8)} ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(out)

def raw_tokens(X):
    """pixels bruts : (n,T,H,H,3) -> (n, T/2, npf, 2*P*P*3) — patches 8×8 des 2 frames de chaque pas."""
    n, T = X.shape[:2]; tok = patchify(X, 8)                                   # (n, T*npf, 192)
    npf = tok.shape[1] // T
    tok = tok.reshape(n, T // 2, 2, npf, -1).transpose(0, 1, 3, 2, 4).reshape(n, T // 2, npf, -1)
    return torch.from_numpy(tok.astype(np.float16))

def fit(Ztr, ytr, Zte, nout, task, dev, steps, bs=128):
    """sonde attentive sur tokens gelés. task 'cls' (2 classes, poids équilibrés) ou 'reg' (MSE)."""
    pr = AttentiveProbe(Ztr.size(-1), nout).to(dev); opt = torch.optim.AdamW(pr.parameters(), 1e-3, weight_decay=1e-2)
    if task == "cls":
        f = ytr.float().mean().clamp(0.01, 0.99); cw = torch.stack([1 / (1 - f), 1 / f]).to(dev)
    for _ in range(steps):
        bi = torch.randint(0, len(Ztr), (bs,)); out = pr(Ztr[bi].to(dev).float()); y = ytr[bi].to(dev)
        l = F.cross_entropy(out, y, weight=cw) if task == "cls" else F.mse_loss(out, y)
        opt.zero_grad(); l.backward(); opt.step()
    pr.eval()
    with torch.no_grad(): return torch.cat([pr(Zte[i:i + 512].to(dev).float()).cpu() for i in range(0, len(Zte), 512)])

def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-8))

def probes(Z, w, dev, steps, name):
    """Z:(n, Tt, k, d). Retourne positions R², impact bal-acc, log-masse R², ratio R² (avec choc)."""
    n, Tt = Z.shape[:2]; ntr = int(0.75 * n); tr, te = slice(0, ntr), slice(ntr, n)
    # par pas temporel : tokens du pas k ; cible = état à la frame 2k+1 / choc dans les frames 2k, 2k+1
    Zk = Z.flatten(0, 1)                                                       # (n*Tt, k, d)
    pos = torch.from_numpy(w["POS"][:, 1::2].reshape(n * Tt, 4))
    imp = torch.from_numpy((w["IMP"][:, 0::2] | w["IMP"][:, 1::2]).reshape(n * Tt)).long()
    ktr, kte = slice(0, ntr * Tt), slice(ntr * Tt, n * Tt)
    mu, sd = pos[ktr].mean(0), pos[ktr].std(0)
    p = fit(Zk[ktr], (pos[ktr] - mu) / sd, Zk[kte], 4, "reg", dev, steps) * sd + mu
    r_pos = np.mean([r2(p[:, j], pos[kte][:, j]) for j in range(4)])
    err_px = float((p - pos[kte]).abs().mean()) * 32
    p = fit(Zk[ktr], imp[ktr], Zk[kte], 2, "cls", dev, steps).argmax(-1); y = imp[kte]
    bacc = float(((p[y == 1] == 1).float().mean() + (p[y == 0] == 0).float().mean()) / 2)
    # par séquence : tous les tokens
    Zs = Z.flatten(1, 2)
    lm = torch.from_numpy(w["LM"]); y = torch.stack([lm[:, 0], lm[:, 1], lm[:, 0] - lm[:, 1]], 1)
    mu, sd = y[tr].mean(0), y[tr].std(0)
    p = fit(Zs[tr], (y[tr] - mu) / sd, Zs[te], 3, "reg", dev, steps) * sd + mu
    hit = torch.from_numpy(w["HIT"])[te]
    r_lm = (r2(p[:, 0], y[te, 0]) + r2(p[:, 1], y[te, 1])) / 2
    r_ratio = r2(p[hit, 2], y[te][hit, 2])
    print(f"  {name:>12s} | positions R² {r_pos:+.2f} (err ~{err_px:.1f} px) | impact {bacc:.0%} | "
          f"log-masse R² {r_lm:+.2f} | ratio masses (choc) R² {r_ratio:+.2f}", flush=True)
    return dict(pos_r2=r_pos, pos_px=err_px, imp_bacc=bacc, lm=r_lm, ratio_hit=r_ratio)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000); p.add_argument("--T", type=int, default=16)
    p.add_argument("--a_sub", type=int, default=2); p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--pool", type=int, default=4, help="moyenne spatiale des 16×16 patches (4 -> 4×4)")
    p.add_argument("--enc_bs", type=int, default=8); p.add_argument("--probe_steps", type=int, default=2000)
    p.add_argument("--save", type=str, default="", help="chemin .pt pour garder les features (étape 1)")
    p.add_argument("--skip_vj", type=int, default=0, help="1 = pixels bruts seulement (test local)")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    w = gen_world(a.n, a.T, 32, seed=a.seed, a_sub=a.a_sub)
    print(f"monde : {a.n} séquences T={a.T} | choc disque-disque {w['HIT'].mean():.0%} ({time.time() - t0:.0f}s)", flush=True)
    res = {"pixels bruts": probes(raw_tokens(w["X"]), w, dev, a.probe_steps, "pixels bruts")}
    if not a.skip_vj:
        Z = encode_vjepa2(w["X"], a, dev)
        print(f"  features V-JEPA 2 : {tuple(Z.shape)} ({time.time() - t0:.0f}s)", flush=True)
        if a.save: torch.save(dict(Z=Z, args=vars(a)), a.save); print(f"  sauvé -> {a.save}")
        res["V-JEPA 2"] = probes(Z, w, dev, a.probe_steps, "V-JEPA 2")
        v, r = res["V-JEPA 2"], res["pixels bruts"]
        print("\nVERDICT ÉTAPE 0 :")
        print(f"  voit les disques ? positions R² {v['pos_r2']:+.2f} (pixels bruts {r['pos_r2']:+.2f}) — il faut ~0.9+ pour le liage")
        print(f"  voit les chocs ?   impact {v['imp_bacc']:.0%} (pixels bruts {r['imp_bacc']:.0%} ; notre JEPA 60 %)")
        print(f"  voit la masse ?    ratio (choc) R² {v['ratio_hit']:+.2f} (notre JEPA −0.20 ; vision parfaite +0.65)")
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
