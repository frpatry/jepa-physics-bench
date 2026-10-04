"""
RUN LONG DÉVELOPPEMENTAL (nuit, A100) — le bébé entend d'abord, voit flou, puis net, avec BEAUCOUP plus
d'expérience (flux continu de données neuves) et une OREILLE plus réaliste.

Différences avec av_dev.py :
  - flux INFINI de séquences neuves (générées en parallèle sur CPU), jamais deux fois la même ;
  - stéréo EXPLICITE (comme le tronc cérébral) : par bande, MOYENNE (L+R)/2 et DIFFÉRENCE (L−R) au lieu
    de deux spectres séparés ;
  - vision plus FINE (patches 4×4 sur 32 px = 64 zones par image au lieu de 16) ;
  - étapes qui ne s'enchaînent que quand la précédente est MAÎTRISÉE (plateau de la perte), avec un max :
      A son seul -> B image très floue (σ=3) -> C flou qui diminue jusqu'à net -> D net ;
  - EXAMENS réguliers (lecteur gelé) : LOCALISATION (stéréo correcte vs inversée, 1000 étiq. ; plafond
    vision parfaite 96 %) et ANTICIPATION (frames 0-7 -> choc 8-9, 2000 étiq.) ;
  - points de REPRISE réguliers (/content/av_dev_long.pt) : relancer la même commande reprend le run.

  python av_dev_long.py --total 100000          # Colab A100
"""
import argparse, copy, math, os, time
import multiprocessing as mp
import numpy as np, torch, torch.nn.functional as F
from av_jepa import gen_world, NB
from av_dev import blur, masks, DevJEPA, fit_reader, represent, bacc
from vjepa import _idx, _gather

T, H = 16, 32

def stereo(A):
    """A (..., T, 2*a_sub, NB) lignes (sous-fenêtre, canal) -> (..., T, 2*a_sub, NB) [moyenne, différence]."""
    L, R = A[..., 0::2, :], A[..., 1::2, :]
    out = torch.empty_like(A); out[..., 0::2, :] = (L + R) / 2; out[..., 1::2, :] = L - R
    return out

def to_tokens(X, A, P, st, sigma=0.0):
    """-> (B, T*npf + T, W) : vision (patches) puis audio stéréo-explicite, largeur commune W."""
    X = blur(X.float() / 255.0 if X.dtype == torch.uint8 else X.float(), sigma)
    B = X.size(0); nP = H // P
    v = X.reshape(B, T, nP, P, nP, P, 3).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T * nP * nP, P * P * 3)
    a = (stereo(A.float()).reshape(B, T, -1) - st["amu"]) / st["asd"]
    W = max(v.size(-1), a.size(-1)); tok = torch.zeros(B, v.size(1) + T, W, device=v.device)
    tok[:, :v.size(1), :v.size(-1)] = v; tok[:, v.size(1):, :a.size(-1)] = a
    return tok

def _batch(job):
    seed, n = job
    w = gen_world(n, T, H, seed=seed, a_sub=2)
    return (w["X"] * 255).round().astype(np.uint8), w["A"].astype(np.float16)

def stream(start, bs, workers):
    jobs = ((10_000_000 + i, bs) for i in range(start, 10 ** 9))
    with mp.get_context("fork").Pool(workers) as pool:
        for X, A in pool.imap(_batch, jobs, chunksize=4):
            yield torch.from_numpy(X), torch.from_numpy(A)

def build_probe(a, dev, st):
    w = gen_world(a.n_probe, T, H, seed=1000, a_sub=2)
    X, A = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)), torch.from_numpy(w["A"])
    lab = (torch.arange(a.n_probe) % 2).long(); Asw = A.clone(); Asw[lab == 1] = A[lab == 1][:, :, [1, 0, 3, 2]]
    mk = lambda A_: torch.cat([to_tokens(X[i:i + 250].to(dev), A_[i:i + 250].to(dev), a.P, st).half().cpu()
                               for i in range(0, a.n_probe, 250)])
    return dict(sw=mk(Asw), tok=mk(A), lab=lab, antic=torch.from_numpy((w["IMP"][:, 8] | w["IMP"][:, 9]).astype(np.int64)))

def exam(m, probe, a, dev, nv, tag):
    N = nv + T; npf = nv // T; frame = np.concatenate([np.arange(nv) // npf, np.arange(T)])
    nte = min(1000, a.n_probe // 3); te = slice(a.n_probe - nte, a.n_probe); res = {}
    L1, L2 = min(1000, a.n_probe - nte), min(2000, a.n_probe - nte)
    R = represent(m, probe["sw"], np.ones(N, bool), dev)
    res["localisation"] = bacc(fit_reader(R[:L1], probe["lab"][:L1], R[te], "bin", dev, a.read_steps).argmax(-1), probe["lab"][te])
    R = represent(m, probe["tok"], frame <= 7, dev)
    res["anticipation"] = bacc(fit_reader(R[:L2], probe["antic"][:L2], R[te], "bin", dev, a.read_steps).argmax(-1), probe["antic"][te])
    print(f"  EXAMEN {tag:>18s} | localisation {res['localisation']:.0%} (plafond 96 %) | anticipation {res['anticipation']:.0%}", flush=True)
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=100000); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--P", type=int, default=4); p.add_argument("--d", type=int, default=192)
    p.add_argument("--nl", type=int, default=6); p.add_argument("--nh", type=int, default=6); p.add_argument("--pred_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--ema", type=float, default=0.996); p.add_argument("--n_masks", type=int, default=3)
    p.add_argument("--sig_max", type=float, default=3.0); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--min_stage", type=int, default=8000); p.add_argument("--max_stage", type=int, default=25000)
    p.add_argument("--blur_down", type=int, default=25000, help="durée de l'étape C (σ 3 -> 0)")
    p.add_argument("--exam_every", type=int, default=10000); p.add_argument("--ckpt_every", type=int, default=2500)
    p.add_argument("--n_probe", type=int, default=3000); p.add_argument("--read_steps", type=int, default=1500)
    p.add_argument("--ckpt", type=str, default="/content/av_dev_long.pt"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True; rng = np.random.default_rng(a.seed)
    nP = H // a.P; nv = T * nP * nP; da = 2 * 2 * NB
    w0 = gen_world(2000, T, H, seed=a.seed, a_sub=2); A0 = stereo(torch.from_numpy(w0["A"])).reshape(2000, T, -1)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev)); del w0
    W = max(a.P * a.P * 3, da)
    print(f"GPU {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'} | {T} frames × {nP * nP} patches {a.P}×{a.P} + {T} audio "
          f"stéréo explicite | tokens {nv + T} × {W}", flush=True)
    probe = build_probe(a, dev, st)
    torch.manual_seed(a.seed); m = DevJEPA(W, da, nv, T, a.d, a.nl, a.nh, a.pred_layers).to(dev)
    tgt = copy.deepcopy(m.enc).eval()
    for p_ in tgt.parameters(): p_.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05)
    state = dict(it=0, stage="A", stage_start=0, hist=[], exams=[])
    if os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        m.load_state_dict(ck["m"]); tgt.load_state_dict(ck["tgt"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
        print(f"REPRISE au pas {state['it']} (étape {state['stage']})", flush=True)
    else:
        class Wrap(torch.nn.Module):
            def __init__(s, enc): super().__init__(); s.enc, s.nv, s.T = enc, nv, T
        state["exams"].append(("init", exam(Wrap(tgt), probe, a, dev, nv, "init (aléatoire)")))
    data = stream(state["it"], a.bs, a.workers); ma = None
    def sigma_of(stage, k):
        return {"A": 0.0, "B": a.sig_max, "D": 0.0}.get(stage, a.sig_max * max(0.0, 1 - k / a.blur_down))
    while state["it"] < a.total:
        it = state["it"] = state["it"] + 1; k = it - state["stage_start"]; stage = state["stage"]
        sig = sigma_of(stage, k)
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, it / 3000)
        X, A = next(data); o = to_tokens(X.to(dev, non_blocking=True), A.to(dev, non_blocking=True), a.P, st, sig)
        present, pairs = masks("a" if stage == "A" else "va", a.bs, T, nP, nv, rng, a.n_masks)
        B, N, _ = o.shape; pidx = _idx(torch.from_numpy(np.broadcast_to(present, (B, N)).copy()).to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            with torch.no_grad(): z = F.layer_norm(tgt(_gather(o, pidx), pidx).float(), (a.d,))
            zf = torch.zeros(B, N, a.d, device=dev).scatter(1, pidx.unsqueeze(-1).expand(-1, -1, a.d), z)
            loss = 0.0
            for c_, g_ in pairs:
                cidx, tidx = _idx(torch.from_numpy(c_).to(dev)), _idx(torch.from_numpy(g_).to(dev))
                loss = loss + F.l1_loss(m.pred(m.enc(_gather(o, cidx), cidx), cidx, tidx).float(), _gather(zf, tidx))
            loss = loss / len(pairs)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        mom = 1 - (1 - a.ema) * (math.cos(math.pi * it / a.total) + 1) / 2
        with torch.no_grad():
            for pt, pc in zip(tgt.parameters(), m.enc.parameters()): pt.mul_(mom).add_(pc.detach(), alpha=1 - mom)
        ma = loss.item() if ma is None else 0.995 * ma + 0.005 * loss.item()
        if it % 500 == 0:
            state["hist"].append((it, stage, ma))
            print(f"  pas {it:6d} | étape {stage} (depuis {k}) | flou σ={sig:.1f} | perte {ma:.4f} | {time.time() - t0:.0f}s", flush=True)
            # passage d'étape : A et B quand la perte plafonne (gain < 2 % sur 3000 pas) ou au max ; C après la descente du flou
            prev = [h[2] for h in state["hist"] if h[1] == stage and h[0] <= it - 3000]
            plateau = k >= a.min_stage and prev and (prev[-1] - ma) / max(prev[-1], 1e-6) < 0.02
            nxt = None
            if stage in ("A", "B") and (plateau or k >= a.max_stage): nxt = {"A": "B", "B": "C"}[stage]
            if stage == "C" and k >= a.blur_down: nxt = "D"
            if nxt:
                print(f"  >>> ÉTAPE {stage} MAÎTRISÉE au pas {it} ({'plateau' if plateau else 'durée max'}) -> étape {nxt}", flush=True)
                state["stage"], state["stage_start"] = nxt, it; ma = None
        if it % a.exam_every == 0:
            m.eval(); r = exam(type("W", (), {"enc": tgt, "nv": nv, "T": T})(), probe, a, dev, nv, f"pas {it} étape {stage}")
            state["exams"].append((f"pas {it} ({stage})", r)); m.train()
        if it % a.ckpt_every == 0 or it == a.total:
            torch.save(dict(m=m.state_dict(), tgt=tgt.state_dict(), opt=opt.state_dict(), state=state), a.ckpt + ".tmp")
            os.replace(a.ckpt + ".tmp", a.ckpt)
    print("\n========== EXAMENS AU FIL DU DÉVELOPPEMENT (hasard 50 % ; plafond localisation 96 %) ==========")
    for tag, r in state["exams"]: print(f"  {tag:>20s} | localisation {r['localisation']:.0%} | anticipation {r['anticipation']:.0%}")
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
