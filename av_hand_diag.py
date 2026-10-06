"""
DIAG — pourquoi l'encodeur bébé JETTE-t-il la main ? (lecteur de positions = instrument, comme l'examen)
Pour chaque instantané : R² disques / MAIN dans le monde avec main, selon le comportement de la main :
  script (viser-pousser à fond) | babillage | main IMMOBILE (rien ne bouge) ; + référence PIXELS bruts.
Lecture : main lisible quand immobile mais pas quand elle bouge -> l'encodeur jette ce qui est IMPRÉVISIBLE
(remède : donner la commande motrice) ; illisible même immobile -> objet NOUVEAU / trop petit (remède :
saillance, plus d'exposition).

  python av_hand_diag.py --ckpts /content/drive/MyDrive/jepa_runs/av_dev_v4_sep_40k.pt,/content/drive/MyDrive/jepa_runs/av_dev_v5_hand_50k.pt
"""
import argparse, copy, time
import numpy as np, torch
from av_jepa import gen_world, NB
from av_act import gen_world_act
from av_dev import DevJEPA
from av_dev_long import to_tokens, stereo, loc_r2, T, H

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=str, required=True); p.add_argument("--sep_enc", type=int, default=1)
    p.add_argument("--hum", type=float, default=0.15); p.add_argument("--n", type=int, default=1500); p.add_argument("--steps", type=int, default=1500)
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    P = 4; nP = H // P; nv = T * nP * nP; da = 2 * 2 * NB; W = max(P * P * 3, da)
    w0 = gen_world(2000, T, H, seed=0, a_sub=2, hum=a.hum); A0 = stereo(torch.from_numpy(w0["A"])).reshape(2000, T, -1)
    st = dict(amu=A0.mean((0, 1)).to(dev), asd=(A0.std((0, 1)) + 1e-4).to(dev)); del w0
    probes = {}
    for name, b in [("script", 0), ("babillage", 1), ("main immobile", 2)]:
        w = gen_world_act(a.n, T, H, seed=1001, a_sub=2, hum=a.hum, fric=0.02, babble=b)
        X, A = torch.from_numpy((w["X"] * 255).round().astype(np.uint8)), torch.from_numpy(w["A"])
        P2 = w["POS"]; o_ = np.argsort(P2[..., 0], axis=-1)
        y = np.concatenate([np.take_along_axis(P2, o_[..., None], axis=2).reshape(a.n, T, 4), w["HAND"]], -1)
        tok = torch.cat([to_tokens(X[i:i + 250].to(dev), A[i:i + 250].to(dev), P, st).half().cpu() for i in range(0, a.n, 250)])
        mv = np.linalg.norm(np.diff(w["HAND"], axis=1), axis=-1).mean() * 32
        probes[name] = dict(htok=tok, hpos=torch.from_numpy(y).float()); print(f"monde {name:>14s} : main bouge {mv:.2f} px/frame ({time.time() - t0:.0f}s)", flush=True)
    G = [slice(0, 4), slice(4, 6)]
    print(f"\n{'encodeur':>28s} | " + " | ".join(f"{n_:>26s}" for n_ in probes)); print(" " * 28 + " | " + " | ".join(f"{'disques R²':>12s} {'MAIN R²':>13s}" for _ in probes))
    rows = [("PIXELS bruts (référence)", None)] + [(c.split("/")[-1], c) for c in a.ckpts.split(",")]
    for name, c in rows:
        if c is None: m = None
        else:
            m0 = DevJEPA(W, da, nv, T, 192, 6, 6, 3).to(dev); enc = copy.deepcopy(m0.enc)
            enc.load_state_dict(torch.load(c, map_location=dev, weights_only=False)["tgt"]); enc.eval(); enc.sep = bool(a.sep_enc)
            m = type("W", (), {"enc": enc})()
        res = []
        for pr in probes.values():              # m None : patches bruts (represent renvoie les tokens tels quels)
            res.append(loc_r2(m, pr, nv, dev, a.steps, tok_key="htok", y_key="hpos", groups=G))
        print(f"{name:>28s} | " + " | ".join(f"{d:+12.2f} {h:+13.2f}" for d, h in res) + f"  ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
