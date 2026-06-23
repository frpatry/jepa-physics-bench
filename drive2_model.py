"""
Conduite PHASE 4b : world model JEPA + actor APPRIS qui conduit la route.

Fidele LeCun :
  - perception = images ego-centrees (pixels), le modele APPREND a voir stops/passages/pietons.
  - world model = JEPA (SIGReg) entraine en SSL sur des fenetres d'images.
  - actor = policy apprise depuis la repr JEPA GELEE, en IMITANT l'oracle (demonstrations).
  - eval = BOUCLE FERMEE : l'actor conduit, on compte collisions + respect des stops + anime.

  python drive2_model.py --steps 2000 --n_train 600
"""
import argparse, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import drive2
from drive_model import Enc, sigreg

G = 16; W = 10                                                # image GxG ; fenetre temporelle (memoire stop)
XLO, XHI, YLO, YHI = -3.0, 30.0, -4.0, 4.0

def render_frame(car_x, events, peds_t):
    """Image ego-centree (G*G,) : route + stops (marque au-dessus) + passages (bande) + pietons (blob)."""
    img = np.zeros((G, G), np.float32)
    rows_road = [r for r in range(G) if abs(YHI - (r + 0.5) / G * (YHI - YLO)) < 2.5]
    for r in rows_road: img[r, :] = 0.2
    for i, e in enumerate(events):
        xr = e["x"] - car_x
        if xr < XLO or xr > XHI: continue
        c = int((xr - XLO) / (XHI - XLO) * G); c = min(G - 1, max(0, c))
        if e["type"] == "stop":
            img[0:2, max(0, c - 1):c + 1] = 1.0                # marque vive AU-DESSUS de la route
        else:
            for r in rows_road: img[r, c] = 0.6               # bande passage SUR la route
            py = peds_t.get(i)
            if py is not None and YLO < py < YHI:
                pr = int((YHI - py) / (YHI - YLO) * G)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = pr + dr, c + dc
                        if 0 <= rr < G and 0 <= cc < G: img[rr, cc] = 1.0
    return img.reshape(-1)

def make_routes(n, a, seed):
    rng = np.random.default_rng(seed); routes = []
    for _ in range(n):
        r = drive2.gen_route(rng, argparse.Namespace(T=a.T, dt=a.dt, cruise=a.cruise, a_acc=a.a_acc,
                             a_brake=a.a_brake, ped_speed=1.4, p_cross=a.p_cross, wait=6))
        routes.append(r)
    return routes

def route_obs(r):
    T = r["T"]; ev = r["events"]
    pe = {int(i): r["peds"][i] for i in r["peds"]}
    obs = np.stack([render_frame(r["ego_x"][t], ev, {i: pe[i][t] for i in pe}) for t in range(T)])
    return obs                                                 # (T, G*G)

class Policy(nn.Module):
    def __init__(s, d, w):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d + w, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, rep, vwin): return s.net(torch.cat([rep, vwin], -1)).squeeze(-1)   # vwin=historique vitesse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=135); p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--cruise", type=float, default=8.0); p.add_argument("--a_acc", type=float, default=2.5)
    p.add_argument("--a_brake", type=float, default=4.0); p.add_argument("--p_cross", type=float, default=0.6)
    p.add_argument("--d_model", type=int, default=128); p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4); p.add_argument("--reg_w", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--pol_steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=128); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--n_train", type=int, default=600); p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--dump_n", type=int, default=1)
    return p.parse_args()

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    obs_dim = G * G
    tr = make_routes(a.n_train, a, 1000 + a.seed); te = make_routes(a.n_test, a, 99999)
    TR = torch.tensor(np.stack([route_obs(r) for r in tr]))    # (n,T,G*G)
    Vtr = torch.tensor(np.array([r["ego_v"] for r in tr]), dtype=torch.float32)
    print(f"device={dev}  routes train={a.n_train} test={a.n_test}  G={G} W={W} T={a.T}", flush=True)

    # ---- 1) WORLD MODEL : JEPA SSL sur fenetres de W frames ----
    torch.manual_seed(a.seed)
    enc = Enc(obs_dim, a.d_model, W, a.n_layer, a.n_head).to(dev)
    pred = nn.Sequential(nn.Linear(a.d_model, a.d_model), nn.GELU(), nn.Linear(a.d_model, a.d_model)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=a.lr)
    n, T = TR.shape[0], TR.shape[1]
    for st in range(a.steps):
        ri = torch.randint(0, n, (a.bs,)); si = torch.randint(0, T - W, (a.bs,))
        win = torch.stack([TR[ri[b], si[b]:si[b] + W] for b in range(a.bs)]).to(dev)   # (bs,W,obs)
        m = torch.rand(a.bs, W, device=dev) < a.mask_ratio; m[:, 0] = False
        z = enc(win, None); p = pred(enc(win, m))
        loss = (F.smooth_l1_loss(p[m], z[m]) if m.any() else (p - z).pow(2).mean()) + a.reg_w * sigreg(z.reshape(-1, a.d_model))
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0: print(f"  [WM] step {st} loss {loss.item():.3f}", flush=True)

    # ---- 2) POLICY : imite l'accel de l'oracle depuis la repr GELEE ----
    for pr_ in enc.parameters(): pr_.requires_grad = False
    pol = Policy(a.d_model, W).to(dev); po = torch.optim.Adam(pol.parameters(), 1e-3)
    acc = (Vtr[:, 1:] - Vtr[:, :-1]) / a.dt                    # accel cible (n,T-1)
    @torch.no_grad()
    def encwin(win): return enc(win.to(dev)).mean(1)           # (B,d)
    for st in range(a.pol_steps):
        ri = torch.randint(0, n, (a.bs,)); ti = torch.randint(W, T - 1, (a.bs,))
        win = torch.stack([TR[ri[b], ti[b] - W + 1:ti[b] + 1] for b in range(a.bs)])
        vwin = torch.stack([Vtr[ri[b], ti[b] - W + 1:ti[b] + 1] for b in range(a.bs)]).to(dev)  # historique vitesse
        rep = encwin(win); tgt = acc[ri, ti].to(dev)
        po.zero_grad(); F.mse_loss(pol(rep, vwin), tgt).backward(); po.step()
    print("  [policy] entrainee (imitation oracle)", flush=True)

    # ---- 3) BOUCLE FERMEE : l'actor conduit les routes test ----
    enc.eval(); pol.eval()
    coll = 0; ranstop = 0; nstop = 0; drives = []
    for r in te:
        ev = r["events"]; pe = {int(i): r["peds"][i] for i in r["peds"]}
        x = 0.0; v = 0.0; buf = [render_frame(0.0, ev, {i: pe[i][0] for i in pe})] * W
        vbuf = [0.0] * W
        xs = []; vs = []; minv_near = {i: 9.9 for i, e in enumerate(ev) if e["type"] == "stop"}
        hit = False
        for t in range(a.T):
            xs.append(round(x, 2)); vs.append(round(v, 2))
            buf = buf[1:] + [render_frame(x, ev, {i: pe[i][t] for i in pe})]
            vbuf = vbuf[1:] + [v]
            win = torch.tensor(np.stack(buf), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                ap = pol(enc(win.to(dev)).mean(1), torch.tensor([vbuf], device=dev)).item()
            ap = max(-a.a_brake, min(a.a_acc, ap))
            v = max(0.0, v + ap * a.dt); x += v * a.dt
            for i, e in enumerate(ev):                          # collision (passage) + respect stop
                if e["type"] == "cw" and abs(x - e["x"]) < 2.0 and abs(pe[i][t]) < 1.0: hit = True
                if e["type"] == "stop" and abs(x - (e["x"] - 1.0)) < 3.0: minv_near[i] = min(minv_near[i], v)
        coll += int(hit)
        for i in minv_near:
            nstop += 1; ranstop += int(minv_near[i] > 1.0)      # n'a pas vraiment ralenti au stop
        if len(drives) < a.dump_n:
            drives.append({"ego_x": xs, "ego_v": vs, "events": ev,
                           "p1": pe[1] if 1 in pe else [9] * a.T, "p3": pe[3] if 3 in pe else [9] * a.T})
    print(f"\n=== ACTOR APPRIS en boucle fermee ({a.n_test} routes) ===", flush=True)
    print(f"  collisions piétons : {coll/a.n_test:.0%}", flush=True)
    print(f"  stops non respectés : {ranstop/max(1,nstop):.0%}", flush=True)
    print(f"  vitesse finale moyenne : {np.mean([r['ego_v'][-1] for r in drives]) if drives else 0:.1f} m/s", flush=True)
    print("DRIVE2B_DUMP " + json.dumps(drives), flush=True)

if __name__ == "__main__":
    main()
