"""
Text-JEPA pour continual learning.

Tokens en ENTRÉE (interface), objectif JEPA (prédiction latente de zones
masquées), PAS de cross-entropy de token :

    tokens -> on MASQUE une partie (cibles)
    encodeur-contexte (bidirectionnel) voit le contexte + mask-tokens
    prédicteur prédit le LATENT des positions masquées
    cible = représentation d'un encodeur-cible EMA (momentum de l'online sur
            l'entrée PROPRE), normalisée, stop-gradient
    perte = smooth-L1 en espace latent

IMPORTANT (correctif) : la cible EMA rend la tâche APPRENABLE (vraie amplitude
de perte), contrairement à une cible aléatoire gelée (quasi non-apprenable ->
oubli "1.0x" artefactuel). Garde anti-collapse : terme de variance (VICReg) sur
les représentations + monitoring du std (un std -> 0 = collapse).

Hypothèse testée (INCHANGÉE) : routage STABLE adressé par le contexte (route-emb
GELÉE -> projection GELÉE -> top-K sur le FFN) protège-t-il de l'oubli ?
+ Diagnostics : sparsité, chevauchement des activations APPRISES (séparation
naturelle des domaines), et SONDE DE DOMAINE (les représentations capturent-elles
de l'info spécifique au domaine ? sinon "pas d'oubli" est ambigu).

Modes : dense / routed_hard / learned_topk / routed_protect.
  python cl_jepa_text.py                 # vrai texte (HuggingFace)
  python cl_jepa_text.py --synthetic     # tokens aléatoires (smoke test)
"""
import argparse, json, os, copy, torch, torch.nn as nn, torch.nn.functional as F
from cl_scale import load_domains            # réutilise les domaines (specs HF corrigées)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--modes", nargs="+",
                   default=["dense", "routed_hard", "learned_topk", "routed_protect"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--route_dim", type=int, default=32)
    p.add_argument("--route_win", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--domains", type=int, default=3)
    p.add_argument("--target", choices=["ema", "frozen"], default="ema")
    p.add_argument("--ema_decay", type=float, default=0.996)
    p.add_argument("--var_w", type=float, default=1.0, help="poids anti-collapse (variance)")
    p.add_argument("--syn_diff", type=float, default=0.2)
    p.add_argument("--out", type=str, default="runs_jtext")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

# --------------------------- routage (INCHANGÉ) -----------------------------
class RoutedFFN(nn.Module):
    def __init__(self, d, d_ff, k, mode, route_dim):
        super().__init__()
        self.mode, self.k = mode, k
        self.fc1 = nn.Linear(d, d_ff); self.fc2 = nn.Linear(d_ff, d)
        self.capture = False; self._cap = None
        if mode in ("routed_hard", "routed_protect"):
            self.register_buffer("Wr", torch.randn(route_dim, d_ff))
    def forward(self, x, route_ctx):
        h = F.relu(self.fc1(x))
        if self.capture: self._cap = h.detach()
        if self.mode in ("routed_hard", "routed_protect"):
            sc = route_ctx @ self.Wr
            thr = sc.topk(self.k, -1).values[..., -1:]
            h = h * (sc >= thr).float()
        elif self.mode == "learned_topk":
            thr = h.topk(self.k, -1).values[..., -1:].detach()
            h = torch.where(h >= thr, h, torch.zeros_like(h))
        return self.fc2(h)
    def routed_mask(self, route_ctx):
        sc = route_ctx @ self.Wr
        thr = sc.topk(self.k, -1).values[..., -1:]
        return (sc >= thr)

class Block(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.ln1 = nn.LayerNorm(a.d_model); self.ln2 = nn.LayerNorm(a.d_model)
        self.attn = nn.MultiheadAttention(a.d_model, a.n_head, batch_first=True)
        self.ffn = RoutedFFN(a.d_model, a.d_ff, a.k, a.mode, a.route_dim)
    def forward(self, x, route_ctx):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))   # bidirectionnel
        x = x + h
        return x + self.ffn(self.ln2(x), route_ctx)

class Encoder(nn.Module):
    """tok + pos + blocs + ln. Réutilisé pour l'online ET la cible EMA."""
    def __init__(self, a, V):
        super().__init__()
        self.tok = nn.Embedding(V, a.d_model)
        self.pos = nn.Embedding(a.seq, a.d_model)
        self.blocks = nn.ModuleList([Block(a) for _ in range(a.n_layer)])
        self.ln = nn.LayerNorm(a.d_model)
    def forward(self, ids, rc, mask=None, mask_token=None):
        emb = self.tok(ids)
        if mask is not None:
            emb = torch.where(mask.unsqueeze(-1), mask_token, emb)
        x = emb + self.pos(torch.arange(ids.size(1), device=ids.device))
        for b in self.blocks: x = b(x, rc)
        return self.ln(x)

# --------------------------- text-JEPA --------------------------------------
class TextJEPA(nn.Module):
    def __init__(self, a, V):
        super().__init__()
        d = a.d_model
        self.enc = Encoder(a, V)                                  # online
        self.predictor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.route_emb = nn.Embedding(V, a.route_dim); self.route_emb.weight.requires_grad_(False)
        self.win = a.route_win; self.var_w = a.var_w
        self.target_mode = a.target
        if a.target == "ema":
            self.tgt = Encoder(a, V)
            self.tgt.load_state_dict(self.enc.state_dict())       # même init que l'online
            for p in self.tgt.parameters(): p.requires_grad_(False)
            self.ema_decay = a.ema_decay
        else:                                                     # cible figée (option héritée)
            self.target = nn.Embedding(V, d); self.target.weight.requires_grad_(False)
    def route_context(self, ids):
        e = self.route_emb(ids)
        e = e.transpose(1, 2); e = F.pad(e, (self.win - 1, 0))
        e = F.avg_pool1d(e, self.win, 1)
        return e.transpose(1, 2)
    def forward(self, ids, mask):
        rc = self.route_context(ids.masked_fill(mask, 0))         # cibles cachées (pas de fuite)
        h = self.enc(ids, rc, mask, self.mask_token)
        pred = self.predictor(h)
        if self.target_mode == "ema":
            with torch.no_grad():
                z = self.tgt(ids, self.route_context(ids))        # entrée PROPRE
            raw_std = z.std().item()
            zt = F.layer_norm(z, (z.size(-1),))                   # normalise (stabilité d'échelle)
        else:
            with torch.no_grad(): z = self.target(ids)
            raw_std = z.std().item(); zt = z
        pred_loss = F.smooth_l1_loss(pred[mask], zt[mask].detach())
        # anti-collapse : variance par dim des représentations online aux cibles
        hm = h[mask]
        std = hm.std(0)
        var_loss = F.relu(1.0 - std).mean()
        total = pred_loss + self.var_w * var_loss
        return total, pred_loss.item(), std.mean().item()
    @torch.no_grad()
    def ema_update(self):
        if self.target_mode != "ema": return
        for po, pt in zip(self.enc.parameters(), self.tgt.parameters()):
            pt.mul_(self.ema_decay).add_(po.detach(), alpha=1 - self.ema_decay)
    @torch.no_grad()
    def features(self, ids):                                      # rep propre (sonde, diag)
        return self.enc(ids, self.route_context(ids))
    @torch.no_grad()
    def activations(self, ids, layer):
        rc = self.route_context(ids)
        self.enc.blocks[layer].ffn.capture = True
        self.enc(ids, rc)
        self.enc.blocks[layer].ffn.capture = False
        h = self.enc.blocks[layer].ffn._cap; self.enc.blocks[layer].ffn._cap = None
        return h

# --------------------------- masques / éval ---------------------------------
def make_mask(B, T, ratio, gen):
    r = torch.rand(B, T, generator=gen)
    nmask = max(1, int(ratio * T))
    idx = r.topk(nmask, dim=1).indices
    m = torch.zeros(B, T, dtype=torch.bool); m.scatter_(1, idx, True)
    return m

def get_batch(data, bs, dev, gen=None):
    idx = torch.randint(0, data.size(0), (bs,), generator=gen)
    return data[idx].to(dev)

def jepa_eval(model, data, a, dev, seed=1234):
    """Métrique d'oubli = perte de PRÉDICTION pure (sans le terme de variance)."""
    gen = torch.Generator().manual_seed(seed)
    model.eval(); tot = 0.0
    with torch.no_grad():
        for _ in range(a.eval_iters):
            x = get_batch(data, 32, dev, gen)
            mask = make_mask(x.size(0), x.size(1), a.mask_ratio, gen).to(dev)
            _, pred_loss, _ = model(x, mask)
            tot += pred_loss
    return tot / a.eval_iters

def train_domain(model, data, a, dev, tag=""):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    for step in range(a.steps):
        x = get_batch(data, a.bs, dev)
        mask = make_mask(x.size(0), x.size(1), a.mask_ratio, None).to(dev)
        total, pred_loss, std = model(x, mask)
        opt.zero_grad(); total.backward(); opt.step()
        model.ema_update()
        if a.log_every and (step % a.log_every == 0 or step == a.steps - 1):
            print(f"    {tag} step {step:5d}/{a.steps}  pred {pred_loss:.4f}  rep_std {std:.3f}", flush=True)

# --------------------------- diagnostics ------------------------------------
def routing_overlap(model, dA, dB, dev, layer=0):
    ffn = model.enc.blocks[layer].ffn
    if not hasattr(ffn, "Wr"): return None
    with torch.no_grad():
        def masks(d):
            x = d[:256].to(dev); rc = model.route_context(x)
            return ffn.routed_mask(rc).reshape(-1, ffn.fc1.out_features)
        ma, mb = masks(dA), masks(dB); n = min(len(ma), len(mb))
        inter = (ma[:n] & mb[:n]).sum(-1).float(); union = (ma[:n] | mb[:n]).sum(-1).float()
        return (inter / union).mean().item()

def activation_overlap(model, dA, dB, dev, layer=0):
    with torch.no_grad():
        def top(d):
            h = model.activations(d[:256].to(dev), layer).reshape(-1, model.enc.blocks[layer].ffn.fc1.out_features)
            thr = h.topk(model.enc.blocks[layer].ffn.k, -1).values[..., -1:]
            return h >= thr
        ma, mb = top(dA), top(dB); n = min(len(ma), len(mb))
        inter = (ma[:n] & mb[:n]).sum(-1).float(); union = (ma[:n] | mb[:n]).sum(-1).float()
        return (inter / union).mean().item()

def activation_sparsity(model, data, dev, layer=0):
    with torch.no_grad():
        return (model.activations(data[:256].to(dev), layer) > 0).float().mean().item()

def mean_pairwise(fn, model, domains, dev):
    vals = [fn(model, domains[i], domains[j], dev)
            for i in range(len(domains)) for j in range(i + 1, len(domains))]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def domain_probe(model, domains, dev, steps=300, n=400):
    """Classifieur linéaire (encodeur GELÉ) pour prédire le domaine.
    Accuracy >> 1/n_domaines -> les représentations portent de l'info de domaine."""
    Xs, ys = [], []
    for di, d in enumerate(domains):
        f = model.features(d[:n].to(dev)).mean(1)
        Xs.append(f); ys.append(torch.full((f.size(0),), di, device=dev, dtype=torch.long))
    X = torch.cat(Xs); y = torch.cat(ys)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(X.size(0), generator=g).to(dev)
    X, y = X[perm], y[perm]; ntr = int(0.8 * X.size(0))
    clf = nn.Linear(X.size(1), len(domains)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = F.cross_entropy(clf(X[:ntr]), y[:ntr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (clf(X[ntr:]).argmax(1) == y[ntr:]).float().mean().item()

# --------------------------- tracé ------------------------------------------
def plot_curves(results, names, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = len(names)
    fig, axes = plt.subplots(1, D, figsize=(5 * D, 4), squeeze=False)
    for j in range(D):
        ax = axes[0][j]
        for mode, r in results.items():
            sp = r["stage_loss"]
            xs = [s for s in range(D) if str(j) in sp.get(str(s), {})]
            ys = [sp[str(s)][str(j)] for s in xs]
            if xs: ax.plot(xs, ys, marker="o", label=mode)
        ax.set_title(f"domaine {j} ({names[j]})")
        ax.set_xlabel("après entraînement du domaine #"); ax.set_ylabel("perte JEPA (latent)")
        ax.set_xticks(range(D)); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Text-JEPA : perte par domaine au fil du CL (hausse = oubli)", y=1.02)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=120)
    print(f"[plot] -> {out_png}")

# --------------------------- main -------------------------------------------
def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    domains, V, names = load_domains(a, tok_vocab=512)
    print(f"device={dev}  vocab={V}  domaines={len(domains)} {names}  "
          f"shapes={[tuple(d.shape) for d in domains]}  cible={a.target}\n", flush=True)

    results = {}
    for mode in a.modes:
        print(f"\n========== mode = {mode} ==========", flush=True)
        a.mode = mode
        model = TextJEPA(a, V).to(dev)
        stage_loss, first_loss, final_loss, init_loss = {}, {}, {}, {}
        seen = []
        for di, d in enumerate(domains):
            if mode == "routed_protect" and di == 1:
                for p in (*model.enc.tok.parameters(), *model.enc.pos.parameters(),
                          *model.predictor.parameters()):
                    p.requires_grad_(False)
                model.mask_token.requires_grad_(False)
                print("  [routed_protect] gel tok-emb + pos + prédicteur après domaine 0", flush=True)
            # perte AVANT entraînement de ce domaine (pour normaliser le gain)
            init_loss[di] = jepa_eval(model, d.to(dev), a, dev)
            print(f"  -- entraînement domaine {di} ({names[di]}) | perte initiale {init_loss[di]:.4f} --", flush=True)
            train_domain(model, d.to(dev), a, dev, tag=f"[{mode} d{di}]")
            seen.append(di)
            stage_loss[str(di)] = {}
            for j in seen:
                L = jepa_eval(model, domains[j], a, dev)
                stage_loss[str(di)][str(j)] = L
                if j == di: first_loss[j] = L
                final_loss[j] = L
            line = "  ".join(f"d{j}={stage_loss[str(di)][str(j)]:.4f}" for j in seen)
            print(f"  >> après domaine {di} | perte: {line}", flush=True)
        # oubli ratio (final/first) ET oubli normalisé par le gain appris
        forg = [final_loss[j] / first_loss[j] for j in range(len(domains) - 1)]
        forg_mean = sum(forg) / len(forg) if forg else None
        # fraction du gain effacée = (final-first)/(init-first), robuste à la plasticité
        frac = []
        for j in range(len(domains) - 1):
            gain = init_loss[j] - first_loss[j]
            frac.append((final_loss[j] - first_loss[j]) / gain if gain > 1e-9 else float("nan"))
        frac_mean = sum(f for f in frac if f == f) / max(1, sum(1 for f in frac if f == f)) if frac else None
        ov_routed = mean_pairwise(lambda m, x, y, dv: routing_overlap(m, x, y, dv, 0), model, domains, dev)
        ov_act = mean_pairwise(lambda m, x, y, dv: activation_overlap(m, x, y, dv, 0), model, domains, dev)
        spars = sum(activation_sparsity(model, d, dev) for d in domains) / len(domains)
        probe = domain_probe(model, domains, dev)
        print(f"  oubli {forg_mean:.2f}x | gain effacé {frac_mean:.0%} | "
              f"overlap routage(gelé)={ov_routed if ov_routed is None else f'{ov_routed:.1%}'} | "
              f"overlap activ={ov_act:.1%} | sparsité={spars:.1%} | sonde domaine={probe:.1%}", flush=True)
        results[mode] = {
            "stage_loss": stage_loss,
            "init_loss": {str(j): init_loss[j] for j in init_loss},
            "first_loss": {str(j): first_loss[j] for j in first_loss},
            "final_loss": {str(j): final_loss[j] for j in final_loss},
            "forgetting_mean": forg_mean,
            "gain_erased_frac": frac_mean,
            "routing_overlap": ov_routed,
            "activation_overlap": ov_act,
            "activation_sparsity": spars,
            "domain_probe_acc": probe,
            "final_loss_last_domain": final_loss[len(domains) - 1],
        }

    print("\n================= RÉSUMÉ =================")
    hdr = (f"{'mode':>15} | {'oubli':>6} | {'gain effacé':>11} | {'ov.routage':>10} | "
           f"{'ov.activ':>8} | {'sparsité':>8} | {'sonde dom.':>10}")
    print(hdr); print("-" * len(hdr))
    for mode, r in results.items():
        fm = f"{r['forgetting_mean']:.2f}x" if r["forgetting_mean"] is not None else "n/a"
        fe = f"{r['gain_erased_frac']:.0%}" if r["gain_erased_frac"] is not None else "n/a"
        ovr = f"{r['routing_overlap']:.1%}" if r["routing_overlap"] is not None else "  -  "
        print(f"{mode:>15} | {fm:>6} | {fe:>11} | {ovr:>10} | {r['activation_overlap']:>7.1%} | "
              f"{r['activation_sparsity']:>7.1%} | {r['domain_probe_acc']:>9.1%}")

    meta = {"args": {k: v for k, v in vars(a).items() if k != "mode"},
            "device": dev, "vocab": V, "domains": names, "chance_probe": 1.0 / len(domains)}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"[json] -> {os.path.join(a.out, 'metrics.json')}  (chance sonde={1/len(domains):.1%})")
    plot_curves(results, names, os.path.join(a.out, "jtext_curves.png"))

if __name__ == "__main__":
    main()
