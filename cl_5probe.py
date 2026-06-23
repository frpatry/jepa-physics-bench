"""
Probe à 5 vecteurs — où vit la meilleure représentation dans le multi-vue ?

On sonde SÉPARÉMENT chaque composant : NN1 (cible), NN2/3/4 (contexte), le
predictor, ET leur concaténation. Pour chacun : protocole « sonde gelée » CL
(entraîner la sonde après le domaine j, la geler, ré-évaluer à la fin).

But : NN1 (qu'on sondait jusqu'ici) est le plus sous-entraîné ; les autres
composants portent peut-être une meilleure représentation (plasticité/rétention).

  python cl_5probe.py --n_views 3 --domains 3 --steps 1000 --seeds 3
"""
import argparse, json, os, statistics, torch, torch.nn as nn, torch.nn.functional as F
from cl_jepa_demo import load_labeled, split_90_10, train_with_replay
from cl_multiview import MultiViewBarlow

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n_views", type=int, default=3)
    p.add_argument("--corruptions", nargs="+", default=["mask"])
    p.add_argument("--domains", type=int, default=3)
    p.add_argument("--n_per_domain", type=int, default=4000)
    p.add_argument("--d_model", type=int, default=256); p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4); p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--k", type=int, default=128); p.add_argument("--route_dim", type=int, default=32)
    p.add_argument("--route_win", type=int, default=4); p.add_argument("--seq", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.3); p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--bt_lambda", type=float, default=5e-3); p.add_argument("--var_w", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--probe_steps", type=int, default=300)
    p.add_argument("--seeds", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="dense"); p.add_argument("--target", default="multiview")
    p.add_argument("--ema_decay", type=float, default=0.996); p.add_argument("--log_every", type=int, default=0)
    p.add_argument("--out", type=str, default="runs_5probe")
    return p.parse_args()

@torch.no_grad()
def comp_feats(model, ids, dev, bs=128):
    """Dict composant -> features (N, d), entrée propre, modèle en eval (BN déterministe)."""
    was = model.training; model.eval()
    acc = {}
    for i in range(0, ids.size(0), bs):
        for k, v in model.component_features(ids[i:i+bs].to(dev)).items():
            acc.setdefault(k, []).append(v)
    if was: model.train()
    return {k: torch.cat(v) for k, v in acc.items()}

def train_probe(X, y, n_cls, dev, steps):
    clf = nn.Linear(X.size(1), n_cls).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(clf(X), y).backward(); opt.step()
    for p in clf.parameters(): p.requires_grad_(False)
    return clf

@torch.no_grad()
def acc_of(clf, X, y):
    return (clf(X).argmax(-1) == y).float().mean().item()

def run_seed(domains, V, a, dev, seed):
    torch.manual_seed(seed)
    model = MultiViewBarlow(a, V).to(dev)
    g = torch.Generator().manual_seed(1000 + seed)
    splits = [split_90_10(ids, y, g) for ids, y, _, _ in domains]
    # probes[domain][component] = (clf, heldout_feats? no, recompute), acc_before
    store = {}                                   # (j, comp) -> (clf, acc_before)
    for j in range(len(domains)):
        (tr_ids, tr_y), (ho_ids, ho_y) = splits[j]
        ncls = domains[j][2]
        train_with_replay(model, tr_ids, [], 0.0, a, dev)
        trf = comp_feats(model, tr_ids, dev); hof = comp_feats(model, ho_ids, dev)
        tr_y_d, ho_y_d = tr_y.to(dev), ho_y.to(dev)
        for comp in trf:
            clf = train_probe(trf[comp], tr_y_d, ncls, dev, a.probe_steps)
            store[(j, comp)] = (clf, acc_of(clf, hof[comp], ho_y_d))
    # à la fin : ré-évaluer chaque sonde gelée avec le modèle ACTUEL
    out = {}                                     # comp -> list per old domain of (before, after)
    last = len(domains) - 1
    for j in range(len(domains)):
        (_, _), (ho_ids, ho_y) = splits[j]
        hof = comp_feats(model, ho_ids, dev); ho_y_d = ho_y.to(dev)
        for comp in hof:
            clf, before = store[(j, comp)]
            after = acc_of(clf, hof[comp], ho_y_d)
            out.setdefault(comp, []).append({"before": before, "after": after, "is_last": j == last})
    return out

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    domains, V = load_labeled(a)
    names = [d[3] for d in domains]
    print(f"device={dev} vocab={V} domaines={names} n_views={a.n_views} corruptions={a.corruptions}\n", flush=True)

    # agrège par composant sur les graines
    comp_stats = {}      # comp -> {"plast":[], "final":[], "forget":[]}
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        out = run_seed(domains, V, a, dev, a.seed + s)
        for comp, recs in out.items():
            plast = sum(r["before"] for r in recs) / len(recs)
            final = sum(r["after"] for r in recs) / len(recs)
            olds = [r for r in recs if not r["is_last"]]
            forget = sum(r["before"] - r["after"] for r in olds) / len(olds) if olds else 0.0
            d = comp_stats.setdefault(comp, {"plast": [], "final": [], "forget": []})
            d["plast"].append(plast); d["final"].append(final); d["forget"].append(forget)
        print("    " + " | ".join(f"{c}: oubli {sum(comp_stats[c]['forget'][-1:]):.0%}" for c in out), flush=True)

    def ms(xs): return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)
    print("\n========== PROBE 5 VECTEURS (multi-vue, par composant) ==========")
    hdr = f"{'composant':>11} | {'plasticité':>16} | {'rétention finale':>18} | {'oubli (anciens)':>16}"
    print(hdr); print("-" * len(hdr))
    order = ["NN1"] + [f"NN{i+2}" for i in range(a.n_views)] + ["predictor", "concat"]
    summary = {}
    for comp in order:
        if comp not in comp_stats: continue
        pm, ps = ms(comp_stats[comp]["plast"]); fm, fs = ms(comp_stats[comp]["final"])
        om, osd = ms(comp_stats[comp]["forget"])
        summary[comp] = {"plast": pm, "plast_std": ps, "final": fm, "final_std": fs,
                         "forget": om, "forget_std": osd}
        print(f"{comp:>11} | {pm:.3f} ± {ps:>5.3f} | {fm:.3f} ± {fs:>5.3f} | {om:+.1%} ± {osd:.1%}")

    with open(os.path.join(a.out, "probe5_metrics.json"), "w") as f:
        json.dump({"meta": {"domains": names, "n_views": a.n_views, "corruptions": a.corruptions},
                   "summary": summary}, f, indent=2)
    print(f"\n[json] -> {os.path.join(a.out, 'probe5_metrics.json')}")

if __name__ == "__main__":
    main()
