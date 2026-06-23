"""
DÉMONSTRATION jouet : stabilité-plasticité en continual learning JEPA.

But : montrer qu'un modèle JEPA **entièrement entraînable** (PAS de gel, PAS de
modules par classe) peut apprendre de nouveaux domaines (PLASTICITÉ) tout en
RETENANT les anciens — grâce au REPLAY (rejeu d'un petit tampon d'exemples
passés, entremêlés à l'entraînement). C'est général (niveau données), et c'est
l'analogue du replay du sommeil / consolidation chez l'enfant.

Mesure NON biaisée = sonde linéaire gelée sur VRAIS LABELS (classification) :
accuracy haute et stable (70-90%), contrairement à la sonde token (~5%, bruitée).
  - PLASTICITÉ = accuracy juste après avoir appris le domaine (a-t-il bien appris ?)
  - RÉTENTION  = accuracy finale (même sonde gelée) ; oubli = chute.

Conditions comparées (EMA fixe, modèle jamais figé) :
  - replay=0.0  : témoin -> oubli attendu
  - replay>0    : un peu de rejeu -> rétention SANS perdre la plasticité

Le JEPA reste self-supervised (prédiction latente de tokens masqués, cible EMA) ;
les labels ne servent QU'À mesurer.

  python cl_jepa_demo.py --replays 0 0.5 --seeds 3
"""
import argparse, json, os, statistics, torch, torch.nn as nn, torch.nn.functional as F
from cl_jepa_text import TextJEPA, make_mask, get_batch
from cl_barlow import BarlowJEPA
from cl_multiview import MultiViewBarlow

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", type=float, nargs="+", default=[0.0, 0.5],
                   help="fractions de replay à comparer (0 = témoin)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--domains", type=int, default=3)
    p.add_argument("--n_per_domain", type=int, default=4000)
    p.add_argument("--buffer", type=int, default=512, help="exemples gardés par domaine passé")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--route_dim", type=int, default=32)
    p.add_argument("--route_win", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema_decay", type=float, default=0.996)
    p.add_argument("--var_w", type=float, default=1.0)
    p.add_argument("--probe_steps", type=int, default=300)
    p.add_argument("--mode", default="dense")          # PAS de routage : on veut du général
    p.add_argument("--target", default="ema")
    p.add_argument("--objective", choices=["jepa", "barlow", "multiview"], default="barlow",
                   help="barlow = 1 vue corrompue ; multiview = N vues corrompues -> prédisent la vue propre ; jepa = ancien (EMA)")
    p.add_argument("--n_views", type=int, default=3, help="nb de vues corrompues (multiview)")
    p.add_argument("--corruptions", nargs="+", default=["mask"],
                   help="types de brouillage par vue (cyclique) : mask span subst noise drop")
    p.add_argument("--separate_embed", action="store_true",
                   help="multiview: chaque NN a sa propre table d'embedding (enlève la surface partagée)")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--bt_lambda", type=float, default=5e-3)
    p.add_argument("--bt_stopgrad", action="store_true")
    p.add_argument("--mixed", action="store_true",
                   help="entraînement CONJOINT (tous les domaines mélangés i.i.d.) = plafond, pas de CL")
    p.add_argument("--synthetic", action="store_true", help="labels synthétiques (smoke test)")
    p.add_argument("--out", type=str, default="runs_demo")
    p.add_argument("--log_every", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

# --------------------------- données étiquetées -----------------------------
def load_labeled(a):
    """Domaines de CLASSIFICATION distincts (vrais labels) -> sonde nette.
    JEPA s'entraîne sans labels ; labels = mesure seulement."""
    if a.synthetic:
        # synthétique étiqueté : k clusters de Markov, label = cluster
        V = 512; outs = []
        g = torch.Generator().manual_seed(a.seed)
        for d in range(a.domains):
            ncls = 3 + d
            seqs, labs = [], []
            rules = [torch.randperm(V, generator=g) for _ in range(ncls)]
            for c in range(ncls):
                n = a.n_per_domain // ncls
                s = torch.zeros(n, a.seq, dtype=torch.long)
                s[:, 0] = torch.randint(0, V, (n,), generator=g)
                for i in range(1, a.seq): s[:, i] = rules[c][s[:, i - 1]]
                seqs.append(s); labs.append(torch.full((n,), c))
            ids = torch.cat(seqs); y = torch.cat(labs)
            perm = torch.randperm(ids.size(0), generator=g)
            outs.append((ids[perm], y[perm], ncls, f"syn{d}"))
        return outs, V
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    V = tok.vocab_size
    specs = [("fancyzhx/ag_news", None, "text", "ag_news", 4),
             ("fancyzhx/dbpedia_14", None, "content", "dbpedia", 14),
             ("stanfordnlp/imdb", None, "text", "imdb", 2),
             ("fancyzhx/yelp_polarity", None, "text", "yelp", 2)][:a.domains]
    outs = []
    for repo, cfg, col, name, ncls in specs:
        ds = load_dataset(repo, cfg, split="train").shuffle(seed=a.seed)
        ds = ds.select(range(min(a.n_per_domain, len(ds))))
        enc = tok(list(ds[col]), truncation=True, max_length=a.seq,
                  padding="max_length", return_tensors="pt")
        ids = enc.input_ids; y = torch.tensor(ds["label"])
        outs.append((ids, y, ncls, name))
        print(f"[data] {name:>9}: {ids.shape[0]} séq, {ncls} classes", flush=True)
    return outs, V

def split_90_10(ids, y, gen):
    n = ids.size(0); perm = torch.randperm(n, generator=gen); ntr = int(0.9 * n)
    return (ids[perm[:ntr]], y[perm[:ntr]]), (ids[perm[ntr:]], y[perm[ntr:]])

# --------------------------- sonde sur labels (gelée) -----------------------
@torch.no_grad()
def features(model, ids, dev, bs=128):
    outs = []
    for i in range(0, ids.size(0), bs):
        outs.append(model.features(ids[i:i+bs].to(dev)).mean(1))
    return torch.cat(outs)

@torch.no_grad()
def effective_rank(Z):
    """Finesse de la représentation = participation ratio des valeurs propres de
    la covariance. Dans [1, d] : haut = beaucoup de dims décorrélées utilisées
    (rep FINE) ; bas = info compressée/redondante (rep grossière)."""
    Z = Z - Z.mean(0, keepdim=True)
    C = (Z.t() @ Z) / max(1, Z.size(0))
    ev = torch.linalg.eigvalsh(C).clamp(min=1e-12)
    return (ev.sum() ** 2 / (ev ** 2).sum()).item()

@torch.no_grad()
def compute_effrank(model, splits, dev, n=512):
    Z = torch.cat([features(model, splits[j][1][0][:n], dev) for j in range(len(splits))])
    return effective_rank(Z)

def train_label_probe(model, ids, y, n_cls, a, dev):
    X = features(model, ids, dev); y = y.to(dev)
    clf = nn.Linear(X.size(1), n_cls).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for _ in range(a.probe_steps):
        opt.zero_grad(); F.cross_entropy(clf(X), y).backward(); opt.step()
    for pp in clf.parameters(): pp.requires_grad_(False)
    return clf

@torch.no_grad()
def eval_label_probe(model, clf, ids, y, dev):
    X = features(model, ids, dev)
    return (clf(X).argmax(-1) == y.to(dev)).float().mean().item()

# --------------------------- entraînement avec replay -----------------------
def train_with_replay(model, data, buffer, rho, a, dev):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    buf = torch.cat(buffer) if buffer else None       # exemples des domaines passés
    n_rep = int(rho * a.bs) if buf is not None else 0
    for step in range(a.steps):
        x = get_batch(data, a.bs - n_rep, dev)
        if n_rep > 0:
            idx = torch.randint(0, buf.size(0), (n_rep,))
            x = torch.cat([x, buf[idx].to(dev)], 0)   # entremêle ancien + neuf
        mask = make_mask(x.size(0), x.size(1), a.mask_ratio, None).to(dev)
        total, _, _ = model(x, mask)
        opt.zero_grad(); total.backward(); opt.step(); model.ema_update()

# --------------------------- entraînement simple (pour le mode mélangé) -----
def train_plain(model, data, steps, a, dev):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    for _ in range(steps):
        x = get_batch(data, a.bs, dev)
        mask = make_mask(x.size(0), x.size(1), a.mask_ratio, None).to(dev)
        total, _, _ = model(x, mask)
        opt.zero_grad(); total.backward(); opt.step(); model.ema_update()

# --------------------------- une exécution CL -------------------------------
def run_cl(domains, V, rho, a, dev, seed):
    torch.manual_seed(seed)
    a.mode = a.mode; a.ema_decay = a.ema_decay
    model = {"barlow": BarlowJEPA, "multiview": MultiViewBarlow,
             "jepa": TextJEPA}[a.objective](a, V).to(dev)
    g = torch.Generator().manual_seed(1000 + seed)
    splits = [split_90_10(ids, y, g) for ids, y, _, _ in domains]

    if a.mixed:
        # CONJOINT : on mélange tous les domaines et on entraîne UNE fois (budget total identique)
        pool = torch.cat([splits[j][0][0] for j in range(len(domains))], 0)
        train_plain(model, pool.to(dev), a.steps * len(domains), a, dev)
        out = {}
        for j in range(len(domains)):
            (tr_ids, tr_y), (ho_ids, ho_y) = splits[j]
            ncls, name = domains[j][2], domains[j][3]
            clf = train_label_probe(model, tr_ids.to(dev), tr_y, ncls, a, dev)
            acc = eval_label_probe(model, clf, ho_ids.to(dev), ho_y, dev)
            out[name] = {"before": acc, "after": acc, "chance": 1.0 / ncls, "is_last": False}
        return out, compute_effrank(model, splits, dev)
    probes, buffer = {}, []
    for j in range(len(domains)):
        (tr_ids, tr_y), (ho_ids, ho_y) = splits[j]
        ncls, name = domains[j][2], domains[j][3]
        train_with_replay(model, tr_ids, buffer, rho, a, dev)
        clf = train_label_probe(model, tr_ids, tr_y, ncls, a, dev)
        acc_before = eval_label_probe(model, clf, ho_ids, ho_y, dev)
        probes[j] = {"clf": clf, "ho": (ho_ids, ho_y), "before": acc_before,
                     "name": name, "chance": 1.0 / ncls}
        buffer.append(tr_ids[:a.buffer])               # garde un échantillon pour le replay
    out = {}
    for j, p in probes.items():
        acc_after = eval_label_probe(model, p["clf"], p["ho"][0], p["ho"][1], dev)
        out[p["name"]] = {"before": p["before"], "after": acc_after, "chance": p["chance"],
                          "is_last": j == len(domains) - 1}
    return out, compute_effrank(model, splits, dev)

# --------------------------- main -------------------------------------------
def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.mixed: a.replays = [0.0]                  # conjoint : le replay n'a pas de sens
    os.makedirs(a.out, exist_ok=True)
    domains, V = load_labeled(a)
    names = [d[3] for d in domains]
    print(f"device={dev}  vocab={V}  domaines={names}  "
          f"classes={[d[2] for d in domains]}\n", flush=True)

    results, results_er = {}, {}
    for rho in a.replays:
        runs, ers = [], []
        for s in range(a.seeds):
            print(f"--- replay={rho} seed={s} ---", flush=True)
            out, er = run_cl(domains, V, rho, a, dev, seed=a.seed + s)
            for dn, r in out.items():
                tag = " (dernier)" if r["is_last"] else ""
                print(f"    {dn:>9}: appris {r['before']:.3f} -> final {r['after']:.3f} "
                      f"(chance {r['chance']:.2f}){tag}", flush=True)
            print(f"    rang effectif (finesse) = {er:.1f}", flush=True)
            runs.append(out); ers.append(er)
        results[str(rho)] = runs; results_er[str(rho)] = ers

    # agrégation
    def stat(runs, key, only_old=False):
        vals = []
        for out in runs:
            xs = [ (r["before"] if key=="before" else r["after"] if key=="after"
                    else r["before"]-r["after"])
                   for r in out.values() if (not only_old or not r["is_last"]) ]
            if xs: vals.append(sum(xs)/len(xs))
        return (statistics.mean(vals), statistics.pstdev(vals) if len(vals)>1 else 0.0) if vals else (float("nan"),0)

    print("\n================= DÉMONSTRATION stabilité-plasticité =================")
    print("(modèle 100% entraînable, pas de gel, pas de modules par classe)\n")
    hdr = (f"{'replay':>7} | {'plasticité (appris)':>19} | {'rétention (final)':>18} | "
           f"{'OUBLI (anciens)':>16} | {'rang eff. (finesse)':>19}")
    print(hdr); print("-" * len(hdr))
    summary = {}
    for rho in a.replays:
        pm, ps = stat(results[str(rho)], "before")           # plasticité = bien appris ?
        fm, fs = stat(results[str(rho)], "after")            # accuracy finale moyenne
        om, os_ = stat(results[str(rho)], "drop", only_old=True)  # oubli sur anciens
        ers = results_er[str(rho)]
        erm = statistics.mean(ers); ersd = statistics.pstdev(ers) if len(ers) > 1 else 0.0
        summary[str(rho)] = {"plasticity": pm, "plasticity_std": ps, "final_acc": fm,
                             "final_acc_std": fs, "forget": om, "forget_std": os_,
                             "eff_rank": erm, "eff_rank_std": ersd}
        print(f"{rho:>7} | {pm:.3f} ± {ps:>5.3f}      | {fm:.3f} ± {fs:>5.3f}     | "
              f"{om:+.1%} ± {os_:.1%} | {erm:>7.1f} ± {ersd:.1f}")

    # verdict prudent (effet doit dépasser le bruit)
    print("\n--- VERDICT ---")
    if len(a.replays) >= 2:
        r0, r1 = str(min(a.replays)), str(max(a.replays))
        d_for = summary[r0]["forget"] - summary[r1]["forget"]      # réduction d'oubli
        noise = summary[r0]["forget_std"] + summary[r1]["forget_std"]
        d_plast = summary[r1]["plasticity"] - summary[r0]["plasticity"]
        if summary[r0]["forget"] < 0.03:
            print(f"⚠️  Le témoin (replay={r0}) n'oublie pas ({summary[r0]['forget']:+.1%}) "
                  f"-> rien à démontrer. Domaines trop compatibles.")
        elif d_for > noise and d_for > 0.03:
            print(f"✅ Le replay RÉDUIT l'oubli de {summary[r0]['forget']:+.1%} à "
                  f"{summary[r1]['forget']:+.1%} (effet {d_for:.1%} > bruit {noise:.1%}), "
                  f"en gardant la plasticité ({summary[r1]['plasticity']:.3f} vs {summary[r0]['plasticity']:.3f}). "
                  f"-> stabilité ET plasticité, modèle non figé.")
        else:
            print(f"❌ Effet du replay ({d_for:.1%}) DANS le bruit ({noise:.1%}) -> non concluant.")

    with open(os.path.join(a.out, "demo_metrics.json"), "w") as f:
        json.dump({"meta": {"domains": names, "args": {k: v for k, v in vars(a).items()}},
                   "summary": summary}, f, indent=2)
    print(f"\n[json] -> {os.path.join(a.out, 'demo_metrics.json')}")

if __name__ == "__main__":
    main()
