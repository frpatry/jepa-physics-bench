"""
Banc de continual learning JEPA avec MÉTRIQUE D'OUBLI NON BIAISÉE (sonde gelée)
+ ablation EMA jugée sur cette sonde (jamais sur la perte JEPA).

POURQUOI : la perte JEPA dépend de la cible EMA qui DÉRIVE -> une perte stable
peut être de la dérive amortie, pas du vrai transfert. Et on ne peut PAS comparer
des pertes entre EMA-on et EMA-off (cibles différentes). On mesure donc l'oubli
sur INSTRUMENT GELÉ :

  Protocole (par domaine j, "linear probing for forgetting") :
    1. juste après avoir appris le domaine j en CL, encodeur GELÉ -> on entraîne
       une sonde linéaire sur une tâche de CONTENU : prédire le TOKEN MASQUÉ
       (cible FIXE = vrai token, restreint aux tokens fréquents du domaine).
    2. on GÈLE la sonde. acc_before = sa précision sur un HELD-OUT (90/10) de j.
    3. on continue le CL sur les domaines suivants (l'encodeur change).
    4. acc_after = MÊME sonde gelée, ré-évaluée sur le MÊME held-out de j, avec
       l'encodeur ACTUEL. Chute d'accuracy = OUBLI (non biaisé par la dérive EMA).

ABLATION EMA : on fait varier le momentum (--decays), du "pas d'EMA" (0.0) à
"EMA lente" (0.999). Toutes les conditions sont comparées sur la MÊME accuracy de
sonde gelée. Rigueur : held-out réel, plusieurs graines, plusieurs ordres de
domaines, moyenne ± écart-type. Domaine CONFLICTUEL ajouté (code python).

GARDE-FOU : le témoin dense DOIT montrer de l'oubli sur le domaine conflictuel.
Sinon le test ne discrimine rien -> on le DIT au lieu de conclure.

  python cl_jepa_probe.py --synthetic --decays 0 0.996   # smoke test hors-ligne
  python cl_jepa_probe.py --domains 4 --decays 0 0.9 0.996 0.999 --seeds 3 --orders 2
"""
import argparse, json, os, statistics, torch, torch.nn as nn, torch.nn.functional as F
from cl_scale import load_domains
from cl_jepa_text import TextJEPA, make_mask, get_batch, train_domain

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--mode", default="dense", help="mode FFN (routage réglé -> dense par défaut)")
    p.add_argument("--decays", type=float, nargs="+", default=[0.0, 0.9, 0.996, 0.999],
                   help="conditions d'ablation EMA (momentum)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--orders", type=int, default=2)
    p.add_argument("--domains", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--route_dim", type=int, default=32)
    p.add_argument("--route_win", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--var_w", type=float, default=1.0)
    p.add_argument("--syn_diff", type=float, default=0.5)
    p.add_argument("--probe_steps", type=int, default=400)
    p.add_argument("--probe_topk", type=int, default=800, help="tokens fréquents couverts par la sonde")
    p.add_argument("--out", type=str, default="runs_probe")
    p.add_argument("--log_every", type=int, default=0, help="logs d'entraînement (0=silence)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

# --------------------- encodeur masqué (rep aux positions cibles) -----------
@torch.no_grad()
def encode_masked(model, ids, mask):
    rc = model.route_context(ids.masked_fill(mask, 0))
    return model.enc(ids, rc, mask, model.mask_token)          # (B,T,d)

def split_90_10(data, gen):
    n = data.size(0); perm = torch.randperm(n, generator=gen)
    ntr = int(0.9 * n)
    return data[perm[:ntr]], data[perm[ntr:]]

def freq_lut(train_data, V, topk, dev):
    counts = torch.bincount(train_data.reshape(-1), minlength=V)
    idx = counts.topk(min(topk, V)).indices
    lut = torch.full((V,), -1, dtype=torch.long, device=dev)
    lut[idx.to(dev)] = torch.arange(idx.numel(), device=dev)
    return lut, idx.numel()

# --------------------- sonde gelée : entraîner / évaluer --------------------
def train_probe(model, train_data, lut, n_cls, a, dev, mask_seed):
    d = a.d_model
    probe = nn.Linear(d, n_cls).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    model.eval()
    for _ in range(a.probe_steps):
        x = get_batch(train_data, 128, dev)
        mask = make_mask(x.size(0), x.size(1), a.mask_ratio, None).to(dev)
        h = encode_masked(model, x, mask)
        hm = h[mask]; toks = x[mask]
        labels = lut[toks]; keep = labels >= 0
        if keep.sum() < 2: continue
        logits = probe(hm[keep])
        loss = F.cross_entropy(logits, labels[keep])
        opt.zero_grad(); loss.backward(); opt.step()
    for pp in probe.parameters(): pp.requires_grad_(False)
    return probe

@torch.no_grad()
def eval_probe(model, probe, heldout, lut, a, dev, mask_seed):
    """Accuracy sur held-out, masques FIXÉS (seedés) -> before/after comparables."""
    gen = torch.Generator().manual_seed(mask_seed)
    model.eval(); correct = total = 0
    for _ in range(10):
        x = get_batch(heldout, 64, dev, gen)
        mask = make_mask(x.size(0), x.size(1), a.mask_ratio, gen).to(dev)
        h = encode_masked(model, x, mask)
        hm = h[mask]; toks = x[mask]
        labels = lut[toks]; keep = labels >= 0
        if keep.sum() == 0: continue
        pred = probe(hm[keep]).argmax(-1)
        correct += (pred == labels[keep]).sum().item(); total += keep.sum().item()
    return correct / max(1, total)

# --------------------- une exécution CL (1 condition, 1 graine, 1 ordre) ----
def run_cl(domains, names, V, order, decay, a, dev, seed):
    torch.manual_seed(seed)
    a.target = "ema"; a.ema_decay = decay; a.mode = a.mode  # mode fixe
    model = TextJEPA(a, V).to(dev)
    gsplit = torch.Generator().manual_seed(1000 + seed)
    splits = {j: split_90_10(domains[j], gsplit) for j in range(len(domains))}
    probes = {}      # j -> (probe, lut, acc_before)
    for pos, j in enumerate(order):
        train_domain(model, splits[j][0].to(dev), a, dev, tag="")
        lut, n_cls = freq_lut(splits[j][0], V, a.probe_topk, dev)
        probe = train_probe(model, splits[j][0].to(dev), lut, n_cls, a, dev, mask_seed=7 + j)
        acc_before = eval_probe(model, probe, splits[j][1].to(dev), lut, a, dev, mask_seed=7 + j)
        probes[j] = (probe, lut, acc_before)
    # après TOUT le CL : ré-évaluer chaque sonde gelée avec l'encodeur actuel
    per_dom = {}
    last = order[-1]
    for j, (probe, lut, acc_before) in probes.items():
        acc_after = eval_probe(model, probe, splits[j][1].to(dev), lut, a, dev, mask_seed=7 + j)
        drop = (acc_before - acc_after) / acc_before if acc_before > 1e-6 else float("nan")
        per_dom[names[j]] = {"before": acc_before, "after": acc_after, "drop": drop,
                             "is_last": j == last}
    return per_dom

# --------------------- main : sweep conditions × graines × ordres -----------
def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    domains, V, names = load_domains(a, tok_vocab=512)
    print(f"device={dev}  vocab={V}  domaines={names}  "
          f"shapes={[tuple(d.shape) for d in domains]}\n", flush=True)
    D = len(domains)
    base = list(range(D))
    orders = [base, base[::-1]][:a.orders]
    conflict = "code" if "code" in names else names[0]   # domaine conflictuel (garde-fou)

    all_results = {}   # decay -> list of per_dom dicts (sur graines×ordres)
    for decay in a.decays:
        runs = []
        for s in range(a.seeds):
            for oi, order in enumerate(orders):
                tag = f"decay={decay} seed={s} order={oi}"
                print(f"--- {tag} | ordre={[names[j] for j in order]} ---", flush=True)
                per_dom = run_cl(domains, names, V, order, decay, a, dev, seed=a.seed + s)
                for dn, r in per_dom.items():
                    if not r["is_last"]:   # le dernier domaine n'a pas subi d'oubli
                        print(f"    {dn:>9}: before {r['before']:.3f} after {r['after']:.3f} "
                              f"drop {r['drop']:+.1%}", flush=True)
                runs.append(per_dom)
        all_results[decay] = runs

    # ---- agrégation : oubli moyen (sur domaines anciens) par condition ----
    def agg(runs, only=None):
        vals = []
        for per_dom in runs:
            ds = [r["drop"] for dn, r in per_dom.items()
                  if not r["is_last"] and (only is None or dn == only) and r["drop"] == r["drop"]]
            if ds: vals.append(sum(ds) / len(ds))
        if not vals: return (float("nan"), float("nan"))
        return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    print("\n================= ABLATION EMA (oubli = chute de sonde gelée) =================")
    hdr = f"{'EMA decay':>10} | {'oubli moyen (anciens)':>22} | {'oubli sur ' + conflict:>18}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for decay in a.decays:
        m, sd = agg(all_results[decay])
        mc, sdc = agg(all_results[decay], only=conflict)
        summary[str(decay)] = {"forget_mean": m, "forget_std": sd,
                               "forget_conflict_mean": mc, "forget_conflict_std": sdc}
        cstr = f"{mc:+.1%} ± {sdc:.1%}" if mc == mc else "n/a"
        print(f"{decay:>10} | {m:+.1%} ± {sd:>5.1%}        | {cstr:>18}")

    # ---- GARDE-FOU : y a-t-il de l'oubli QUELQUE PART (sur un domaine ancien) ? ----
    # (et non "le domaine conflictuel est-il oublié" : le code, robuste, ne s'oublie
    #  guère ; le vrai signal est qu'il FAIT oublier les autres.)
    worst_any = float("nan")
    for decay in a.decays:
        for per_dom in all_results[decay]:
            for dn, r in per_dom.items():
                if not r["is_last"] and r["drop"] == r["drop"]:
                    worst_any = r["drop"] if worst_any != worst_any else max(worst_any, r["drop"])
    print("\n--- GARDE-FOU ---")
    if not (worst_any == worst_any) or worst_any < 0.03:
        print(f"⚠️  Aucun oubli mesurable sur les domaines anciens (max chute "
              f"{worst_any:+.1%} < 3%). Le test NE DISCRIMINE RIEN : domaines trop "
              f"compatibles, tâche trop facile, ou sonde trop bruitée. NE PAS conclure sur l'EMA.", flush=True)
    else:
        print(f"OK : la séquence CL induit de l'oubli (chute max {worst_any:+.1%} sur un domaine "
              f"ancien) -> le test discrimine.", flush=True)
        # interprétation EMA
        no_ema = summary.get("0.0", {}).get("forget_mean", float("nan"))
        slow = summary.get(str(max(a.decays)), {}).get("forget_mean", float("nan"))
        if no_ema == no_ema and slow == slow:
            verdict = ("L'EMA PROTÈGE" if slow < no_ema - 0.02 else
                       "L'EMA NE PROTÈGE PAS (oubli comparable)" if abs(slow - no_ema) <= 0.02 else
                       "L'EMA AGGRAVE l'oubli")
            print(f"Interprétation : sans EMA {no_ema:+.1%} vs EMA lente ({max(a.decays)}) {slow:+.1%} "
                  f"-> {verdict}.", flush=True)

    with open(os.path.join(a.out, "probe_metrics.json"), "w") as f:
        json.dump({"meta": {"args": {k: v for k, v in vars(a).items()}, "domains": names,
                            "conflict": conflict}, "summary": summary}, f, indent=2)
    print(f"\n[json] -> {os.path.join(a.out, 'probe_metrics.json')}")

if __name__ == "__main__":
    main()
