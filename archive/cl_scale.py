"""
Continual learning à plus grande échelle — test de l'hypothèse :
le routage STABLE adressé par le contexte protège-t-il de l'oubli quand les
contextes se chevauchent beaucoup (cas du vrai texte) ?

Modes comparés (FFN de chaque bloc Transformer) :
  - dense        : FFN standard (référence, oubli attendu élevé)
  - routed_hard  : gate FIXE = hash du contexte local (route-emb gelée -> proj
                   gelée -> top-K). Stable, sans labels, identique train/infer.
  - learned_topk : top-K sur les activations APPRISES (le sélecteur dérive ->
                   sert à montrer que la stabilité, pas la sparsité, compte).

Objectif : LM causal (prédire le token suivant). Le gate est causal (fenêtre
de tokens passés) donc utilisable en génération autorégressive.

Boucle CL : entraîner domaine 1, puis 2, ... ; après CHAQUE domaine, évaluer la
perplexité sur TOUS les domaines vus -> l'oubli = hausse de perplexité sur les
domaines antérieurs.

Diagnostic clé : Jaccard moyen PAR POSITION des unités routées entre domaines.
  -> s'il reste bas malgré le chevauchement des contextes, la protection tient.
  -> s'il explose, le routage fixe ne suffit pas sur du vrai texte.

Lancer en local avec Claude Code :
  python cl_scale.py                      # vrai texte (HuggingFace) si dispo
  python cl_scale.py --synthetic          # mode hors-ligne (smoke test)

Sorties (dossier --out, défaut ./runs) :
  metrics.json   : toutes les perplexités par étape + résumé par mode
  ppl_curves.png : perplexité par domaine au fil de la séquence CL (1 sous-graphe
                   par domaine, 1 courbe par mode)
"""
import argparse, math, json, os, torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------- config ---------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="données hors-ligne")
    p.add_argument("--modes", nargs="+",
                   default=["dense", "routed_hard", "learned_topk"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)   # dim cachée du FFN routé
    p.add_argument("--k", type=int, default=128)        # top-K sur d_ff (~12%)
    p.add_argument("--route_win", type=int, default=4)  # fenêtre de contexte (causale)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)   # pas par domaine
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--domains", type=int, default=3)
    # --- journalisation / sorties (n'affecte pas le mécanisme) ---
    p.add_argument("--out", type=str, default="runs", help="dossier des artefacts")
    p.add_argument("--log_every", type=int, default=50, help="pas entre 2 logs de perte")
    p.add_argument("--eval_iters", type=int, default=20, help="batches d'éval par mesure de ppl")
    p.add_argument("--seed", type=int, default=0)
    # --- knob ADDITIF : distinction des domaines synthétiques ---
    # fraction des transitions de Markov qui diffèrent entre domaines.
    # 0.2 = très chevauchant (défaut historique) ; ↑ = domaines plus distincts
    # -> à monter SI le dense n'oublie pas (sans toucher au mécanisme de routage).
    p.add_argument("--syn_diff", type=float, default=0.2)
    return p.parse_args()

# --------------------------- données ----------------------------------------
def load_domains(args, tok_vocab):
    """Retourne (liste de tenseurs LongTensor [N,seq], taille_vocab, noms).
    Vrai texte via HuggingFace si dispo ; sinon synthétique à contextes
    chevauchants (règles de Markov partageant 1-syn_diff des transitions).

    Domaines vrai-texte : registres VOLONTAIREMENT distincts (encyclopédie /
    dépêches / critiques / médical) — c'est la distinction des distributions qui
    crée l'oubli à mesurer. Tokenizer gpt2 partagé entre tous les domaines."""
    if not args.synthetic:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
            V = tok.vocab_size
            # (repo_id NAMESPACÉ, config, split, text_column, nom court) — registres distincts.
            # Les datasets récents exigent 'namespace/nom' (les noms nus comme
            # 'wikitext' lèvent HfUriError).
            specs = [
                ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text",     "wikitext"),  # encyclopédie
                ("fancyzhx/ag_news",    None,                "train", "text",     "ag_news"),   # dépêches d'actu
                ("stanfordnlp/imdb",    None,                "train", "text",     "imdb"),      # critiques de films
                ("google/code_x_glue_ct_code_to_text", "python", "train", "code", "code"),      # CODE python (conflictuel)
                ("qiaojin/PubMedQA",    "pqa_labeled",       "train", "question", "pubmed"),    # médical
            ][:args.domains]
            outs, names = [], []
            for repo, cfg, split, col, short in specs:
                try:                                            # un dataset qui foire n'annule pas les autres
                    ds = load_dataset(repo, cfg, split=f"{split}[:4000]")
                    if col not in ds.column_names:
                        col = "text" if "text" in ds.column_names else ds.column_names[0]
                    texts = [t for t in ds[col] if isinstance(t, str) and t.strip()]
                    ids = tok("\n".join(texts)[:2_000_000],
                              return_tensors="pt", truncation=False).input_ids[0]
                    n = (ids.numel() // args.seq) * args.seq
                    if n == 0:
                        print(f"[!] domaine {short} vide après tokenisation -> ignoré."); continue
                    outs.append(ids[:n].view(-1, args.seq)); names.append(short)
                    print(f"[data] {short:>10}: {outs[-1].shape[0]} séquences de {args.seq} tokens", flush=True)
                except Exception as e:
                    print(f"[!] dataset {repo} échoué ({type(e).__name__}: {e}) -> ignoré.", flush=True)
            if len(outs) >= 2:
                return outs, V, names
            print(f"[!] <2 domaines chargés -> bascule synthétique.")
        except Exception as e:
            print(f"[!] HuggingFace indisponible ({type(e).__name__}: {e}) -> mode synthétique.")
    # ---- synthétique : contextes chevauchants (syn_diff contrôle la distinction) ----
    V = tok_vocab
    base = torch.randperm(V)
    domains, names = [], []
    g = torch.Generator().manual_seed(args.seed)
    for d in range(args.domains):
        rule = base.clone()
        k = max(1, int(args.syn_diff * V))     # transitions qui diffèrent
        ix = torch.randperm(V, generator=g)[:k]
        rule[ix] = torch.randperm(V, generator=g)[:k]
        N = 3000
        s = torch.zeros(N, args.seq, dtype=torch.long)
        s[:, 0] = torch.randint(0, V, (N,), generator=g)
        for i in range(1, args.seq):
            s[:, i] = rule[s[:, i - 1]]
        domains.append(s); names.append(f"syn{d}")
    return domains, V, names

# --------------------------- modèle -----------------------------------------
class RoutedFFN(nn.Module):
    def __init__(self, d, d_ff, k, mode, route_dim):
        super().__init__()
        self.mode, self.k = mode, k
        self.fc1 = nn.Linear(d, d_ff); self.fc2 = nn.Linear(d_ff, d)
        # routed_protect = MÊME routage FFN que routed_hard (ajout additif, ne
        # change pas routed_hard) ; la différence se joue dans la boucle CL où
        # routed_protect gèle aussi les params partagés (embeddings + tête).
        if mode in ("routed_hard", "routed_protect"):
            # projection de routage GELÉE, propre à ce bloc
            self.register_buffer("Wr", torch.randn(route_dim, d_ff))
    def forward(self, x, route_ctx):
        h = F.relu(self.fc1(x))                 # (B,T,d_ff)
        if self.mode in ("routed_hard", "routed_protect"):
            sc = route_ctx @ self.Wr            # (B,T,d_ff) fonction FIXE du contexte
            thr = sc.topk(self.k, -1).values[..., -1:]
            h = h * (sc >= thr).float()         # masque stable, sans gradient
        elif self.mode == "learned_topk":
            thr = h.topk(self.k, -1).values[..., -1:].detach()
            h = torch.where(h >= thr, h, torch.zeros_like(h))  # sélecteur appris (dérive)
        return self.fc2(h)
    def routed_mask(self, route_ctx):
        sc = route_ctx @ self.Wr
        thr = sc.topk(self.k, -1).values[..., -1:]
        return (sc >= thr)

class Block(nn.Module):
    def __init__(self, a, route_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(a.d_model); self.ln2 = nn.LayerNorm(a.d_model)
        self.attn = nn.MultiheadAttention(a.d_model, a.n_head, batch_first=True)
        self.ffn = RoutedFFN(a.d_model, a.d_ff, a.k, a.mode, route_dim)
    def forward(self, x, mask, route_ctx):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + h
        return x + self.ffn(self.ln2(x), route_ctx)

class LM(nn.Module):
    def __init__(self, a, V):
        super().__init__()
        a_route = 32
        self.tok = nn.Embedding(V, a.d_model)
        self.pos = nn.Embedding(a.seq, a.d_model)
        self.route_emb = nn.Embedding(V, a_route); self.route_emb.weight.requires_grad_(False)
        self.win = a.route_win
        self.blocks = nn.ModuleList([Block(a, a_route) for _ in range(a.n_layer)])
        self.ln = nn.LayerNorm(a.d_model); self.head = nn.Linear(a.d_model, V)
        self.seq = a.seq
    def route_context(self, ids):
        # moyenne CAUSALE des route-emb gelées sur une fenêtre -> stable, causal
        e = self.route_emb(ids)                              # (B,T,r)
        e = e.transpose(1, 2)                                # (B,r,T)
        e = F.pad(e, (self.win - 1, 0))
        e = F.avg_pool1d(e, self.win, 1)                     # moyenne glissante gauche
        return e.transpose(1, 2)                             # (B,T,r)
    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok(ids) + self.pos(pos)
        rc = self.route_context(ids)
        cmask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        for b in self.blocks:
            x = b(x, cmask, rc)
        return self.head(self.ln(x))

# --------------------------- entraînement / éval ----------------------------
def batch(data, bs, dev, gen=None):
    idx = torch.randint(0, data.size(0), (bs,), generator=gen)
    s = data[idx].to(dev)
    return s[:, :-1], s[:, 1:]
def ppl(model, data, dev, iters=20, seed=1234):
    """Perplexité moyenne. Batches d'éval FIXÉS (générateur seedé) pour que les
    courbes ne soient pas bruitées par l'échantillonnage."""
    gen = torch.Generator().manual_seed(seed)
    model.eval(); tot = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = batch(data, 32, dev, gen=gen)
            loss = F.cross_entropy(model(x).reshape(-1, model.head.out_features),
                                   y.reshape(-1))
            tot += loss.item()
    return math.exp(tot / iters)
def train_domain(model, data, a, dev, tag=""):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    for step in range(a.steps):
        x, y = batch(data, a.bs, dev)
        loss = F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if a.log_every and (step % a.log_every == 0 or step == a.steps - 1):
            print(f"    {tag} step {step:5d}/{a.steps}  loss {loss.item():.3f}  "
                  f"ppl {math.exp(min(loss.item(), 20)):8.1f}", flush=True)

def routing_overlap(model, dA, dB, dev, layer=0):
    """Jaccard moyen par position des unités routées (couche `layer`) entre 2
    domaines. LA métrique : reste-t-il bas malgré le chevauchement des contextes ?"""
    ffn = model.blocks[layer].ffn
    if not hasattr(ffn, "Wr"): return None      # mode sans routage gelé
    with torch.no_grad():
        def masks(d):
            x = d[:256, :-1].to(dev); rc = model.route_context(x)
            return ffn.routed_mask(rc).reshape(-1, ffn.fc1.out_features)
        ma, mb = masks(dA), masks(dB)
        n = min(len(ma), len(mb)); ma, mb = ma[:n], mb[:n]
        inter = (ma & mb).sum(-1).float(); union = (ma | mb).sum(-1).float()
        return (inter / union).mean().item()

def mean_pairwise_overlap(model, domains, dev, layer=0):
    """Chevauchement de routage moyen sur TOUTES les paires de domaines (1 couche)."""
    if not hasattr(model.blocks[layer].ffn, "Wr"): return None
    vals = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            vals.append(routing_overlap(model, domains[i], domains[j], dev, layer))
    return sum(vals) / len(vals) if vals else None

def overlap_per_layer(model, domains, dev):
    """Chevauchement moyen par couche -> teste si les couches profondes
    collisionnent plus (l'overlap n'est plus mesuré que sur la couche 0)."""
    if not hasattr(model.blocks[0].ffn, "Wr"): return None
    return [mean_pairwise_overlap(model, domains, dev, L) for L in range(len(model.blocks))]

# --------------------------- tracé ------------------------------------------
def plot_curves(results, names, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = len(names)
    fig, axes = plt.subplots(1, D, figsize=(5 * D, 4), squeeze=False)
    for j in range(D):
        ax = axes[0][j]
        for mode, r in results.items():
            sp = r["stage_ppl"]                       # {stage: {dom: ppl}}
            xs = [s for s in range(D) if str(j) in sp.get(str(s), {})]
            ys = [sp[str(s)][str(j)] for s in xs]
            if xs:
                ax.plot(xs, ys, marker="o", label=mode)
        ax.set_title(f"domaine {j} ({names[j]})")
        ax.set_xlabel("après entraînement du domaine #")
        ax.set_ylabel("perplexité")
        ax.set_xticks(range(D))
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Perplexité par domaine au fil de la séquence CL "
                 "(hausse sur un domaine ancien = oubli)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=120)
    print(f"[plot] courbe enregistrée -> {out_png}")

def main():
    a = get_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    domains, V, names = load_domains(a, tok_vocab=256)
    print(f"device={dev}  vocab={V}  domaines={len(domains)} {names}  "
          f"shapes={[tuple(d.shape) for d in domains]}\n", flush=True)

    results = {}
    for mode in a.modes:
        print(f"\n========== mode = {mode} ==========", flush=True)
        a.mode = mode
        model = LM(a, V).to(dev)
        # stage_ppl[stage][dom] = ppl du domaine `dom` après avoir entraîné `stage`
        stage_ppl, first_ppl, final_ppl = {}, {}, {}
        seen = []
        for di, d in enumerate(domains):
            # routed_protect : après le 1er domaine, GELER les params partagés
            # contenu-lourds (embedding tokens + tête de sortie) pour tester si
            # l'oubli résiduel passe par eux. Coût attendu : plasticité réduite
            # sur les domaines suivants (compromis isolation/transfert visible).
            if mode == "routed_protect" and di == 1:
                model.tok.weight.requires_grad_(False)
                model.head.weight.requires_grad_(False); model.head.bias.requires_grad_(False)
                print("  [routed_protect] gel de tok-emb + tête après domaine 0", flush=True)
            print(f"  -- entraînement domaine {di} ({names[di]}) --", flush=True)
            train_domain(model, d.to(dev), a, dev, tag=f"[{mode} d{di}]")
            seen.append(di)
            stage_ppl[str(di)] = {}
            for j in seen:                                  # éval sur tout le vu
                p = ppl(model, domains[j], dev, iters=a.eval_iters)
                stage_ppl[str(di)][str(j)] = p
                if j == di: first_ppl[j] = p                # ppl juste après apprentissage
                final_ppl[j] = p                            # ppl courante
            # log de l'état après ce domaine
            line = "  ".join(f"d{j}={stage_ppl[str(di)][str(j)]:.1f}" for j in seen)
            print(f"  >> après domaine {di} | ppl: {line}", flush=True)
        # oubli = hausse moyenne de ppl sur les domaines antérieurs
        forg = [final_ppl[j] / first_ppl[j] for j in range(len(domains) - 1)]
        forg_mean = sum(forg) / len(forg) if forg else None
        ov = mean_pairwise_overlap(model, domains, dev)
        ov_layers = overlap_per_layer(model, domains, dev)
        if ov_layers is not None:
            print("  overlap par couche: " +
                  "  ".join(f"L{L}={v:.1%}" for L, v in enumerate(ov_layers)), flush=True)
        results[mode] = {
            "stage_ppl": stage_ppl,
            "first_ppl": {str(j): first_ppl[j] for j in first_ppl},
            "final_ppl": {str(j): final_ppl[j] for j in final_ppl},
            "forgetting_per_old_domain": {str(j): forg[j] for j in range(len(forg))},
            "forgetting_mean": forg_mean,
            "routing_overlap": ov,
            "routing_overlap_per_layer": ov_layers,
            "final_ppl_last_domain": final_ppl[len(domains) - 1],
        }

    # ----- résumé tabulaire -----
    print("\n================= RÉSUMÉ =================")
    hdr = f"{'mode':>13} | {'ppl finale (dern.)':>18} | {'oubli moyen (anciens)':>21} | {'chevauch. routage':>17}"
    print(hdr); print("-" * len(hdr))
    for mode, r in results.items():
        fm = f"{r['forgetting_mean']:.2f}x" if r["forgetting_mean"] is not None else "n/a"
        ov = f"{r['routing_overlap']:.1%}" if r["routing_overlap"] is not None else "  -  "
        print(f"{mode:>13} | {r['final_ppl_last_domain']:18.1f} | {fm:>21} | {ov:>17}")

    # ----- artefacts -----
    meta = {"args": {k: v for k, v in vars(a).items() if k != "mode"},
            "device": dev, "vocab": V, "domains": names}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"[json] métriques -> {os.path.join(a.out, 'metrics.json')}")
    plot_curves(results, names, os.path.join(a.out, "ppl_curves.png"))

if __name__ == "__main__":
    main()
