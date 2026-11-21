"""
Outils SANS réseau de neurones (numpy seulement) de l'étude histo_jepa : configuration, séparation des données,
sous-ensembles d'étiquettes, AUC, liste des entraînements à faire, rapport final.
Séparé de histo_jepa.py pour pouvoir être vérifié sans PyTorch.
"""
import csv, glob, hashlib, json, os
import numpy as np

NOMS_PARTS = ["entrainement", "validation", "test"]


def load_cfg(path):
    import tomllib                                   # Python ≥ 3.11 (c'est le cas sur Kaggle)
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def auc(y, s):
    """AUC = chance qu'une image malade reçoive une note plus haute qu'une image saine (ex æquo comptés 1/2)."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    debut = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]; fin = np.r_[debut[1:], len(s)]
    rang = np.empty(len(s)); rang[o] = np.repeat((debut + fin - 1) / 2.0 + 1.0, fin - debut)     # rang moyen pour les ex æquo
    return float((rang[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def make_split(labels, parts, seed, lames=None):
    """→ part[i] ∈ {0 entraînement, 1 validation, 2 test}, faite une seule fois avec une graine fixe.
    Sans lames : tirage au hasard, classe par classe. Avec lames : une lame entière va d'un seul côté (pas de fuite entre voisins)."""
    labels = np.asarray(labels); rng = np.random.default_rng(seed); n = len(labels); part = np.zeros(n, dtype=np.int8)
    if lames is None:
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            a = int(round(parts[0] * len(idx))); b = int(round((parts[0] + parts[1]) * len(idx)))
            part[idx[a:b]] = 1; part[idx[b:]] = 2
        return part
    _, inv, cnt = np.unique(np.asarray(lames), return_inverse=True, return_counts=True)
    cible = np.asarray(parts, dtype=np.float64) * n; pris = np.zeros(3); gpart = np.zeros(len(cnt), dtype=np.int8)
    for g in rng.permutation(len(cnt)):              # chaque lame va du côté où il manque le plus d'images
        k = int(np.argmax((cible - pris) / cible)); gpart[g] = k; pris[k] += cnt[g]
    return gpart[inv]


def label_subset(train_idx, labels, frac, seed):
    """Sous-ensemble d'étiquettes, par classe, EMBOÎTÉ : à graine égale le 1 % est dans le 10 %, lui-même dans le 100 %."""
    train_idx = np.asarray(train_idx)
    if frac >= 1.0:
        return np.sort(train_idx)
    rng = np.random.default_rng(1000 + seed); out = []
    for c in np.unique(labels[train_idx]):
        idx = rng.permutation(train_idx[labels[train_idx] == c])
        out.append(idx[:max(1, int(round(frac * len(idx))))])
    return np.sort(np.concatenate(out))


def read_lames(patterns, ids):
    """Cherche un fichier public « image → lame ». → (liste des lames dans l'ordre de ids, chemin) ou (None, None)."""
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            with open(f, newline="") as h:
                rd = csv.DictReader(h); cols = rd.fieldnames or []
                col = next((c for c in cols if any(k in c.lower() for k in ("wsi", "slide", "lame"))), None)
                if "id" not in cols or col is None:
                    continue
                m = {r["id"]: r[col] for r in rd}
            connus = sum(1 for i in ids if i in m)
            print("fichier des lames : {} — {} images reconnues sur {}".format(f, connus, len(ids)), flush=True)
            if connus < 0.5 * len(ids):
                continue
            return [m.get(i, "seule_" + i) for i in ids], f           # image inconnue = traitée comme une lame à elle seule
    return None, None


# ---------- liste des entraînements avec étiquettes ----------
def fractions_du_bras(ft, bras):
    """→ [(fraction, nombre de graines, pas)] pour ce bras ; ft["fractions_<bras>"] (facultatif) restreint ses fractions."""
    permis = ft.get("fractions_" + bras, ft["fractions"])
    return [(f, g, p) for f, g, p in zip(ft["fractions"], ft["graines"], ft["pas"]) if f in permis]


def job_id(j):
    return "{}_f{:g}_g{}_lr{:g}".format(j["bras"], j["frac"], j["graine"], j["lr"])


def read_results(D):
    R = {}
    for f in glob.glob(os.path.join(D, "res", "*.json")):
        with open(f) as h:
            r = json.load(h)
        R[r["id"]] = r
    return R


def plan_jobs(cfg, D):
    """→ (entraînements à faire maintenant, tout est fini ?).
    Phase 1 : graine 0 avec chaque taux d'apprentissage. Phase 2, quand la phase 1 du couple (bras, fraction) est finie :
    les autres graines avec le meilleur taux, choisi sur la VALIDATION (jamais sur le test)."""
    R = read_results(D); ft = cfg["ft"]; todo = []; fini = True
    for frac in ft["fractions"]:                     # les petites fractions d'abord, pour tous les bras
        for bras in ft["bras"]:
            ng = next((g for f, g, _ in fractions_du_bras(ft, bras) if f == frac), None)
            if ng is None:
                continue
            p1 = [dict(bras=bras, frac=frac, graine=0, lr=lr) for lr in ft["lr_" + bras]]
            manque = [j for j in p1 if job_id(j) not in R]
            if not manque:
                lr = max(p1, key=lambda j: R[job_id(j)]["val_auc"])["lr"]
                manque = [j for j in (dict(bras=bras, frac=frac, graine=g, lr=lr) for g in range(1, ng)) if job_id(j) not in R]
            if manque:
                todo += manque; fini = False
    return todo, fini


# ---------- rapport ----------
def report(cfg, D, meta=None, test_ids=None):
    """Lit les résultats, choisit le taux d'apprentissage sur la validation, et SEULEMENT ALORS calcule l'AUC sur notre test.
    Écrit resultats.csv, resultats.md, courbe.png et submission.csv (prédictions du test Kaggle ; jamais envoyé par le code)."""
    R = read_results(D); ft = cfg["ft"]; rows = []; soumission = None
    for bras in ft["bras"]:
        for frac, ng, _ in fractions_du_bras(ft, bras):
            p1 = [R.get(job_id(dict(bras=bras, frac=frac, graine=0, lr=lr))) for lr in ft["lr_" + bras]]
            faits = [r for r in p1 if r]
            if not faits:
                continue
            lr = max(faits, key=lambda r: r["val_auc"])["lr"]
            runs = sorted((r for r in R.values() if r["bras"] == bras and r["frac"] == frac and r["lr"] == lr), key=lambda r: r["graine"])
            P = [np.load(os.path.join(D, "pred", r["id"] + ".npz")) for r in runs]
            y = P[0]["test_y"]
            a = [auc(y, p["test"]) for p in P]; at = [auc(y, p["test_tta"]) for p in P]
            rows.append(dict(bras=bras, fraction=frac, lr=lr, n_graines=len(runs), complet=int(len(faits) == len(p1) and len(runs) == ng),
                             val_auc=float(np.mean([r["val_auc"] for r in runs])),
                             test_auc=float(np.mean(a)), test_auc_ecart=float(np.std(a)),
                             test_auc_8vues=float(np.mean(at)), test_auc_8vues_ecart=float(np.std(at)),
                             test_auc_ensemble=auc(y, np.mean([p["test_tta"] for p in P], axis=0))))
            if bras == ft["bras_soumission"] and frac >= 1.0 and all("kaggle" in p.files for p in P):
                soumission = 1.0 / (1.0 + np.exp(-np.mean([p["kaggle"] for p in P], axis=0)))
    if not rows:
        print("rapport : aucun résultat pour l'instant", flush=True); return rows
    cols = list(rows[0].keys())
    with open(os.path.join(D, "resultats.csv"), "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols); w.writeheader(); w.writerows(rows)
    L = ["# Résultats — JEPA sur Histopathologic Cancer Detection", ""]
    for k, v in (meta or {}).items():
        L.append("- {} : {}".format(k, v))
    L += ["", "AUC sur NOTRE test (jamais vu pendant l'entraînement ni pour choisir les réglages). « 8 vues » = moyenne des prédictions sur les 8 retournements/rotations.",
          "", "| bras | étiquettes | taux choisi | graines | AUC validation | AUC test | AUC test 8 vues | AUC test, moyenne des graines | complet |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| {} | {:g} % | {:g} | {} | {:.4f} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:.4f} | {} |".format(
            r["bras"], 100 * r["fraction"], r["lr"], r["n_graines"], r["val_auc"], r["test_auc"], r["test_auc_ecart"],
            r["test_auc_8vues"], r["test_auc_8vues_ecart"], r["test_auc_ensemble"], "oui" if r["complet"] else "NON"))
    L += ["", "Bras : jepa_ft = pré-entraîné puis ajusté en entier ; jepa_gel = encodeur pré-entraîné gelé + petite tête ; zero = le même modèle sans pré-entraînement (repère).",
          "", "Limites : notre test est pris dans le « train » Kaggle — le test Kaggle vient d'autres lames et son score sera sans doute plus bas ;",
          "le repère « zero » est un ViT appris de zéro, faible quand les étiquettes sont rares : un écart dit « le pré-entraînement aide ce modèle », pas « il bat tout » ;",
          "le calcul sur GPU n'est pas reproductible au chiffre près."]
    with open(os.path.join(D, "resultats.md"), "w") as h:
        h.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for bras in ft["bras"]:
            rr = [r for r in rows if r["bras"] == bras]
            if rr:
                ax.errorbar([100 * r["fraction"] for r in rr], [r["test_auc_8vues"] for r in rr], yerr=[r["test_auc_8vues_ecart"] for r in rr], marker="o", capsize=3, label=bras)
        ax.set_xscale("log"); ax.set_xticks([100 * f for f in ft["fractions"]]); ax.set_xticklabels(["{:g} %".format(100 * f) for f in ft["fractions"]])
        ax.set_xlabel("part des étiquettes utilisée"); ax.set_ylabel("AUC sur notre test (8 vues)"); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(D, "courbe.png"), dpi=150); plt.close(fig)
    except ImportError:
        print("matplotlib absent : pas de courbe", flush=True)
    if soumission is not None and test_ids is not None and len(test_ids) == len(soumission):
        with open(os.path.join(D, "submission.csv"), "w", newline="") as h:
            w = csv.writer(h); w.writerow(["id", "label"]); w.writerows(zip(test_ids, ("{:.6f}".format(p) for p in soumission)))
        print("submission.csv écrit ({} lignes) — non envoyé".format(len(soumission)), flush=True)
    return rows
