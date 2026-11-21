#!/usr/bin/env python3
"""
ÉTUDE histo_jepa (21 nov. 2026) — un JEPA pré-entraîné SANS étiquettes sur « Histopathologic Cancer Detection » (Kaggle), puis ajusté avec les étiquettes.
But : le meilleur AUC avec NOTRE JEPA (appris sur ces images seulement), mesuré sur un test séparé par nous-mêmes ; points secondaires à 1 % et 10 % d'étiquettes.

Étapes (toutes reprennent là où elles s'étaient arrêtées ; réglages dans config.toml) :
  prep     décode les images une fois (tableau sur disque) et fait la séparation entraînement / validation / test (graine fixe, split.csv)
  chrono   chronomètre quelques pas et estime les heures de GPU du plan complet — à lancer AVANT le vrai run
  pre      pré-entraînement de type I-JEPA : blocs masqués tirés partout dans l'image, cibles = latents d'un encodeur-cible ; AUCUNE perte en pixels, AUCUNE étiquette
  ft       entraînements avec étiquettes (ajustement complet, et encodeur gelé + tête), un processus par GPU
  rapport  tableau, courbe et submission.csv (jamais envoyé par le code)
  tout     prep → pre → ft → rapport
Essai de tuyauterie sans GPU ni données :  python histo_jepa.py tout --synth --mini 600 --out /tmp/histo_mini --cache /tmp/histo_mini_cache
"""
import argparse, contextlib, copy, csv, glob, json, math, multiprocessing as mp, os, queue, random, subprocess, sys, threading, time, traceback
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ICI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ICI)
import histo_util as U

MOY, ECT = (0.70, 0.54, 0.69), (0.24, 0.28, 0.21)     # moyenne / écart-type approximatifs des images (rose-violet), pour centrer l'entrée
MINI = dict(modele=dict(largeur=96, profondeur=2, tetes=4),                                         # essai de tuyauterie : tout petit, quelques pas
            pre=dict(pas_total=30, lot=64, echauffement=5, pred_largeur=64, pred_profondeur=1, pred_tetes=4, sauv_min=1),
            ft=dict(pas=[20, 20, 20], graines=[2, 1, 1], lot=32, n_eval=2, lr_jepa_ft=[1e-3], lr_jepa_gel=[1e-3], lr_zero=[1e-3]))


def dit(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def appareil(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def amp(dev): return torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" else contextlib.nullcontext()
def graine_partout(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def sauver(path, obj): torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)
def ecrire_json(path, obj):
    with open(path + ".tmp", "w") as h: json.dump(obj, h, indent=1)
    os.replace(path + ".tmp", path)
def version_code():
    try: return subprocess.check_output(["git", "-C", ICI, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return "inconnue"
def facteur_lr(pas, total, chauffe, fin):               # montée linéaire puis descente en cosinus jusqu'à « fin » (fraction du taux de départ)
    if pas < chauffe: return (pas + 1) / chauffe
    return fin + (1 - fin) * 0.5 * (1 + math.cos(math.pi * min(1.0, (pas - chauffe) / max(1, total - chauffe))))


# ===================== DONNÉES =====================
def trouver_competition():
    for d in ["/kaggle/input/histopathologic-cancer-detection", "/kaggle/input/competitions/histopathologic-cancer-detection"] + \
             [os.path.dirname(f) for p in ("/kaggle/input/*/train_labels.csv", "/kaggle/input/*/*/train_labels.csv") for f in glob.glob(p)]:
        if os.path.isfile(d + "/train_labels.csv") and os.path.isdir(d + "/train"): return d
    raise SystemExit("Données introuvables : ajouter la compétition histopathologic-cancer-detection au kernel (règles acceptées sur Kaggle).")

def _decode(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)

def decoder_tout(chemins, sortie):                      # les images décodées UNE fois dans un grand tableau uint8 sur disque [N, 96, 96, 3]
    arr = np.lib.format.open_memmap(sortie + ".tmp", mode="w+", dtype=np.uint8, shape=(len(chemins), 96, 96, 3)); t0 = time.time()
    with mp.Pool(os.cpu_count()) as pool:
        for i, a in enumerate(pool.imap(_decode, chemins, chunksize=256)):
            arr[i] = a
            if (i + 1) % 50000 == 0: dit(f"  décodé {i + 1}/{len(chemins)} ({(i + 1) / (time.time() - t0):.0f} images/s)")
    arr.flush(); del arr; os.replace(sortie + ".tmp", sortie)

def prep(cfg, D, cache, mini=0, synth=False):
    os.makedirs(cache, exist_ok=True); os.makedirs(D, exist_ok=True); mf = os.path.join(cache, "meta.npz")
    if not os.path.exists(mf):
        rng = np.random.default_rng(cfg["graine"])
        if synth:                                       # fausses images (bruit ; les « malades » ont le centre plus clair) : pour essayer la tuyauterie seulement
            n = mini or 600; nt = max(8, n // 4); y = rng.integers(0, 2, n).astype(np.int8)
            x = rng.integers(0, 200, (n, 96, 96, 3), dtype=np.uint8); x[y == 1, 32:64, 32:64] += 50
            np.save(os.path.join(cache, "train_x.npy"), x); np.save(os.path.join(cache, "test_x.npy"), rng.integers(0, 200, (nt, 96, 96, 3), dtype=np.uint8))
            ids = [f"faux{i:06d}" for i in range(n)]; tids = [f"fauxtest{i:06d}" for i in range(nt)]
        else:
            comp = trouver_competition(); dit("compétition :", comp)
            with open(comp + "/train_labels.csv", newline="") as h: lab = sorted((r["id"], int(r["label"])) for r in csv.DictReader(h))
            tids = sorted(os.path.splitext(f)[0] for f in os.listdir(comp + "/test") if f.endswith(".tif"))
            if mini: lab = [lab[i] for i in sorted(rng.choice(len(lab), min(mini, len(lab)), replace=False))]; tids = tids[:max(8, mini // 4)]
            ids = [i for i, _ in lab]; y = np.array([l for _, l in lab], dtype=np.int8)
            dit(f"décodage : {len(ids)} images d'entraînement, {len(tids)} images de test Kaggle")
            decoder_tout([f"{comp}/train/{i}.tif" for i in ids], os.path.join(cache, "train_x.npy"))
            decoder_tout([f"{comp}/test/{i}.tif" for i in tids], os.path.join(cache, "test_x.npy"))
        np.savez(mf, ids=np.array(ids), y=y, test_ids=np.array(tids))
    m = np.load(mf); ids = [str(i) for i in m["ids"]]; y = m["y"]
    lames, f_lames = (None, None) if synth else U.read_lames(cfg["donnees"]["lames_csv"], ids)
    part = U.make_split(y, cfg["donnees"]["parts"], cfg["graine"], lames)
    sf = os.path.join(D, "split.csv")
    if os.path.exists(sf):                              # déjà faite dans une session précédente : elle doit être IDENTIQUE, sinon les résultats se mélangeraient
        with open(sf, newline="") as h: vieux = {r["id"]: int(r["part"]) for r in csv.DictReader(h)}
        if len(vieux) != len(ids) or any(vieux.get(i) != int(p) for i, p in zip(ids, part)):
            raise SystemExit(f"La séparation recalculée diffère de {sf} (fichier des lames apparu ou disparu ? réglages changés ?). Utiliser un autre --out.")
    else:
        with open(sf + ".tmp", "w", newline="") as h:
            w = csv.writer(h); w.writerow(["id", "label", "part", "lame"]); w.writerows(zip(ids, y.tolist(), part.tolist(), lames or [""] * len(ids)))
        os.replace(sf + ".tmp", sf)
    info = dict(mode="par lame" if lames else "au hasard par classe (pas de fichier des lames : des morceaux d'une même lame peuvent être des deux côtés)", fichier_lames=f_lames,
                n_lames=len(set(lames)) if lames else None, **{U.NOMS_PARTS[k]: dict(n=int((part == k).sum()), part_malades=round(float(y[part == k].mean()), 4)) for k in range(3)})
    ecrire_json(os.path.join(D, "split_info.json"), info); dit("séparation :", json.dumps(info, ensure_ascii=False))
    return info

def charger(cache, D):
    m = np.load(os.path.join(cache, "meta.npz")); ids = [str(i) for i in m["ids"]]
    with open(os.path.join(D, "split.csv"), newline="") as h: p = {r["id"]: int(r["part"]) for r in csv.DictReader(h)}
    return (np.load(os.path.join(cache, "train_x.npy"), mmap_mode="r"), m["y"].astype(np.int64), np.load(os.path.join(cache, "test_x.npy"), mmap_mode="r"),
            [str(i) for i in m["test_ids"]], np.array([p[i] for i in ids], dtype=np.int8))

def lots_pre(X, idx, Xt, lot, seed):                    # lots sans étiquettes : images d'entraînement (+ test Kaggle), tirées sans remise à chaque passage
    rng = np.random.default_rng(seed); n0 = len(idx); n = n0 + (0 if Xt is None else len(Xt))
    while True:
        p = rng.permutation(n)
        for i in range(0, n - lot + 1, lot):
            j = p[i:i + lot]; a = np.sort(idx[j[j < n0]]); b = np.sort(j[j >= n0] - n0)
            yield torch.from_numpy(np.ascontiguousarray(X[a]) if len(b) == 0 else np.concatenate([X[a], Xt[b]]))

def lots_ft(X, y, idx, lot, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(idx)
        for i in range(0, len(p) - lot + 1, lot):
            j = np.sort(p[i:i + lot]); yield torch.from_numpy(np.ascontiguousarray(X[j])), torch.from_numpy(y[j])

def en_avance(gen, n=6):                                # un fil prépare les lots pendant que le GPU travaille
    q = queue.Queue(n)
    def boucle():
        for b in gen: q.put(b)
    threading.Thread(target=boucle, daemon=True).start()
    while True: yield q.get()

# Augmentations, sur le GPU. L'image reste ENTIÈRE (pas de recadrage) : seulement les 8 retournements/rotations de 90° et un léger changement de couleur.
def vers_float(xb): return xb.permute(0, 3, 1, 2).float() / 255
def centrer(x): return (x - x.new_tensor(MOY).view(1, 3, 1, 1)) / x.new_tensor(ECT).view(1, 3, 1, 1)
def vue(x, t):                                          # une des 8 façons de tourner / retourner une image carrée
    x = torch.rot90(x, t % 4, (2, 3)); return x.flip(3) if t >= 4 else x
def augmenter(xb, couleur):
    x = vers_float(xb); B = x.shape[0]; k = torch.randint(0, 8, (B,), device=x.device); out = torch.empty_like(x)
    for t in range(8):
        s = k == t
        if s.any(): out[s] = vue(x[s], t)
    if couleur > 0:
        r = lambda *f: 1 + couleur * (2 * torch.rand(B, *f, device=x.device) - 1)
        m = out.mean((1, 2, 3), keepdim=True)           # contraste, luminosité, puis un petit gain par canal (teinte de la coloration)
        out = (((out - m) * r(1, 1, 1) + m) * r(1, 1, 1) * (1 + 0.5 * (r(3, 1, 1) - 1))).clamp(0, 1)
    return centrer(out)


# ===================== MODÈLE =====================
def pos2d(d, g):                                        # position de chaque jeton, en sinus/cosinus fixes [g·g, d]
    y, x = torch.meshgrid(torch.arange(g, dtype=torch.float32), torch.arange(g, dtype=torch.float32), indexing="ij")
    w = 1.0 / 10000 ** (torch.arange(d // 4, dtype=torch.float32) / (d // 4)); x = x.flatten()[:, None] * w; y = y.flatten()[:, None] * w
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], 1)

def init_poids(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)

class Bloc(nn.Module):
    def __init__(s, d, h, dp=0.0):
        super().__init__(); s.h, s.dp = h, dp
        s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.n2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def saut(s, x):                                     # « drop path » : pendant l'ajustement, une branche est parfois sautée (par image)
        if not s.training or s.dp == 0: return x
        return x * (torch.rand(x.shape[0], 1, 1, device=x.device) >= s.dp).to(x.dtype) / (1 - s.dp)
    def forward(s, x):
        B, N, D = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.saut(s.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)))
        return x + s.saut(s.mlp(s.n2(x)))

class ViT(nn.Module):
    def __init__(s, patch, d, prof, tetes, dp=0.0, img=96):
        super().__init__(); s.g = img // patch; s.d = d
        s.embed = nn.Conv2d(3, d, patch, patch); s.register_buffer("pos", pos2d(d, s.g), persistent=False)
        s.blocks = nn.ModuleList(Bloc(d, tetes, dp * i / max(1, prof - 1)) for i in range(prof)); s.norm = nn.LayerNorm(d); s.apply(init_poids)
    def forward(s, x, garde=None):                      # garde [B, K] : indices des jetons visibles (None = tous)
        x = s.embed(x).flatten(2).transpose(1, 2) + s.pos
        if garde is not None: x = x.gather(1, garde[..., None].expand(-1, -1, x.shape[-1]))
        for b in s.blocks: x = b(x)
        return s.norm(x)

class Predicteur(nn.Module):                            # à partir des jetons visibles, prédit les LATENTS des blocs cachés (chaque bloc-cible séparément)
    def __init__(s, d_enc, d, prof, tetes, g):
        super().__init__(); s.entree = nn.Linear(d_enc, d); s.cache = nn.Parameter(torch.zeros(1, 1, d)); s.register_buffer("pos", pos2d(d, g), persistent=False)
        s.blocks = nn.ModuleList(Bloc(d, tetes) for _ in range(prof)); s.norm = nn.LayerNorm(d); s.sortie = nn.Linear(d, d_enc)
        s.apply(init_poids); nn.init.trunc_normal_(s.cache, std=0.02)
    def forward(s, z, ctx, cib):                        # z [B, Kc, d_enc] ; ctx [B, Kc] ; cib [B, M, Kt] → [B, M, Kt, d_enc]
        B, M, Kt = cib.shape; c = s.entree(z) + s.pos[ctx]
        c = c[:, None].expand(-1, M, -1, -1).reshape(B * M, c.shape[1], c.shape[2])
        x = torch.cat([c, (s.cache + s.pos[cib]).reshape(B * M, Kt, -1).to(c.dtype)], 1)
        for b in s.blocks: x = b(x)
        return s.sortie(s.norm(x[:, -Kt:])).view(B, M, Kt, -1)

class JEPA(nn.Module):                                  # encodeur (jetons visibles) + encodeur-cible (image entière, moyenne glissante, sans gradient) + prédicteur
    def __init__(s, cfg):
        super().__init__(); mo, pc = cfg["modele"], cfg["pre"]; s.couleur = pc["couleur"]
        s.enc = ViT(mo["patch"], mo["largeur"], mo["profondeur"], mo["tetes"]); s.tgt = copy.deepcopy(s.enc).requires_grad_(False)
        s.pred = Predicteur(mo["largeur"], pc["pred_largeur"], pc["pred_profondeur"], pc["pred_tetes"], s.enc.g)
    def forward(s, xb, ctx, cib):                       # xb uint8 [B, 96, 96, 3] → perte par image [B], et dispersion des latents-cibles [1] (≈ 0 = effondrement)
        with amp(xb.device):
            x = augmenter(xb, s.couleur); B, M, Kt = cib.shape
            with torch.no_grad():
                h = F.layer_norm(s.tgt(x).float(), (s.enc.d,))
                y = h.gather(1, cib.reshape(B, M * Kt)[..., None].expand(-1, -1, h.shape[-1])).view(B, M, Kt, -1)
            p = s.pred(s.enc(x, ctx), ctx, cib)
            perte = F.smooth_l1_loss(p.float(), y, reduction="none").mean((1, 2, 3))      # la cible est dans les latents, jamais dans les pixels
        return perte, h.mean(1).std(0).mean()[None]

class Tete(nn.Module):                                  # lecture : une requête apprise regarde tous les jetons (elle apprend seule où regarder), puis une sortie « malade / sain »
    def __init__(s, d, h):
        super().__init__(); s.q = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(s.q, std=0.02)
        s.n = nn.LayerNorm(d); s.att = nn.MultiheadAttention(d, h, batch_first=True); s.n2 = nn.LayerNorm(d); s.sortie = nn.Linear(d, 1)
    def forward(s, z):
        z = s.n(z.float()); a, _ = s.att(s.q.expand(z.shape[0], -1, -1), z, z, need_weights=False)
        return s.sortie(s.n2(a[:, 0]))[:, 0]

def masques(B, g, pc, dev):
    """Blocs-cibles et contexte, comme I-JEPA : tailles tirées une fois par lot, positions tirées par image, PARTOUT dans la grille (aucune zone privilégiée).
    → ctx [B, K] jetons visibles (jamais dans une cible), cib [B, M, th·tw] jetons à prédire."""
    def taille(ech, rat):
        a = random.uniform(*ech) * g * g; r = random.uniform(*rat)
        return max(1, min(g, round(math.sqrt(a * r)))), max(1, min(g, round(math.sqrt(a / r))))
    M = pc["n_cibles"]; th, tw = taille(pc["cible_echelle"], pc["cible_ratio"]); ch, cw = taille(pc["ctx_echelle"], (1.0, 1.0)); ar = lambda n: torch.arange(n, device=dev)
    haut = torch.randint(0, g - th + 1, (B, M, 1, 1), device=dev); gauche = torch.randint(0, g - tw + 1, (B, M, 1, 1), device=dev)
    cib = ((haut + ar(th).view(1, 1, th, 1)) * g + gauche + ar(tw).view(1, 1, 1, tw)).flatten(2)
    est_cible = torch.zeros(B, g * g, device=dev).scatter_(1, cib.flatten(1), 1.0) > 0
    ch0 = torch.randint(0, g - ch + 1, (B, 1, 1), device=dev); cg0 = torch.randint(0, g - cw + 1, (B, 1, 1), device=dev); yy = ar(g).view(1, g, 1); xx = ar(g).view(1, 1, g)
    est_ctx = ((yy >= ch0) & (yy < ch0 + ch) & (xx >= cg0) & (xx < cg0 + cw)).flatten(1) & ~est_cible
    K = max(pc["ctx_min"], int(est_ctx.sum(1).min()))   # même nombre de jetons visibles pour tout le lot ; s'il en manque, on complète hors des cibles
    note = torch.rand(B, g * g, device=dev) + 2.0 * est_ctx + 1.0 * (~est_cible)
    return note.topk(K, dim=1).indices.sort(1).values, cib

def groupes(blocs, lr, wd):                             # blocs = [(paramètres, échelle du taux)] ; pas de « weight decay » sur les biais, normes et jetons appris
    G = []
    for params, ech in blocs:
        params = [p for p in params if p.requires_grad]
        for ps, w in (([p for p in params if p.ndim in (2, 4)], wd), ([p for p in params if p.ndim not in (2, 4)], 0.0)):
            if ps: G.append(dict(params=ps, weight_decay=w, lr=lr * ech, base=lr * ech))
    return G

def echelle_grad(dev):
    try: return torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")


# ===================== PRÉ-ENTRAÎNEMENT (sans étiquettes) =====================
def pretrain(cfg, D, cache, fin, chrono=0):
    """→ True si le pré-entraînement est complet. chrono > 0 : fait seulement ce nombre de pas, sans rien lire ni sauver, et renvoie les secondes par pas."""
    pc = cfg["pre"]; dev = appareil(); graine_partout(cfg["graine"]); X, y, Xt, tids, part = charger(cache, D)
    idx = np.flatnonzero(part == 0); Xt = Xt if cfg["donnees"]["test_kaggle_dans_pre"] else None      # jamais notre validation ni notre test
    n = len(idx) + (0 if Xt is None else len(Xt)); lot = min(pc["lot"], n); total = pc["pas_total"]; ck = os.path.join(D, "pre.pt"); pas = 0
    model = JEPA(cfg).to(dev); opt = torch.optim.AdamW(groupes([(list(model.enc.parameters()) + list(model.pred.parameters()), 1.0)], pc["lr"], pc["wd"]), betas=(0.9, 0.95))
    scaler = echelle_grad(dev)
    if not chrono and os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False); model.load_state_dict(s["modele"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"]); pas = s["pas"]
        dit(f"pré-entraînement : REPRISE au pas {pas}/{total}")
        if pas >= total: return True
    par = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    dit(f"pré-entraînement : {n} images sans étiquettes ({len(idx)} d'entraînement + {n - len(idx)} de test Kaggle), lot {lot}, {max(1, torch.cuda.device_count())} GPU/CPU, "
        f"{sum(p.numel() for p in model.enc.parameters()) / 1e6:.1f} M de paramètres (encodeur)")
    it = en_avance(lots_pre(X, idx, Xt, lot, cfg["graine"] + pas)); g = model.enc.g; cible = chrono or total; t0 = t_sauv = time.time(); p0 = pas; t_mi = None; cumul = []
    while pas < cible and time.time() < fin:
        xb = next(it).to(dev, non_blocking=True); ctx, cib = masques(xb.shape[0], g, pc, dev)
        f = facteur_lr(pas, total, pc["echauffement"], pc["lr_fin"])
        for gr in opt.param_groups: gr["lr"] = gr["base"] * f
        perte, disp = par(xb, ctx, cib); perte = perte.mean()
        if not torch.isfinite(perte): raise SystemExit(f"pré-entraînement : perte non finie au pas {pas} — arrêt sans sauvegarde")
        opt.zero_grad(set_to_none=True); scaler.scale(perte).backward(); scaler.step(opt); scaler.update()
        m = pc["ema"] + (1 - pc["ema"]) * pas / total
        with torch.no_grad():
            for pt, pe in zip(model.tgt.parameters(), model.enc.parameters()): pt.mul_(m).add_(pe.detach(), alpha=1 - m)
        pas += 1; cumul.append((perte.item(), disp.mean().item()))
        if chrono and pas == chrono // 2:
            if dev.type == "cuda": torch.cuda.synchronize()
            t_mi = time.time()
        if pas % 100 == 0 or pas == cible:
            v = (pas - p0) * lot / (time.time() - t0); a = np.mean(cumul, 0); cumul = []
            dit(f"  pas {pas}/{total}  perte {a[0]:.4f}  dispersion des cibles {a[1]:.3f}  taux {pc['lr'] * f:.2e}  {v:.0f} images/s  reste {(total - pas) * lot / v / 3600:.1f} h")
            if not chrono:
                with open(os.path.join(D, "pre_journal.csv"), "a") as h: h.write(f"{pas},{a[0]:.5f},{a[1]:.4f}\n")
        if not chrono and (time.time() - t_sauv > 60 * pc["sauv_min"] or pas == total):
            sauver(ck, dict(modele=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), pas=pas, cfg=cfg)); t_sauv = time.time()
    if chrono:
        if dev.type == "cuda": torch.cuda.synchronize()
        return (time.time() - t_mi) / max(1, pas - chrono // 2)
    sauver(ck, dict(modele=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), pas=pas, cfg=cfg))
    dit(f"pré-entraînement : {'FINI' if pas >= total else 'interrompu (temps écoulé)'} au pas {pas}/{total}"); return pas >= total


# ===================== AVEC ÉTIQUETTES =====================
@torch.no_grad()
def predire(enc, tete, X, idx, dev, huit_vues=False, lot=1024):
    enc.eval(); tete.eval(); out = []
    for i in range(0, len(idx), lot):
        x = centrer(vers_float(torch.from_numpy(np.ascontiguousarray(X[idx[i:i + lot]])).to(dev)))
        with amp(dev): out.append((sum(tete(enc(vue(x, t))) for t in range(8)) / 8 if huit_vues else tete(enc(x))).float().cpu())
    return torch.cat(out).numpy()

def run_job(job, cfg, D, cache, dev, fin, chrono=0):
    """Un entraînement avec étiquettes. Écrit pred/<id>.npz (notes sur validation, notre test, test Kaggle) et res/<id>.json.
    L'AUC du test n'est PAS calculée ici : elle l'est une seule fois, dans le rapport, après le choix des réglages sur la validation."""
    t0 = time.time(); jid = U.job_id(job); ft, mo = cfg["ft"], cfg["modele"]; graine_partout(cfg["graine"] * 1000 + job["graine"]); X, y, Xt, tids, part = charger(cache, D)
    tr = U.label_subset(np.flatnonzero(part == 0), y, job["frac"], job["graine"]); va = np.flatnonzero(part == 1); te = np.flatnonzero(part == 2)
    gel = job["bras"] == "jepa_gel"; enc = ViT(mo["patch"], mo["largeur"], mo["profondeur"], mo["tetes"], dp=0.0 if gel else ft["drop_path"]).to(dev)
    f_pre = os.path.join(D, "pre.pt"); pas_pre = 0
    if job["bras"] != "zero" and not (chrono and not os.path.exists(f_pre)):
        s = torch.load(f_pre, map_location="cpu", weights_only=False); pas_pre = s["pas"]
        enc.load_state_dict({k[4:]: v for k, v in s["modele"].items() if k.startswith("tgt.")})          # on part de l'encodeur-cible (le plus stable)
    tete = Tete(mo["largeur"], mo["tetes"]).to(dev); L = len(enc.blocks) + 1; dc = ft["decroissance_couches"]
    if gel: enc.requires_grad_(False); blocs = [(list(tete.parameters()), 1.0)]
    else: blocs = [(list(enc.embed.parameters()), dc ** L)] + [(list(b.parameters()), dc ** (L - 1 - i)) for i, b in enumerate(enc.blocks)] + [(list(enc.norm.parameters()) + list(tete.parameters()), 1.0)]
    opt = torch.optim.AdamW(groupes(blocs, job["lr"], ft["wd"])); scaler = echelle_grad(dev); params = [p for g in opt.param_groups for p in g["params"]]
    n_pas = chrono or ft["pas"][ft["fractions"].index(job["frac"])]; lot = min(ft["lot"], len(tr)); it = en_avance(lots_ft(X, y, tr, lot, 7 + job["graine"]))
    chauffe = max(1, int(ft["echauffement"] * n_pas)); ev = max(1, n_pas // ft["n_eval"]); best, best_etat, sans, pas, t_mi = -1.0, None, 0, 0, None
    etat = lambda: copy.deepcopy(dict(enc=None if gel else enc.state_dict(), tete=tete.state_dict()))
    while pas < n_pas:
        if time.time() > fin: dit(f"  {jid} : abandonné (temps écoulé), sera refait"); return None
        enc.train(not gel); tete.train(); xb, yb = next(it); xb = xb.to(dev, non_blocking=True); yb = yb.to(dev).float()
        f = facteur_lr(pas, n_pas, chauffe, 0.0)
        for g in opt.param_groups: g["lr"] = g["base"] * f
        with amp(dev):
            x = augmenter(xb, ft["couleur"])
            if gel:
                with torch.no_grad(): z = enc(x)
            else: z = enc(x)
            perte = F.binary_cross_entropy_with_logits(tete(z).float(), yb)
        opt.zero_grad(set_to_none=True); scaler.scale(perte).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); pas += 1
        if chrono:
            if pas == chrono // 2:
                if dev.type == "cuda": torch.cuda.synchronize()
                t_mi = time.time()
            continue
        if pas % ev == 0 or pas == n_pas:
            a = U.auc(y[va], predire(enc, tete, X, va, dev))
            if a > best or best_etat is None: best, best_etat, sans = a, etat(), 0
            else: sans += 1
            dit(f"  {jid} : pas {pas}/{n_pas}  perte {perte.item():.4f}  AUC validation {a:.4f}  (meilleure {best:.4f})")
            if sans >= ft["patience"]: break
    if chrono:
        if dev.type == "cuda": torch.cuda.synchronize()
        return (time.time() - t_mi) / max(1, pas - chrono // 2)
    if best_etat["enc"] is not None: enc.load_state_dict(best_etat["enc"])
    tete.load_state_dict(best_etat["tete"])
    P = dict(val=predire(enc, tete, X, va, dev), val_y=y[va], test=predire(enc, tete, X, te, dev), test_tta=predire(enc, tete, X, te, dev, True), test_y=y[te])
    if job["bras"] == ft["bras_soumission"] and job["frac"] >= 1.0: P["kaggle"] = predire(enc, tete, Xt, np.arange(len(Xt)), dev, True)
    os.makedirs(os.path.join(D, "pred"), exist_ok=True); os.makedirs(os.path.join(D, "res"), exist_ok=True)
    f_np = os.path.join(D, "pred", jid); np.savez(f_np + ".tmp.npz", **P); os.replace(f_np + ".tmp.npz", f_np + ".npz")
    r = dict(id=jid, **job, val_auc=float(best), n_etiquettes=int(len(tr)), pas=pas, pas_pre=pas_pre, minutes=round((time.time() - t0) / 60, 1), code=version_code(), config=U.cfg_hash(cfg))
    ecrire_json(os.path.join(D, "res", jid + ".json"), r); dit(f"  {jid} : FINI, AUC validation {best:.4f}, {r['minutes']} min"); return r

def prendre(cache, jid):                                # un entraînement n'est fait que par un processus à la fois (verrou = petit fichier avec le n° du processus)
    d = os.path.join(cache, "verrous"); os.makedirs(d, exist_ok=True); f = os.path.join(d, jid)
    for _ in range(2):
        try:
            h = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(h, str(os.getpid()).encode()); os.close(h); return True
        except FileExistsError:
            try: os.kill(int(open(f).read() or 0), 0); return False          # son propriétaire travaille encore
            except (OSError, ValueError):
                with contextlib.suppress(OSError): os.remove(f)              # verrou d'un processus mort : on le reprend
    return False

def ouvrier(cfg, D, cache, fin):
    dev = appareil(); rates = set()
    while time.time() < fin:
        todo, fini = U.plan_jobs(cfg, D)
        if fini: return True
        todo = [j for j in todo if U.job_id(j) not in rates]
        if not todo: dit("ouvrier : des entraînements ont échoué, arrêt"); return False
        job = next((j for j in todo if prendre(cache, U.job_id(j))), None)
        if job is None: time.sleep(20); continue        # l'autre GPU finit la phase 1 : on attend
        try:
            if run_job(job, cfg, D, cache, dev, fin) is None: break
        except Exception:
            traceback.print_exc(); rates.add(U.job_id(job))
        finally:
            with contextlib.suppress(OSError): os.remove(os.path.join(cache, "verrous", U.job_id(job)))
    return False

def etape_ft(args, cfg, D, cache, fin):
    if not os.path.exists(os.path.join(D, "pre.pt")) and any(b != "zero" for b in cfg["ft"]["bras"]): raise SystemExit("pas de pré-entraînement (pre.pt) : lancer l'étape « pre » d'abord")
    n = torch.cuda.device_count()
    if n <= 1: return ouvrier(cfg, D, cache, fin)
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "_ouvrier", "--config", args.config, "--out", D, "--cache", cache, "--fin", str(fin)] + (["--mini", str(args.mini)] if args.mini else [])
    ps = [subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))) for i in range(n)]      # un processus par GPU
    return all([p.wait() == 0 for p in ps]) and U.plan_jobs(cfg, D)[1]


# ===================== CHRONOMÉTRAGE =====================
def chrono(cfg, D, cache, n=60):
    """Mesure quelques pas de chaque sorte et estime les heures du plan complet. Ne sauve aucun modèle."""
    dev = appareil(); ng = max(1, torch.cuda.device_count()); ft = cfg["ft"]; loin = time.time() + 1e9; X, y, Xt, tids, part = charger(cache, D)
    s_pre = pretrain(cfg, D, cache, loin, chrono=n); h_pre = s_pre * cfg["pre"]["pas_total"] / 3600
    dit(f"CHRONO pré-entraînement : {s_pre:.3f} s/pas ({cfg['pre']['lot'] / s_pre:.0f} images/s sur {ng} GPU) → {cfg['pre']['pas_total']} pas = {h_pre:.1f} h")
    mo = cfg["modele"]; enc = ViT(mo["patch"], mo["largeur"], mo["profondeur"], mo["tetes"]).to(dev); tete = Tete(mo["largeur"], mo["tetes"]).to(dev); idx = np.arange(min(4096, len(X)))
    predire(enc, tete, X, idx[:256], dev); t = time.time(); predire(enc, tete, X, idx, dev)
    if dev.type == "cuda": torch.cuda.synchronize()
    v = len(idx) / (time.time() - t); del enc, tete; dit(f"CHRONO prédiction : {v:.0f} images/s (une vue, un GPU)")
    nv, nt = int((part == 1).sum()), int((part == 2).sum()); h_ft = 0.0
    for bras in ft["bras"]:
        s = run_job(dict(bras=bras, frac=1.0, graine=0, lr=1e-4), cfg, D, cache, dev, loin, chrono=n)
        for frac, g, pas in zip(ft["fractions"], ft["graines"], ft["pas"]):
            runs = len(ft["lr_" + bras]) + g - 1; kag = 8 * len(Xt) if (bras == ft["bras_soumission"] and frac >= 1.0) else 0
            h = runs * (pas * s + (ft["n_eval"] * nv + nv + 9 * nt + kag) / v) / 3600; h_ft += h
            dit(f"CHRONO {bras} à {100 * frac:g} % : {s:.3f} s/pas × {pas} pas × {runs} entraînements ≈ {h:.1f} h de GPU (au plus : l'arrêt anticipé raccourcit)")
    tot = h_pre + h_ft / ng
    dit(f"CHRONO TOTAL ≈ {h_pre:.1f} h (pré-entraînement) + {h_ft / ng:.1f} h (avec étiquettes, {ng} GPU en parallèle) = {tot:.1f} h de session Kaggle")
    ecrire_json(os.path.join(D, "chrono.json"), dict(s_pas_pre=s_pre, h_pre=h_pre, images_s_prediction=v, h_ft_gpu=h_ft, n_gpu=ng, h_total=tot, config=U.cfg_hash(cfg), code=version_code()))


# ===================== ENTRÉE =====================
def main():
    ap = argparse.ArgumentParser(description="Étude histo_jepa (voir l'en-tête du fichier).")
    ap.add_argument("etape", choices=["prep", "chrono", "pre", "ft", "rapport", "tout", "_ouvrier"])
    ap.add_argument("--config", default=os.path.join(ICI, "config.toml")); ap.add_argument("--out", default="/kaggle/working/histo_runs"); ap.add_argument("--cache", default="/tmp/histo_cache")
    ap.add_argument("--heures", type=float, default=11.0, help="durée maximale de cette session ; le travail reprend à la suivante")
    ap.add_argument("--fin", type=float, default=0.0, help=argparse.SUPPRESS); ap.add_argument("--mini", type=int, default=0, help="essai : N images seulement, modèle minuscule, quelques pas")
    ap.add_argument("--synth", action="store_true", help="essai sur de fausses images (sans la compétition)")
    a = ap.parse_args(); cfg = U.load_cfg(a.config); D, cache = a.out, a.cache
    if a.mini:
        for k, v in MINI.items(): cfg[k].update(v)
    fin = a.fin or time.time() + 3600 * a.heures - 600  # 10 minutes gardées pour le rapport
    if a.etape == "_ouvrier": sys.exit(0 if ouvrier(cfg, D, cache, fin) else 1)
    dit(f"étape {a.etape} — code {version_code()}, config {U.cfg_hash(cfg)}, sortie {D}")
    if a.etape in ("prep", "chrono", "pre", "ft", "tout"): info = prep(cfg, D, cache, a.mini, a.synth)
    if a.etape in ("pre", "ft", "tout"):                # les résultats d'un dossier viennent tous des MÊMES réglages
        f = os.path.join(D, "config_utilisee.json")
        if os.path.exists(f) and json.load(open(f)) != cfg: raise SystemExit(f"config.toml a changé depuis les résultats de {D} : utiliser un autre --out (ou effacer ce dossier).")
        ecrire_json(f, cfg)
    if a.etape == "chrono": chrono(cfg, D, cache)
    ok = True
    if a.etape in ("pre", "tout"):
        ok = pretrain(cfg, D, cache, fin)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if a.etape == "ft" or (a.etape == "tout" and ok): ok = etape_ft(a, cfg, D, cache, fin)
    if a.etape in ("ft", "rapport", "tout"):
        meta = dict(code=version_code(), config=U.cfg_hash(cfg)); f = os.path.join(D, "split_info.json"); mf = os.path.join(cache, "meta.npz")
        if os.path.exists(f): meta["séparation"] = json.load(open(f))["mode"]
        if os.path.exists(os.path.join(D, "pre.pt")): meta["pas de pré-entraînement"] = torch.load(os.path.join(D, "pre.pt"), map_location="cpu", weights_only=False)["pas"]
        U.report(cfg, D, meta, [str(i) for i in np.load(mf)["test_ids"]] if os.path.exists(mf) else None)
    if a.etape in ("pre", "ft", "tout"): dit("TOUT EST FINI" if ok and a.etape != "pre" else "pré-entraînement fini" if ok else "PAS FINI : relancer la même commande pour continuer")


if __name__ == "__main__":
    main()
