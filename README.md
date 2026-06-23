# jepa-physics-bench

Banc d'essai SSL anti-collapse **dans un JEPA** (style LeJEPA / AMI Labs), sur une
physique jouet : des projectiles sous gravité. On sonde la gravité `g` apprise
**sans étiquettes** pour mesurer la qualité de la représentation.

## Idée

- Données = trajectoires de balles (gravité `g` varie, 6 niveaux). Observations =
  projection (capteur fixe). Options réalistes : **rebond** au sol (`--bounce`),
  **vent** turbulent (`--wind`).
- JEPA : contexte masqué → encodeur → prédicteur → prédit la cible propre.
  Terme anti-collapse comparé : `none` / `vicreg` / `sigreg_off` (package officiel
  [lejepa](https://github.com/rbalestr-lab/lejepa)).
- Lecture : `--readout pool` (moyenne temporelle) ou `time` (par instant, façon V-JEPA).
- **Sonde-g** (linéaire, gelée) = qualité ; **rang effectif** = anti-collapse.
- **Oracle** (`--oracle`) = plafond supervisé sur données brutes : « l'info est-elle là ? ».

## Usage Colab (zéro upload)

Ouvre `cl_sslbench_colab.ipynb` directement depuis GitHub dans Colab, puis
re-roule la cellule 1 pour récupérer la dernière version du code (`git pull`).

```bash
python cl_sslbench.py --regs sigreg_off --readout time --bounce 0.5 --wind 1.0 \
    --n_trains 1000 3000 --seeds 3 --steps 1500
python cl_sslbench.py --oracle --bounce 0.5 --wind 1.0   # plafond
```

## Résultat clé

Sur physique réaliste (rebond + vent), SIGReg-SSL apprend une notion **robuste** de
gravité sans jamais voir la réponse, en restant proche du plafond oracle. Un bug de
projection (train/test dans des espaces différents) avait longtemps plafonné le banc
au hasard — corrigé par un capteur `W` fixe partagé train/test.

## V-JEPA fidèle (vidéo réelle + conduite)

`vjepa.py` = architecture **fidèle à la vision LeCun** (I-JEPA / V-JEPA), partagée par
tout le pipeline vidéo :

- **encodeur-contexte sur les tokens VISIBLES uniquement** (pas de mask-token dans
  l'encodeur, contrairement à MAE/BERT) ;
- **prédicteur ATTENTIONNEL** (Transformer) : reçoit les latents visibles + des
  mask-tokens **positionnels** aux positions cibles, prédit leurs latents par attention ;
- **masquage par BLOCS / tubelets** spatio-temporels (+ masquage temporel pour imaginer
  le futur), pas du Bernoulli token-par-token ;
- **multi-masque** (N cibles par clip) ;
- anti-collapse **SIGReg**, **sans EMA, sans stop-grad** (thèse LeJEPA).

Pipeline :

- `driving_transfer.py` — transfert **UCF101 → dashcam Nexar** à **H=96** (12×12 patches),
  encodeur gelé, sonde danger (lin + MLP) vs in-domaine vs hasard.
- `driving_rollout.py` — **prédiction de collision par rollout latent** : on masque le
  futur, le prédicteur l'imagine, une tête lit `[contexte observé ; futur imaginé]`,
  K futurs échantillonnés → risque + heatmap de saillance.
- `drive_plan.py` — **System-2 : planner MPC latent**. World model V-JEPA gelé →
  dynamique action-conditionnée `g(z,a)→z'` + tête danger `c(z)`. À chaque pas, on
  imagine les futurs sous chaque frein candidat et on choisit l'argmin (danger + coût).
  Boucle fermée : **naïf 56 % collisions → réactif 8 % → MPC ~0 %** (compromis
  sécurité/confort réglable par `w_coll`/`w_prog`/`horizon`).

```bash
python driving_transfer.py --n_nexar 400 --ucf_source full --ucf_nclass 50 --ucf_per 40
python driving_rollout.py  --n_nexar 800 --H 96 --patch 8 --T 16 --k_viz 3
python drive_plan.py       --steps 2000 --n_train 600 --n_test 150
```
