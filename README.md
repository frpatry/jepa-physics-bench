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
