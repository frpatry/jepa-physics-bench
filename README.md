# jepa-physics-bench — un « bébé » JEPA qui apprend en jouant

Projet de recherche personnel, en français, sur les **modèles du monde de type JEPA**
(Joint-Embedding Predictive Architecture), dans l'esprit de la thèse de Yann LeCun
(LeJEPA, V-JEPA 2, AMI Labs) : apprendre sans étiquettes en **prédisant dans un espace
latent**, puis **agir en planifiant dans son imagination**.

Le nom du dépôt vient de sa première expérience (un banc d'essai SSL sur une physique
jouet). Le projet a beaucoup évolué depuis : voir [l'historique](docs/HISTORIQUE.md).

## Le but

Construire un « bébé » artificiel, 100 % auto-supervisé, qui :

1. apprend à **voir, entendre, sentir (toucher, sens du bras) et bouger** dans un petit
   monde simulé, en babillant, sans jamais recevoir d'étiquette ;
2. se construit ainsi un modèle du monde (« si je fais ce geste, voici ce que je verrai,
   entendrai, sentirai ») ;
3. accomplit ensuite une **tâche qu'il n'a jamais apprise** (pousser un objet sur une
   cible) en **imaginant** les conséquences de gestes candidats et en choisissant le
   meilleur (planification MPC dans l'imagination).

Le fil conducteur est **développemental** : chaque choix (ordre des étapes, vue floue au
début, la main avant les objets, etc.) doit pouvoir se rattacher à ce que l'on sait du
nourrisson. Les recherches sourcées sont dans
[recherche_bebe_mains.md](recherche_bebe_mains.md) et
[recherche_bebe_objets.md](recherche_bebe_objets.md). Les étiquettes ne servent
**que d'instruments de mesure** (sondes, oracles), jamais à l'apprentissage.

## Où en est le projet (9 octobre 2026)

**Phase 0 « mes mains »** (`av_world0.py`, `av_phase0.py`) : le bébé découvre son corps
avant les objets. Quatre sens entrent dans l'encodeur (vue 32×32, ouïe stéréo, toucher
sur 4 côtés de la main, sens du bras) ; la **commande** (le geste voulu) n'entre que dans
le prédicteur. Le monde se complexifie par étapes (0a bras → 0b mains → 0c mobile relié
au geste → 0d contact avec des objets → 0e objets hors de portée), avec une vue qui
passe du flou au net et un monde « lisible » (objets au repos par défaut, pauses).

- **Acquis** : connaissance du corps propre (il prédit nettement mieux quand on lui donne
  ses vrais gestes : erreur sur le sens du bras +106 % si on lui ment), lien vu/senti
  (100 % de surprise quand le sens du bras ne correspond pas à l'image), toucher
  (88 à 98 % de surprise quand un contact visible n'est pas senti, selon les runs),
  geste deviné par la vision (R² ≈ 0,8). Dans l'imagination, **la main** est bien
  prédite (2,7 px contre 4,1 px pour « rien ne bouge »).
- **Verrou** : **l'objet n'est pas gardé dans l'imagination**. Un objet immobile
  « dérive » d'environ 10 px en quelques images imaginées, l'objet poussé est plus mal
  prédit que par la simple copie de l'image. Résultat : la planification « pousse
  l'objet sur la cible » reste **au niveau du hasard, voire en dessous** (distance
  finale 0,27 contre 0,21 au hasard et 0,11 pour un oracle). Cinq variantes
  d'entraînement (prédicteur résiduel, contraste d'action, monde calme, table qui
  déborde, séquences de 32 images) n'ont pas réglé le problème.
- **En cours** : les **boîtes-objets** (`av_slots0.py`). Hypothèse : le verrou est
  structurel (l'imagination travaille sur 64 carrés qui décrivent chacun toute la scène,
  sans place réservée à « une chose »). On ajoute, au-dessus de l'œil figé de la phase 0,
  quelques boîtes apprises par compétition (Slot Attention), suivies d'une image à
  l'autre, et une imagination **par paires** (boîte–boîte, boîte–corps). Conception :
  [conception_boites_objets.md](conception_boites_objets.md). Trois essais le 9 octobre ;
  le troisième (perception pixel plus fine) n'a pas encore de résultat.

Détails et chiffres : [docs/ETAT_DU_PROJET.md](docs/ETAT_DU_PROJET.md).

## Lancer les scripts principaux

Les entraînements tournent sur **Colab GPU** (A100 conseillé pour la phase 0, environ
5 h 30 pour 60 000 pas) ; le monde et les figures se génèrent aussi en local.

```bash
git clone https://github.com/frpatry/jepa-physics-bench.git && cd jepa-physics-bench
pip install torch numpy scipy matplotlib
pip install git+https://github.com/rbalestr-lab/lejepa      # SIGReg officiel (vjepa.sigreg)

# 1. Voir le monde de la phase 0 (local, quelques secondes) -> av_world0.png, av_world0_sens.png
python av_world0.py

# 2. Entraîner le bébé de phase 0 (curriculum 0a -> 0e). Relancer la même commande = reprise.
#    Instantanés automatiques : <ckpt>_<étape>_<k>k.pt (ex. runs/phase0_0e_60k.pt)
python av_phase0.py --total 60000 --ckpt runs/phase0.pt \
    --touch_masks 1 --contact_w 4 --p_obj 0.8 --mob_size 0.16 --calm 2 --outside 1 --inv_clip 1
#    (options du « run 4 » ; --T 32 --bs 32 pour des séquences de 2 s, à lancer depuis zéro)

# 3. Le test qui compte : imagination décodée + planification « pousse l'objet sur la cible »
python av_plan0.py --ckpt runs/phase0_0e_60k.pt
python av_plan0.py --ckpt runs/phase0_0e_60k.pt --obj_diag 1     # que devient l'objet dans l'imagination ?

# 4. Boîtes-objets au-dessus de l'œil figé de la phase 0
python av_slots0.py --eye runs/phase0_0e_60k.pt --ckpt runs/slots0.pt
```

Options principales (voir `--help` de chaque script) :

| script | options utiles |
|---|---|
| `av_phase0.py` | `--total`, `--T` (images par séquence), `--bs`, `--ckpt`, `--stop_at`, `--calm` (0/1/2), `--outside`, `--touch_masks`, `--contact_w`, `--p_obj`, `--mob_size`, `--inv_head attn\|diff`, `--resid`, `--loss l1\|mse`, `--act_w` |
| `av_plan0.py` | `--ckpt` (obligatoire), `--episodes`, `--plan_h`, `--plan_steps`, `--pop`, `--iters`, `--obj_diag 1` |
| `av_slots0.py` | `--eye` (instantané de phase 0, obligatoire), `--ckpt`, `--steps`, `--K` (nombre de boîtes), `--ds`, `--w_dyn`, `--w_img` |

**Important** : tous les scripts sont à la racine et s'importent entre eux par leur nom
(`av_phase0.py` importe `av_jepa`, `av_dev`, `av_dev_long`, `av_world0`, `vjepa`…), et
les notebooks Colab les appellent par chemin après un `git pull`. C'est pourquoi le code
n'est pas rangé en dossiers.

## Documentation

- [docs/ETAT_DU_PROJET.md](docs/ETAT_DU_PROJET.md) — la phase actuelle en détail : architecture, runs, chiffres, questions ouvertes.
- [docs/CARTE_DES_FICHIERS.md](docs/CARTE_DES_FICHIERS.md) — quels fichiers comptent, regroupés par époque de recherche (actuels / historiques / instruments).
- [docs/HISTORIQUE.md](docs/HISTORIQUE.md) — les étapes du projet depuis juin 2026 et les leçons retenues.
- [recherche_bebe_mains.md](recherche_bebe_mains.md), [recherche_bebe_objets.md](recherche_bebe_objets.md) — recherches sourcées sur le nourrisson.
- [conception_boites_objets.md](conception_boites_objets.md) — conception des boîtes-objets.
- `session_handoff.md` — ancienne note de reprise (juillet 2026), conservée pour l'histoire.

Les messages de commit sont très descriptifs (`git log --oneline`) : ils servent de
journal de bord détaillé.
