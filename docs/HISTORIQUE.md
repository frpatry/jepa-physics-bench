# Historique du projet et leçons retenues

Résumé des grandes étapes depuis juin 2026. Le détail de chaque essai est dans les
messages de commit (`git log`), qui servent de journal de bord. Les chiffres viennent
des runs Colab de chaque époque ; quand un résultat est fragile ou n'a pas été
confirmé, c'est indiqué.

Fil rouge : de « un JEPA apprend-il quelque chose d'utile ? » vers « un agent qui
apprend comme un bébé peut-il planifier une tâche jamais apprise ? ». À chaque époque,
on a cherché à **mesurer honnêtement** (repères copie / hasard / oracle, plafonds par
instruments) avant de conclure.

---

## 1. Apprentissage continu (mi-juin 2026) — `cl_*`

Question de départ : comment éviter l'oubli catastrophique quand un modèle apprend des
domaines l'un après l'autre ?

- Modèle de langage à **routage gelé** (top-K d'unités choisies par une projection fixe
  du contexte, `cl_scale.py`) : sur du vrai texte (wikitext → ag_news → imdb), le
  routage oublie environ **3 fois moins** que le modèle dense ; l'oubli restant passe
  par les paramètres partagés non routés (plongements, attention).
- Passage au **JEPA** (`cl_jepa.py`, `cl_jepa_text.py`) : en JEPA, le routage **ne
  protège plus** (sur données synthétiques, il oublie même plus que le dense).
- Barlow-JEPA (`cl_jepa_demo.py`) mesuré par sonde linéaire gelée sur de vraies
  étiquettes : le **replay** réduit l'oubli de 40 % à 24 % sans perdre la plasticité ;
  le **multi-vue** « 4 réseaux » (`cl_multiview.py`) le réduit à 26 % sans replay —
  le seul levier « intrinsèque » qui ait marché. Largeur, nombre de pas, EMA,
  routage : aucun effet sur la rétention.

## 2. Banc SSL sur physique jouet (22 juin 2026) — `cl_sslbench.py`

JEPA sur des trajectoires de balles sous gravité ; on sonde la gravité `g` apprise
sans étiquette. Comparaison des termes anti-effondrement (aucun, VICReg, SIGReg).

- Trois versions maison de SIGReg ont échoué (gradient nul au point d'effondrement) ;
  le **SIGReg officiel** (package `lejepa`, test d'Epps-Pulley) est adopté pour la
  suite du projet.
- Sur une physique plus réaliste (rebond + vent), la représentation SIGReg retrouve une
  notion robuste de la gravité, proche du plafond supervisé (`--oracle`). Un bug de
  projection (train et test dans des espaces différents) avait longtemps plafonné le
  banc au hasard.
- Extensions du même script : vision (`--vision`), trajectoire décodée (`--traj`),
  hiérarchie macro/micro (`--macro`, `--forecast`), raquette qui intercepte (`--actor`).

## 3. Conduite, vidéo réelle et V-JEPA fidèle (23–24 juin 2026)

- **Jouets de conduite** (`drive*.py`) : world model JEPA sur images ego-centrées et
  conducteur appris par imitation ; sur une route avec stops et passages piétons,
  0 % de collision et 96 % de stops respectés. Mais un modèle plus gros fait pire, et
  l'imitation est fragile en boucle fermée — argument pour la **planification**.
- **Vidéo réelle** (`lejepa_video.py`) : LeJEPA sur UCF101 ; une sonde sur 15 classes
  jamais vues atteint 0,52 (hasard 0,07).
- **Transfert vers la dashcam** (`driving_transfer.py`, données Nexar) : un encodeur
  entraîné sur UCF101, sans jamais voir de route, détecte le danger presque aussi bien
  qu'un encodeur entraîné sur dashcam (0,65 contre 0,68, hasard 0,50 au premier run).
- `vjepa.py` : architecture **fidèle à V-JEPA** (encodeur sur les seuls tokens
  visibles, prédicteur attentionnel, masques en tubelets, SIGReg sans EMA) ; c'est le
  socle commun de tout ce qui suit.
- Prédiction de collision par rollout (`driving_rollout.py`) : premier run au hasard ;
  correctif poussé mais **jamais validé**. Planificateur MPC latent (`drive_plan.py`) :
  56 % de collisions (naïf) → 8 % (réactif) → environ 0 % (MPC) en essai réduit, mais
  la dynamique latente apprise restait presque l'identité.

## 4. Monde d'objets et apprentissage actif (24 juin – 4 juillet 2026)

- `objects.py` : disques qui rebondissent et se percutent. La position est lisible
  dans les latents (R² 0,60), mais **prédire le latent brut du futur échoue** ; un bon
  score antérieur venait d'un encodeur bidirectionnel qui **voyait le futur** — en
  encodage causal honnête, il s'effondre.
- `objects_plan.py` : le System-2 ne bat pas une politique réactive simple.
- `explore.py` / `explore_state.py` : à données égales, la **curiosité naïve** (aller là
  où les prédicteurs sont en désaccord) fait **moins bien que le hasard** (elle
  s'obsède sur quelques objets).

## 5. Objets émergents : les slots (4–6 juillet 2026)

- `slots.py` : après 12 runs, chacun fermant une « échappatoire » du modèle, **un slot
  par objet sans supervision** (erreur de position 0,048 de la largeur, sur des scènes
  de 1 à 4 objets ; le comptage émerge). Recette : épluchage récursif (« peel », idée de
  l'utilisateur), vraisemblance de mélange par pixel, fond de couleur unie, goulot de
  16 dimensions, décodeur faible, scènes variées.
- `slots_dyn.py` : la dynamique sur les slots **bat la copie à tous les horizons**,
  chocs compris (−22 à −28 %) — première prédiction honnête du futur du projet.
- `slots_act.py` (toy-Push) : planification **11/20** (oracle 20/20, hasard 0/20) une
  fois le coût mesuré sur les masques décodés.
- Limite : les slots ne capturent pas l'**orientation** d'un T (`slots_orient.py` :
  erreur d'angle 89°, le hasard) ; deux objets qui se touchent fusionnent.

## 6. Push-T (5–14 juillet 2026)

Objectif : se comparer à DINO-WM (taux de succès 0,90 sur Push-T).

- Slots sur Push-T (`slots_pusht.py`, `pusht_plan.py`) : world model obtenu après
  7 runs de diagnostic, mais la planification échoue (le T est flou, l'angle perdu).
- **World model fondé sur la pose** (`pusht_posewm.py`, perception géométrique, non
  apprise) : couverture moyenne 0,70 contre 0,14 au hasard (×5) sur les tâches non
  dégénérées, jusqu'à 0,94 sur une tâche, mais **aucun succès** au seuil officiel
  de 0,95. Ce n'est pas comparable à DINO-WM (perception codée à la main).
- JEPA + SIGReg officiel (`pusht_jepa.py`) : entraîné conjointement, il **échoue**
  (planification au niveau du hasard) ; en deux étapes (`pusht_jepa2.py`), l'encodeur
  maison reste trop faible (sonde de pose : position du T à 85 px, angle à 79°, soit
  le niveau du hasard pour l'angle).
- **V-JEPA 2 de Meta gelé** à la place (`pusht_vjepa2.py`) : la même sonde lit la
  position à 24 px et l'angle à 10° → le pipeline était sain, l'encodeur maison était
  le problème. La dynamique et le planificateur ont été construits ; leurs résultats
  n'ont **pas été rapportés**.

## 7. Avenue multisensorielle « bébé » (depuis le 2 octobre 2026) — `av_*`

Nouvelle question : un JEPA qui apprend « comme un enfant » (ouïe, vue, puis toucher et
action) apprend-il mieux ? Principe posé dès le départ : **jamais d'apprentissage
supervisé** ; les étiquettes ne servent qu'à mesurer.

### Phase 1 — perception : vision + ouïe

- Sept variantes de JEPA de fusion vision + son (`av_jepa.py`, `av_fusion.py`, avec ou
  sans V-JEPA 2 gelé) : **aucune n'apprend le liage son ↔ disque** (quel son appartient
  à quel objet) ; à pleine échelle, la fusion fait même moins bien que les entrées
  brutes (68 % contre 78 % sur le matériau, `av_fair_eval.py`). Les instruments
  (`av_world_check.py`, `av_bind_check.py`) montrent que l'information existe et que le
  liage est calculable.
- **Mesurer par la surprise** : un JEPA entraîné longtemps (`av_dev_long.py`) relie
  bien l'image et le son (80 % de surprise quand on inverse la stéréo), alors que les
  examens à lecteur supervisé le disaient nul. Leçon : tester un JEPA par sa propre
  prédiction avant tout lecteur.
- Monde v4 (chaque objet émet un son continu) : la vision apprend à **localiser** les
  objets (R² 0,91), mais la localisation **disparaît quand l'image devient nette**.
- **Effondrement lent** de la cible EMA vers 55 000–80 000 pas, reproductible. Ni la
  décroissance du taux d'apprentissage ni une EMA constante ne l'empêchent ; un SIGReg
  fort ajouté en cours de route détruit le savoir. Recette stable : **SIGReg faible
  (0,005) dès le début + taux d'apprentissage cosinus**.
- **Encodeurs séparés image / son** (fusion laissée au prédicteur) : localisation
  R² 0,93 qui **tient à image nette**, surprise 88–90 %. Perception de phase 1 jugée
  acquise.

### Phase 2 — action et toucher (`av_act.py`)

- Prédicteur conditionné par l'action, résiduel, sur V-JEPA 2 gelé : la poussée d'un
  disque est prédite à 1 pas (3,1 px contre 4,8 px pour la copie). Mais le son et le
  toucher n'apportent rien, et même la **masse vraie donnée en entrée n'aide pas** :
  le prédicteur ne sait pas attacher une propriété au bon objet (le liage, encore).
- Planification (pousser un disque vers une cible, monde à frottement) : le MPC reste
  au niveau du hasard (0,19 contre 0,196 ; oracle 0,140). Diagnostic : le prédicteur est
  **presque sourd à l'action**, alors que la même chaîne sur l'**état exact** planifie
  mieux que l'oracle scripté (27 % de réussite contre 20 %). Cause : des latents
  **globaux** (chaque token décrit toute la scène ; l'entraînement JEPA rend les tokens
  plus globaux qu'un réseau aléatoire).
- Monde varié (`av_world5.py`) et vision locale (v7) : premier signe d'écoute de
  l'action, le MPC fait environ 45 % du chemin du hasard à l'oracle (0,176 ; hasard
  0,201 ; oracle 0,145), puis **plateau** quelle que soit la représentation essayée.

### Phase 0 — « mes mains » (7–9 octobre 2026)

Constat : le bébé artificiel avait appris les objets d'abord, en spectateur, et ne
voyait pas sa main. Un vrai bébé découvre d'abord son corps, et sait ce qu'il veut faire
(copie d'efférence). D'où la phase 0 (`av_world0.py`, `av_phase0.py`, `av_plan0.py`) :
voir [ETAT_DU_PROJET.md](ETAT_DU_PROJET.md).

- **Run 1** : corps propre acquis (écoute ses gestes, vu/senti, geste deviné), toucher
  et mobile non appris. Imagination : la main est bien prédite (2,1 px contre 4,6 pour
  la copie), l'objet poussé non (9,6 px contre 5,7). Planification sous le hasard.
- **Runs 2A / 2C / 2D** (prédicteur résiduel, perte MSE, contraste d'action) : le
  toucher est appris (≈ 95 %), mais l'imagination devient une copie, sourde aux gestes.
- **Run 3** (monde calme, puis table qui déborde) et **run 4** (monde lisible) :
  le corps et le toucher restent acquis ; l'objet imaginé colle à la main ou dérive
  d'environ 10 px ; planification 0,27 contre 0,21 au hasard.
- **Run à 32 images** : meilleure perception (objet 2,7 px, main 0,8 px dans le
  présent) mais imagination plus faible ; à 34 000 pas, pas de gain en planification.
  Bilan final non disponible.

### Boîtes-objets (9 octobre 2026, en cours)

Recherche sourcée (`recherche_bebe_objets.md`), conception sur papier
(`conception_boites_objets.md`), puis `av_slots0.py`. Essai 1 : rien n'émerge.
Essai 2 : boîtes identiques. Essai 3 (perception pixel fine reprise de `slots.py`) :
en cours.

---

## Leçons transversales

1. **Toujours un repère et un plafond.** La copie (« rien ne bouge »), le hasard et un
   oracle ; et un instrument qui vérifie que l'information existe avant d'accuser le
   modèle (`av_world_check.py`, `av_test_check.py`).
2. **Mesurer dans l'espace décodé**, pas en distance latente : des latents « 2,4 fois
   meilleurs que la copie » peuvent donner des positions pires que la copie ; un
   planificateur exploite les erreurs de son modèle.
3. **Tester un JEPA par sa propre prédiction** (surprise) avant de le juger avec un
   lecteur supervisé, qui peut être trop faible.
4. **Honnêteté causale** : un encodeur qui voit le futur fausse toute mesure de
   prédiction.
5. **La perception est le goulot de tout l'aval** ; un encodeur maison faible ne se
   rattrape pas par la dynamique ou le planificateur (V-JEPA 2 gelé l'a montré sur
   Push-T).
6. **Le liage propriété ↔ objet** (son, masse, puis l'objet lui-même dans
   l'imagination) est le fil rouge des échecs depuis octobre : d'où les boîtes-objets.
7. **Recette JEPA stable** pour les runs longs : SIGReg faible dès le début, taux
   d'apprentissage cosinus, encodeurs séparés par sens.
8. **Le monde compte autant que le modèle** : un monde chaotique rend cause et effet
   illisibles ; un monde lisible et une complexité progressive sont des choix de
   conception à part entière.
