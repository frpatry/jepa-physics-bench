# État du projet (mis à jour le 9 octobre 2026)

Ce document décrit la phase en cours : l'**avenue multisensorielle « bébé »**, et plus
précisément la **phase 0 « mes mains »** et les **boîtes-objets**. Pour le chemin qui y
mène, voir [HISTORIQUE.md](HISTORIQUE.md).

Conventions : les positions sont mesurées en pixels sur des images 32×32 (ou en fraction
de la largeur quand c'est précisé) ; « copie » = prédire que rien ne change (la dernière
image vue), le repère minimal à battre ; « oracle » = politique qui connaît l'état exact
du monde (instrument, pas méthode).

---

## 1. Le bébé de phase 0

### Le monde (`av_world0.py`)

Une table vue de dessus, une main (carré blanc) au bout d'un bras de longueur limitée,
des objets de formes variées (disque, carré, triangle, T), et un « mobile » hors de
portée relié au geste par un ruban invisible (expérience classique de Rovee-Collier).

| étape | âge équivalent | ce qui s'y passe |
|---|---|---|
| 0a bras | 0–2 mois | main très grande, rien d'autre ; elle gigote |
| 0b mains | 2–4 mois | la main entre et sort du champ (entendue et sentie hors champ) |
| 0c contingence | 2–3 mois | mobile relié au geste, parfois autonome, parfois « déconnecté » en cours de route |
| 0d contact | 3–4 mois | 1–2 objets à portée, surtout au repos ; le lourd repousse la main |
| 0e distance | 5 mois et + | 1–3 objets variés, certains hors de portée |

Options du monde ajoutées au fil des runs (toutes des idées « comment un bébé apprend ») :
complexité progressive et monde **lisible** (`--calm 2` : objets au repos par défaut,
pauses pour regarder), **table qui déborde du champ** (`--outside 1` : objets qui
sortent de la vue et qu'on entend hors champ), séquences de 16 ou 32 images (`--T`).
La vue du bébé passe du flou uniforme, gris et peu contrasté au net (« flou neuronal »,
cf. `recherche_bebe_mains.md`).

### Le modèle (`av_phase0.py`)

- **Encodeur des 4 sens** (vue 64 carrés par image, ouïe, toucher, sens du bras), un
  encodeur par sens ; la fusion se fait dans le prédicteur.
- **Prédicteur conditionné par la commande dès le départ** (style PLDM / V-JEPA 2-AC) :
  contexte + gestes de toutes les images → latents des sens masqués, y compris le futur.
  Règle d'architecture : ce qui est **ressenti** va à l'encodeur, ce qui est **voulu**
  (la commande) uniquement au prédicteur — sinon le modèle pourrait tricher.
- Masques : blocs de vision, futur de tous les sens, un sens entier deviné par les
  autres.
- **Contingence** : deviner son geste en voyant l'avant et l'après (modèle inverse).
- Cible EMA + SIGReg faible (0,005) + taux d'apprentissage cosinus (recette stable
  trouvée en phase 1).
- **Examens sans étiquette** (surprise, façon « violation d'attente ») : écoute-t-il
  ses gestes ? ruban coupé / retard de 3 images ? vu ici, senti là ? contact vu mais
  pas senti ? Plus une sonde ridge (instrument) sur la main et un objet.

### Les runs (tous à 60 000 pas sauf mention)

| run | réglages | ce qu'on a appris |
|---|---|---|
| 1 | options par défaut | corps propre acquis : erreur sur le bras +118 % si on ment sur les gestes, vu/senti 100 %, geste deviné R² 0,83. Toucher non appris (contact non senti : 8 % de surprise), mobile non appris. |
| 2A / 2C | prédicteur résiduel, perte L1 (2A) ou MSE (2C), masques « devine ce que tu sens », poids sur les contacts | **toucher appris** (≈ 95 %), mais l'imagination devient une copie : le bébé n'écoute presque plus ses gestes. Arrêtés vers 34 000 pas. |
| 2D | 2C + contraste d'action | l'écart latent apparaît, mais pas dans les positions décodées (raccourci probable). En pause. |
| 3 | reprise du run 1 à 22 000 pas, monde calme progressif puis table qui déborde | toucher 91–92 %, main imaginée nette ; l'objet imaginé colle à la main, puis est perdu. En pause vers 45 000 pas. |
| 4 | reprise du run 1 à 22 000 pas, monde **lisible** (`--calm 2 --outside 1`) | bras +106 %, vue près de la main +28 %, toucher 88 %, vu/senti 100 %, geste deviné 0,79. Mêmes défauts d'imagination de l'objet. **C'est l'œil utilisé par les boîtes-objets.** |
| T32 | depuis zéro, 32 images par séquence | à 34 000 pas : meilleure perception (objet lu à 2,7 px, main à 0,8 px sur les vrais latents) mais imagination plus faible (main imaginée 7,2 px, pire que la copie 2,9). Run poursuivi ; pas de bilan final à ce jour. |

### Le test qui compte : imaginer et planifier (`av_plan0.py`)

Tâche jamais apprise : pousser un objet au repos sur une cible. Le bébé imagine les
conséquences de séquences de gestes (CEM), joue le premier geste, regarde, recommence
(MPC). Le coût est lu par un lecteur de positions entraîné sur les vrais latents
(instrument).

Résultats du run 4 (60 000 pas) — représentatifs de tous les runs :

| mesure | bébé | repères |
|---|---|---|
| main imaginée (8 images) | 2,71 px | copie 4,05 ; gestes d'une autre séquence 8,11 |
| objet poussé imaginé | 10,5 px | copie 7,8 |
| objet **immobile** imaginé | dérive de 10,7 px | copie 3,5 |
| planification : distance finale objet–cible / réussite | 0,271 / 0 % | hasard 0,214 / 2 % ; oracle 0,114 / 40 % |

Le diagnostic `--obj_diag 1` montre qu'un lecteur **neuf entraîné sur l'imagination**
ne retrouve pas l'objet non plus (environ 8 à 10 px, contre 2 px pour la main) :
l'information sur l'objet est réellement absente des latents imaginés ; ce n'est pas un
simple décalage du lecteur. Le planificateur exploite alors les erreurs d'imagination,
d'où un résultat parfois pire que le hasard.

**Lecture** : le bébé sait où ira **sa main**, pas ce qu'elle fait **à l'objet**. Les
causes « données » (monde chaotique, toucher non appris, objets qui sortent du champ,
durée des séquences) ont été écartées une à une. Reste la **structure** : un prédicteur
qui travaille sur 64 carrés aux contenus globaux, où le corps prend toute la capacité.

---

## 2. Les boîtes-objets (en cours)

Fondements : [recherche_bebe_objets.md](../recherche_bebe_objets.md) (neuf règles
tirées de la littérature : quelques « pointeurs » d'objets interchangeables, la position
avant l'apparence, l'émergence par le mouvement commun, l'objet inerte par défaut, la
main distinguée par la contingence…). Conception :
[conception_boites_objets.md](../conception_boites_objets.md).

Principe (`av_slots0.py`) :

- l'**œil** de la phase 0 (run 4) est **figé** ;
- quelques **boîtes** apprises (5 par défaut : jusqu'à 3 objets, la main, une marge
  pour le fond) se **disputent** les carrés de chaque image (Slot Attention) ; les
  boîtes de l'image précédente, propagées par l'imagination, servent de point de départ
  (suivi « prédire puis corriger », sans appariement par oracle) ;
- **émergence** par le mouvement d'abord (prédire le changement de chaque carré), puis
  par l'apparence ;
- **imagination par paires** : boîte(t+1) = boîte(t) + élan + interactions avec les
  autres boîtes + interaction avec le corps (commande, sens du bras, offerte à toutes
  les boîtes) + toucher/ouïe ; entraînée en JEPA sur 8 pas.
- Interdits : aucun masque ou position d'objet fourni, aucun nombre d'objets, aucune
  boîte réservée à la main, aucune loi physique codée.
- Examens : (A) les boîtes se posent-elles sur les objets et la main ? (B) hors contact,
  mentir sur la commande ne change-t-il **que** la boîte de la main ? (C) un objet
  immobile reste-t-il en place dans l'imagination ?

Essais du 9 octobre :

| essai | changement | résultat |
|---|---|---|
| 1 | les boîtes lisent le résumé global de l'œil | rien n'émerge, puis divergence |
| 2 | les boîtes lisent les carrés « rétinotopiques » (pixels vus + position), boîtes bornées | boîtes identiques entre elles (attention uniforme) |
| 3 | perception pixel fine (CNN 16×16, goulot 16, décodeur 1×1 faible, mélange par pixel — la recette qui avait marché dans `slots.py`) | en cours, pas encore de résultat |

---

## 3. Questions ouvertes et suites envisagées

- Les boîtes-objets émergent-elles sur ce monde, et gardent-elles l'objet immobile en
  place dans l'imagination (examen C) ? C'est la condition pour que la planification
  dépasse le hasard.
- Si les boîtes émergent (plan de `conception_boites_objets.md`, § 8) : dégeler l'œil
  pour un apprentissage conjoint, ajouter les examens de « violation d'attente » sur les
  objets (contact, continuité, permanence hors champ, solidité, cohésion) avec des
  vidéos impossibles, refaire la planification, puis passer aux séquences longues.
  Le lecteur de positions supervisé reste pour l'instant l'instrument du coût de
  planification (comparabilité avec les runs précédents).
- Pistes gardées en réserve : paires contrefactuelles (même état, gestes différents),
  deux yeux (disparité), images plus grandes (64×64), tubelets pour des séquences de 64
  images.
