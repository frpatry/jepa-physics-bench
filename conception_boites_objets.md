# Conception — les boîtes-objets du bébé (sur papier, à valider)

Fondements : `recherche_bebe_objets.md` (règles 1 à 9) et `recherche_bebe_mains.md`.
Contrainte de l'utilisateur : **aucun raccourci qu'on ne pourrait pas rattacher à l'apprentissage d'un bébé.**

---

## 1. Le problème à régler (rappel)

Le bébé **perçoit** bien l'objet dans le présent (lu à 2,7 px près dans le run à 32 images). Mais quand il **imagine** l'avenir, il le perd : un objet immobile « dérive » d'environ 10 px, et l'objet poussé est plus mal prédit que si l'on supposait que rien ne bouge. Seule sa main reste nette. Cinq variantes d'entraînement n'y ont rien changé :

- monde chaotique corrigé ;
- objet qui sortait du champ ;
- toucher appris ;
- contraste d'action ;
- séquences longues.

**Cause retenue : la structure.** L'imagination travaille sur 64 petits carrés qui décrivent chacun un peu toute la scène. Il n'y a pas de place réservée à « une chose », et le corps prend toute la capacité.

---

## 2. Vue d'ensemble

```
 IMAGE ──► ŒIL (encodeur de la phase 0, 64 carrés/image)
                        │
                        ▼
              REGROUPEMENT EN 4 BOÎTES  ◄── les boîtes de l'image précédente (suivi, pas d'oracle)
              (compétition : chaque carré « appartient » surtout à une boîte)
                        │  chaque boîte = { position (où), contenu (quoi) }
                        ▼
              IMAGINATION PAR PAIRES
              boîte_k(t+1) = boîte_k(t) + changement propre (élan)
                                       + Σ interactions avec les autres boîtes
                                       + interaction avec le CORPS (commande + sens du bras)
                                       + contexte des sens (toucher, ouïe)
                        │
                        ▼
              boîtes imaginées  ──►  comparées aux boîtes que l'œil trouvera vraiment (JEPA)
```

Les 4 sens restent ceux de la phase 0. **La commande n'entre jamais dans l'œil**, comme avant.

---

## 3. Les boîtes : ce qu'elles sont, et ce qu'elles ne sont pas

| Choix | Pourquoi (bébé) | Règle |
|---|---|---|
| **4 boîtes** interchangeables, sans étiquette | le nourrisson suit environ 3 objets, puis s'effondre au-delà (Feigenson & Carey) ; il y a 1 à 3 objets + la main | 1 |
| Les carrés se **disputent** entre boîtes (compétition, à la Slot Attention) | un nombre limité de « pointeurs » réassignables (index de Leslie, fichiers d'objets) | 1 |
| Chaque boîte porte d'abord **une position** (le centre de son attention sur l'image), puis un contenu | le bébé retient « il existe, il est là » avant « à quoi il ressemble » (Xu & Carey ; Kibbe & Leslie) | 2 |
| Les boîtes de l'instant t **servent de point de départ** à celles de t+1 | suivre un objet par la continuité de son trajet, sans « tricher » sur l'identité | 8 |

La position d'une boîte n'est **pas** une coordonnée qu'on lui donne. C'est le centre de la zone de l'image qu'elle a « prise ». L'œil utilise déjà la position générique de chaque carré, comme tout ViT, mais jamais celle des objets.

---

## 4. Comment les boîtes émergent : le mouvement d'abord, l'apparence ensuite

Il y a deux exercices auto-supervisés, sans étiquette. Leur dosage change au fil du développement.

1. **« Ce qui bouge ensemble va ensemble »** (destin commun, 4 mois, Kellman & Spelke). Les boîtes doivent permettre de prédire **le changement** de chaque carré entre deux instants. Des carrés qui bougent ensemble sont mieux expliqués par une même boîte, donc ils s'y regroupent.
2. **« Les boîtes doivent expliquer toute la scène »** : reconstruire le résumé de l'œil (les 64 carrés) à partir des 4 boîtes. Ça porte sur l'apparence, la forme, puis la couleur.

| Étape | Poids du mouvement | Poids de l'apparence | Bébé |
|---|---|---|---|
| Début (0d) | fort | faible | 4 mois : le mouvement segmente ; immobile, rien n'est unifié |
| Milieu | moyen | moyen | 4,5 mois : la forme commence à séparer deux objets immobiles |
| Fin (0e) | faible | fort | vers 8 mois : indices statiques fiables (couleur vers 11,5 mois) |

**Le bébé crée lui-même le mouvement qui découpe** : ses poussées font bouger l'objet, ce qui le sépare du fond et de la main (règle 4). Le monde lisible, avec ses objets au repos par défaut et ses pauses pour regarder après un contact, va dans ce sens.

---

## 5. L'imagination par paires (le cœur du changement)

On ne code **pas** « un objet immobile ne bouge pas ». On construit l'imagination de sorte que **le changement d'une boîte passe par des interactions**, comme dans les réseaux d'interaction (C-SWM, l'InteractionLSTM de PLATO) :

```
changement de la boîte k = élan propre(k)
                         + Σ_j interaction(k, j)          (objet–objet : chocs)
                         + interaction(k, CORPS)          (corps = commande + sens du bras)
                         + effet(k, toucher, ouïe)        (contact ressenti, bruit de choc)
```

- **Principe de contact** (2,5 mois, Kotovsky & Baillargeon). Si aucune interaction n'est active, il ne reste que l'élan propre. Un objet au repos a donc tout intérêt à apprendre « je reste où je suis ». Ça émerge des données, ce n'est pas une règle codée.
- **La main sans case réservée.** Le corps (commande + sens du bras) est offert à **toutes** les boîtes. On s'attend à ce qu'**une seule** apprenne à en dépendre fortement hors contact : celle qui suit la main. C'est un **test**, pas une hypothèse codée (règle 5).
- **Les objets poussés.** Quand la boîte de la main et celle d'un objet sont proches et que le toucher signale un contact, l'interaction main→objet produit le mouvement. Après le contact, c'est l'élan propre qui fait glisser l'objet.

**Apprentissage de l'imagination (JEPA).** Les boîtes imaginées à t+h doivent ressembler aux boîtes que l'œil trouvera vraiment à t+h, obtenues par suivi depuis le passé. On le fait sur plusieurs pas d'affilée (jusqu'à 8). Les boîtes imaginées doivent aussi permettre de reconstruire le résumé de l'œil à t+h, ce qui les garde ancrées dans la scène et évite l'effondrement.

---

## 6. Ce qui reste interdit (règle 8)

- ❌ donner les vrais masques ou les vraies positions des objets ;
- ❌ dire combien il y a d'objets ;
- ❌ réserver une boîte à la main ;
- ❌ coder une loi physique (« vitesse = 0 », inertie, gravité) ;
- ❌ apparier les boîtes d'un instant à l'autre par un oracle ;
- ❌ utiliser la couleur trop tôt.

Ce qui reste légitime : la petite capacité (4), la compétition, les sens séparés, la structure par paires, le suivi d'un instant à l'autre.

---

## 7. Les examens

**A. Les boîtes ont-elles émergé ?** (instruments, jamais donnés au modèle)

1. Chaque objet est-il pris par **une** boîte ? Mesure : le centre de chaque boîte comparé aux vrais objets, avec le meilleur appariement possible.
2. Une **boîte-main** est-elle apparue ? Une boîte dont la position suit la main.
3. **Test de la main** : hors contact, seule la boîte-main doit changer quand on ment sur la commande.

**B. Surprises, comme chez le bébé** (paires possible / impossible, surprise = erreur de prédiction) :

| Test | Impossible | Âge chez le bébé |
|---|---|---|
| contact | un objet immobile bouge sans être touché, ou ne bouge pas quand la main le pousse | 2,5 mois |
| continuité | téléportation | 2,5–4 mois |
| permanence | l'objet sort du champ et revient ailleurs, ou disparaît | 3,5–5 mois |
| solidité | la main traverse l'objet | 3–4 mois |
| cohésion | l'objet se scinde | 3–5 mois |
| identité | il change de forme ou de couleur hors champ | plus tard (attendu en dernier) |

Selon V-JEPA (Garrido, LeCun 2025), un modèle sans objets réussit déjà la permanence et la continuité, mais **échoue sur le contact et la solidité**. C'est donc là qu'on attend le gain des boîtes.

**C. Le vrai juge** : l'objet immobile reste-t-il en place dans l'imagination ? Puis le test de planification habituel (hasard 2 %, oracle 40 %).

---

## 8. Plan de réalisation proposé

1. **Premier essai, rapide (≈ 2–3 h de GPU).** On repart de l'**œil du run 4**, à 16 images, figé au début. Boîtes et imagination par paires sont entraînées dans le monde lisible des étapes 0d puis 0e. On regarde : les boîtes émergent-elles, la boîte-main apparaît-elle, l'objet immobile tient-il ?
2. **Si ça marche :** on dégèle l'œil pour un apprentissage conjoint, on ajoute les tests de surprise et la planification, puis on passe aux séquences longues.
3. **Code :** une demi-journée. On réutilise `av_phase0.py` (œil, sens, monde, examens), `av_world0.py` (on y ajoute les variantes « impossibles ») et l'expérience passée de `slots.py` / `slots_dyn.py`.

## 9. Risques connus

- **Une boîte peut prendre « main + objet » au contact.** C'est déjà arrivé dans `slots.py` : les objets qui se touchent fusionnent. Parades : le mouvement commun (après la poussée, l'objet part seul), le suivi d'un instant à l'autre et le toucher.
- **Petite résolution** : la main fait environ 4 px sur 32. L'œil du run 4 la situe à environ 2,6 px près, et celui du run à 32 images à 0,8 px. Si ça bloque, ce sera une raison de partir de cet œil-là.
- **Une boîte « fond » paresseuse** qui avale tout. Parades : la compétition et la petite capacité, déjà vécues dans `slots.py`.

## 10. Questions à valider

1. **Point de départ** : l'œil du run 4 (16 images, plus rapide) pour le premier essai, ou celui du run à 32 images (meilleure perception, plus lent) ?
2. **Coût de la planification** : garder, pour comparer avec les runs précédents, le lecteur de position supervisé comme *instrument*, ou passer directement à « position de la boîte de l'objet » contre « position voulue » (sans lecteur) ?
3. **Variantes impossibles** (téléportation, traversée, etc.) : les ajouter dès le premier essai, ou seulement après avoir vérifié que les boîtes émergent ?
