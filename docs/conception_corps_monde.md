# Conception — séparer le CORPS et le MONDE (sur papier, à valider)

Fondements : `recherche_bebe_corps_monde.md` (règles 1 à 11). Contrainte de l'utilisateur : **aucun raccourci qu'on ne pourrait pas
rattacher au développement d'un bébé.**

---

## 1. Le problème à régler (rappel)

- Le bébé imagine très bien **sa main** (≈ 2 px) mais mal **les objets** : objet immobile qui dérive, objet poussé pas mieux que « rien ne
  bouge », objet perçu **tiré vers la main** (jusqu'à 5–6 px quand il est loin d'elle).
- Six changements côté monde/données (calme, hors champ, 32 images, main du parent, blocs, attention à la surprise) n'y ont **rien changé**.
- Aujourd'hui, **un seul prédicteur** reçoit tous les sens **et la commande motrice**, et doit tout imaginer : le plus rentable pour lui est
  d'apprendre la main (contrôlée, toujours là, prévisible) ; l'objet devient une moyenne. C'est la « compétition de modalités » connue en
  apprentissage automatique, ici entre la main et l'objet.

Chez le nourrisson, c'est organisé autrement : un **modèle du corps** (cervelet, copie d'efférence) reçoit la commande ; le **monde**
reçoit l'**état du corps** (où est la main, ce qu'elle touche) et ne bouge que par **contact** (Wolpert 1998 ; Spelke 1990).

---

## 2. Vue d'ensemble

```
                      ┌──────────────────────────── MODÈLE DU CORPS (« cervelet ») ────────────────────────────┐
 commande (volonté) ─►│ contexte : bras + toucher passés  ──►  prédit : bras (position, vitesse) et toucher      │
                      │                                         des images suivantes                            │
                      └──────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                         │ ÉTAT DU CORPS (12 nombres / image :
                                                         │ bras 4 + toucher 8 — la proprioception, rien d'autre)
                                                         ▼
                      ┌───────────────────────────── MODÈLE DU MONDE ──────────────────────────────────────────┐
 vision + ouïe ──────►│ contexte : vue + ouïe passées + état du corps (passé réel, futur prédit)              │
                      │   ──►  prédit : vue + ouïe des images suivantes (la main vue, les objets, les sons)   │
                      │   ✗ ne reçoit JAMAIS la commande                                                      │
                      └────────────────────────────────────────────────────────────────────────────────────────┘
```

- Les **encodeurs** des sens restent ceux d'aujourd'hui (un passage par sens), avec en option des **poids séparés par sens**
  (étape 2 du plan).
- La commande n'entre **que** dans le modèle du corps (règle 2). Le monde ne connaît que **l'état du corps**, comme le cerveau reçoit
  la proprioception.

---

## 3. Le modèle du corps

| Choix | Pourquoi (bébé) | Réf. |
|---|---|---|
| Entrée : bras + toucher passés + **commandes** | modèle direct : copie d'efférence -> conséquences sensorielles | Wolpert, Miall & Kawato 1998 |
| Sortie : bras et toucher **futurs** (dans l'espace latent de ces sens, comme aujourd'hui) | ce que le cervelet prédit, c'est ce que le corps va SENTIR | Blakemore 1998 |
| Petit réseau, entraîné **en premier** (étapes 0a–0b) | « le corps d'abord », contingence dès les premiers mois | Rochat ; Bahrick & Watson 1985 |
| Il ne voit **pas** l'image | le modèle direct du bras n'a pas besoin de la vue ; la vue de la main se calibre ensuite | Rochat & Morgan 1995 |

**État du corps transmis au monde** : les 12 nombres de la proprioception (position et vitesse de la main, 8 capteurs de toucher), **réels**
pour le passé, **prédits par le corps** pour le futur imaginé. C'est un goulot étroit et lisible : le monde sait **où sera la main et si
elle touchera**, mais pas « quelle commande a été donnée ».

---

## 4. Le modèle du monde

| Choix | Pourquoi (bébé) | Réf. |
|---|---|---|
| Entrée : vue + ouïe passées + **état du corps** (passé et futur) | le monde reçoit l'état du corps, pas la commande | lecture de Sommer & Wurtz 2008 |
| Sortie : vue + ouïe futures (masquées, JEPA comme aujourd'hui) | même apprentissage auto-supervisé | — |
| **Aucune commande** | seul chemin causal vers l'objet = le contact (main là + toucher) | Spelke 1990 |
| La main **vue** est prédite par le monde à partir de l'état du corps | calibration visuo-proprioceptive (« ma main est là où je la sens ») | Rochat & Morgan 1995 |
| La **main du parent** n'a pas d'état du corps | ce n'est pas MON corps : le monde doit la prédire comme un agent extérieur | Filippetti 2013 (soi/autrui) |

**Pourquoi ça devrait aider l'objet** : aujourd'hui, le prédicteur doit *intégrer les commandes* pour savoir où va la main (travail lourd,
qui capte la capacité). Avec l'état du corps, la position de la main est **donnée** (12 nombres) : la dessiner devient facile, la capacité
se libère pour les objets, et le contact devient explicite (main proche + toucher).

---

## 5. Apprentissage

1. **0a–0b (corps d'abord)** : le modèle du corps apprend bras et toucher à partir de la commande ; le monde apprend en parallèle la vue
   de la main à partir de l'état du corps réel (calibration).
2. **0c–0e** : les deux continuent. Le monde reçoit l'état du corps **réel** à l'entraînement (ce que le bébé sent vraiment), et l'état
   **prédit** par le corps en imagination (planification). Option : mélanger progressivement réel/prédit pour que le monde supporte les
   erreurs du corps.
3. Contingence (deviner son geste à partir de la vue) : gardée telle quelle (Bahrick & Watson).
4. Pas de modulation de gradient par modalité, pas de masques d'objets, pas de commande dans le monde (raccourcis interdits, règle 11).

---

## 6. Les examens (toujours comparés au run 5 **dans le même monde** — leçon de l'option B)

**A. Le corps** : erreur bras/toucher imaginés ; « mentir sur la commande » change le corps (attendu) ; contingence comme avant.

**B. Le monde est-il externe ?**
1. Mentir sur la **commande** ne change le monde **que via** l'état du corps (on le vérifie en gelant l'état du corps : alors plus aucun effet).
2. Donner au monde un **faux état du corps** (main ailleurs) : la main vue suit, et l'objet ne bouge que si la fausse main le **touche**.
3. Main du parent : prédite sans état du corps (mesure : erreur sur les cases du parent).

**C. L'objet** (les instruments habituels) : B0 (perception de l'objet selon la distance à la main, biais vers la main), objet immobile
qui dérive, objet poussé imaginé vs copie (par horizon), lecteur neuf sur l'imagination.

**D. Planification** : main, toucher, bouger, direction, objet — avec et sans sous-buts, dans le monde de blocs et avec glissades ;
films. Commande -> corps -> état du corps -> monde -> objet.

**Critères de réussite** : objet immobile imaginé **≤ copie** ; objet poussé imaginé **< copie** ; biais vers la main divisé par deux ;
« bouger » nettement au-dessus du run 5 dans le même monde ; main toujours ≈ 2 px.

---

## 7. Plan de réalisation proposé

1. **Essai A (≈ 3–4 h de GPU, comparaison propre)** : repartir de l'œil du run 1 à 0c (pas 22k, comme les runs 3–5 et B), remplacer le
   prédicteur unique par **corps + monde**, entraîner 0d–0e avec **exactement le monde du run 5** (une seule différence = l'architecture).
2. **Si l'objet s'améliore** : étape 2 = encodeurs à **poids séparés par sens** (depuis zéro), puis boîtes-objets **dans le modèle du
   monde** (recette `slots.py` complète), puis stéréo vers « 4 mois ».
3. **Code** : une demi-journée — `av_phase0.py` (nouveau prédicteur à deux branches, option `--split 1`), `av_plan0.py` (imagination
   commande -> corps -> monde), nouveaux examens B.

## 8. Risques connus

- **Le monde ignore l'état du corps** et apprend la main uniquement par les images passées : examen B2 le détecte.
- **Erreurs du corps qui s'accumulent** en imagination longue : le monde est entraîné sur l'état réel ; l'option mélange réel/prédit y répond.
- **L'objet reste flou malgré tout** (si le problème est la représentation par carrés globaux) : on le saura vite, et ce serait la
  raison de passer aux boîtes-objets **dans** le modèle du monde.
- **Mobile du ruban (0c)** : il bouge sans contact, relié au bras ; le monde peut l'apprendre via l'état du corps (vitesse du bras) — c'est
  la contingence de Rovee-Collier, pas une triche.

## 9. Questions à valider

1. **État du corps** = les 12 nombres de la proprioception (bras + toucher, recommandé : lisible et fidèle), ou un petit vecteur appris ?
2. **Entraînement du monde** : état du corps **réel** (recommandé au début), ou mélange progressif avec le **prédit** ?
3. **Point de départ** : l'œil du run 1 à 0c (rapide, comparaison directe avec le run 5, recommandé), ou tout depuis zéro avec des
   encodeurs séparés par sens (plus fidèle, plus long, deux changements à la fois) ?
4. **Monde de l'essai** : exactement celui du run 5 (une seule variable, recommandé), ou avec les blocs de l'option B ?
