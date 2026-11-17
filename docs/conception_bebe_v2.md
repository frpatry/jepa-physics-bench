# Bébé v2 — conception d'un entraînement complet à zéro, graduel

*14 novembre 2026 — à valider avant tout code. Rien n'est lancé.*

## 1. Pourquoi repartir de zéro

- **Ce qui est acquis :** de bons sens. La vue sépare les objets (même couleur collés : 0,85), le toucher et l'ouïe réels disent presque parfaitement « ça a glissé » (0,94 à 0,99), le bras situe la main à 0,4 px.
- **Ce qui manque :** l'imagination. L'imagination d'origine « saute » d'un coup à une image future ; elle n'a jamais déplacé un objet, et trois pertes ajoutées après coup n'y ont rien changé. Le module pas à pas par objet (posé à côté, sur l'œil gelé) déplace l'objet à 50–70 %, mais seulement pour la vue. Le son et le toucher imaginés tombent au hasard dès que la main est au contact (0,52 à 0,59).
- **Le défaut de fond :** des morceaux qui ne se parlent pas. Bébé v2 = un seul modèle, où l'imagination pas à pas par objet fait partie du bébé et où tous les sens sont imaginés à partir de la même scène imaginée.

**Principe (ta consigne) : on améliore sans rien enlever, et l'entraînement est graduel.**

## 2. Ce qu'on garde — tout ce qui a rendu les sens bons

| Ingrédient | Réglage qui a marché | Ce qu'il a apporté |
|---|---|---|
| Entendre et sentir avant de voir (« dans le ventre ») | étape « son seul » des premiers essais | point de départ du curriculum ; jamais isolé par un test |
| Son continu de chaque objet, frottement, chocs, froissement de la main | `hum 0.15`, sons de frottement | positions lues 0,91 contre ≈ 0 sans |
| Vision floue et grise, puis nette | flou 3,0 → 0, gris 1 → 0, contraste 0,5 → 1 | la localisation naît pendant le flou |
| Un encodeur par sens, fusion dans le prédicteur | `--sep 1` | localisation conservée au net (0,93) |
| Œil-cible lent + anti-effondrement dès le début | EMA 0,996, `--sig_w 0.005`, lr cosinus | 100k pas sans effondrement |
| Le corps d'abord : bras, puis mains (grandes → petites) | étapes 0a–0c | corps acquis, planification de la main |
| Le geste entre par le prédicteur seulement | règle d'architecture, `--inv_w 0.1 --inv_clip 1` | geste deviné 0,83–0,91 |
| Toucher appris | `--touch_masks 1 --contact_w 4` | contact senti 6 % → 95–97 % |
| Rémanence du toucher | `--echo 0.5` | toucher 95–98 % |
| Objets qui ne bougent pas seuls, monde calme | `--calm 2`, objets au repos | avant : la moitié des objets bougeaient sans contact |
| Pas de patinoire | `--damp 0.5` (toujours) | glissade 18 → 3,7 px, plus de contacts |
| Hors champ, parent, mobile | `--outside 1`, `--parent 0.4`, `--mob_size 0.16` | objet plus collé à la main, toucher 83 % |
| Jeu libre + curiosité | `--blind 1 --curious 3` | meilleur corps, plus de fuite par les gestes scriptés |
| Yeux fermés par moments | `--eyes_share 0.15` | l'objet imaginé écoute ses gestes à 1 pas |
| Main cachée par moments | `--hand_hidden 0.5` | distinguer main et objets (ton idée) |
| Œil local, prédicteur causal | `--ctx_w 0.5 --ctx_on enc`, `--causal 1 --fut_share 0.7` | objet lu sur sa case 8,6 → 3,3 px |
| Objets plus gros, puis taille normale | `--obj_scale 1.8`, `--scale1_share 0.5` | notion d'objet au-delà de la couleur |
| Forme qui compte (le T tourne) | `--shape_phys 1` | rotation réaliste |
| Rejouer des scènes à plusieurs objets | `--replay 0.1` | perception 0,84 px |
| Enchaîner plusieurs pas | `--roll_w 1 --roll_k 3` | corps qui tient sur plusieurs pas |

## 3. Ce qu'on améliore

1. **Une vraie étape « dans le ventre ».** Dans la lignée actuelle, la vue était là dès le début (floue). On ajoute une première étape sans vue du tout : ouïe, toucher, bras, gestes.
2. **L'imagination pas à pas par objet, dans le bébé.** Un geste = une image, jamais de saut. Les choses sont transportées, le fond est recopié. Cibles dans les latents seulement.
3. **Tous les sens imaginés depuis la même scène imaginée.** Si la main imaginée rencontre l'objet imaginé, le toucher et le son imaginés en découlent.
4. **Passage d'étape sur critère, pas sur un nombre de pas fixe.** On passe quand la perte plafonne ET que le critère de l'étape est atteint.
5. **Les diagnostics d'aujourd'hui tournent à la fin de chaque étape** (ils existent déjà), au lieu d'être faits des semaines après.
6. **Deux graines** pour le pilote : plusieurs résultats passés variaient d'une graine à l'autre.
7. **Reprise après coupure.** Le Colab s'arrête au bout de 24 h au plus : l'entraînement enregistre son état sur le Drive à intervalles réguliers (modèle, étape en cours, compteurs, critères déjà passés) et repart exactement où il s'était arrêté, en une commande.

## 4. Le curriculum graduel

Chaque étape garde tout ce que les précédentes ont installé.

| Étape | Ce que vit le bébé | Ce qui s'ajoute | Critère de passage |
|---|---|---|---|
| **0 — Dans le ventre** *(nouveau)* | Aucune vue. Il bouge, s'entend bouger, sent son bras, touche | ouïe, toucher, bras, gestes ; anti-effondrement | il écoute ses gestes par le bras ; geste deviné ≥ 0,8 |
| **1 — Naissance, mes mains** | Vue floue et grise ; ses mains, grandes puis petites | la vue ; un encodeur par sens | vu ici = senti là ≈ 100 % ; main lue ≤ 1 px |
| **2 — Le monde lisible** | Mobile, objets au repos, monde calme, parent | objets inertes, frottement, hors champ | il distingue « j'ai fait bouger » de « ça bouge seul » |
| **3 — Le contact** | Il touche des objets ; rémanence du toucher | toucher appris, sons de choc et de frottement | contact senti ≥ 95 % ; toucher et son réels disent « ça glisse » ≥ 0,9 |
| **4 — Vue nette, jeu libre** | Vue nette ; jeu libre curieux ; yeux fermés et main cachée par moments ; gros objets puis taille normale ; formes | notion d'objet | objet perçu ≤ 1,3 px ; même couleur collés ≥ 0,8 |
| **5 — Imaginer les objets** *(nouveau dans le bébé)* | Le même jeu libre | imagination pas à pas par objet : 1 pas, puis 4, 8, 12 | objet poussé imaginé ≥ 50 % de l'amplitude à 8 pas, bonne direction (≥ 0,8) ; objet immobile : dérive ≤ 0,3 px |
| **6 — Imaginer avec tous les sens** *(nouveau)* | Le même jeu libre | toucher, son et bras imaginés pas à pas depuis la scène imaginée | main collée à l'objet : sens imaginés disent « ça va glisser » ≥ 0,8 (aujourd'hui 0,52–0,59) |
| **7 — Jouer avec un but** | Push-T, jamais vu, modèle figé | aucun apprentissage : l'image donne le but, les sens le chemin, les sens réels vérifient après coup | d'abord le jeu (poussées, côtés, retour à l'objet) au-dessus du hasard, sur des épisodes jamais regardés ; la couverture ensuite |

Leçon reprise pour l'étape 5 : l'imagination par objet triche si elle démarre trop tôt (faux décalage de toute l'image). Elle commence donc par un échauffement sans déplacement, puis le déplacement, puis les horizons longs.

## 5. Ce qu'on ne refait pas

- Sous-buts codés à la main ; tête réflexe « état + but → geste » ; tout apprentissage sur Push-T.
- Un juge entraîné à part ; une ouïe entraînée à part.
- Le but sonore par « ressemblance globale » (il récompense le silence).
- Boîtes-objets spécialisées ; séquences de 32 images ; pyramide à niveaux ; œil gelé d'office.
- Push-T avec un T grossi.

## 6. Trois décisions à trancher avec toi

1. **Les pertes tirées des pixels** (`--move_w`, `--comove_w`, `--flow_w`). Elles sont dans la recette actuelle, mais tu as dit qu'elles ne sont pas JEPA, et aucune n'a fait bouger l'objet en imagination. *Ma proposition :* les remplacer par l'étape 5, qui fait leur travail dans les latents. C'est le seul endroit où j'« enlève ».
2. **La tête « gestes manquants »** (`--goal_w`). Elle est aussi dans la recette, mais c'est le réflexe que tu as rejeté. *Ma proposition :* ne pas la reprendre.
3. **L'œil pendant les étapes 5–6.** Le module marche aujourd'hui parce que l'œil est gelé. *Ma proposition :* geler l'œil au début de l'étape 5 (ce qu'on sait faire), puis le laisser bouger doucement, et comparer les deux dans le pilote.

## 7. Ordre de travail et coût

1. **Tu valides cette page** (et les trois décisions).
2. **Pilote court sur la partie nouvelle** : étapes 5 et 6 sur l'œil actuel (D17s) gelé, quelques heures de GPU. Il dit si l'imagination de tous les sens tient au contact avant de payer un entraînement complet.
3. **Grand entraînement à zéro**, étapes 0 à 6. Repère : la lignée actuelle a demandé 270k pas, soit environ 25 h sur A100. Mon estimation pour v2 est du même ordre, 20 à 30 h ; c'est une estimation, pas une mesure.
4. **Étape 7** sur des épisodes jamais regardés.

## 8. Ce que je ne sais pas

- Si l'étape « dans le ventre » change quelque chose : elle n'a jamais été isolée par un test.
- Si apprendre l'œil et l'imagination par objet ensemble améliore les latents ou les abîme.
- Si imaginer le toucher et le son depuis la scène imaginée suffit au contact, ou s'il faut aussi une imagination plus fine que des cases de 4 px.
