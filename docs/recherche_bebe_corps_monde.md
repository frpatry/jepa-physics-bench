# Recherche — séparer le CORPS du MONDE, les SENS, et la 3D (2026-10-11)

Question de l'utilisateur : le bébé perçoit bien mais imagine mal les objets — est-ce un problème de sens ? d'« externalisation » du corps ?
La 3D aiderait-elle ? Constat de départ : objet perçu biaisé vers la main (B0 : jusqu'à +5–6 px quand il est loin), objet imaginé collé à la main ;
6 changements côté données/monde (calme, hors champ, T32, parent, blocs, surprise) sans effet sur l'imagination de l'objet.

## État actuel du modèle (lu dans `av_phase0.py`)
- Entrée : projection séparée par sens + étiquette de sens.
- Encodeur (`Enc4`, `--sep 1`) : attention À L'INTÉRIEUR d'un sens seulement (pas d'attention croisée) **mais mêmes poids pour les 4 sens**.
  ⚠ Donc la perception de l'objet (latents cibles) n'est PAS contaminée par les jetons du bras : le biais objet→main est un phénomène
  interne à la VISION (la main domine les traits visuels).
- Prédicteur (`PredAC`) : **fusion de tout d'un coup** (tous les sens + la commande motrice), un seul réseau imagine tout.
- La main et l'objet sont dans le MÊME sens (vision) : séparer les sens ne suffit pas à séparer main et objet.

## 1. Corps / monde chez le nourrisson
**Établi**
- Modèles internes du corps dans le cervelet : modèle direct (copie d'efférence -> conséquences sensorielles) et inverse
  (Wolpert, Miall & Kawato 1998). Auto-chatouillement atténué, la prédiction est SOUSTRAITE (Blakemore, Wolpert & Frith 1998) ;
  principe de réafférence (von Holst & Mittelstaedt 1950, non revérifié).
- Le corps se construit tôt, par contingence : synchronie visuo-tactile préférée dès ~12 h si l'image est corporelle (Filippetti et al. 2013) ;
  le nouveau-né résiste pour garder sa main en vue (van der Meer et al. 1995) ; calibration visuo-proprioceptive à 3 mois
  (Rochat & Morgan 1995) ; vidéo contingente de ses jambes discriminée à 5 mois, pas à 3 (Bahrick & Watson 1985).
- Le corps n'est relié à l'espace externe que plus tard : toucher codé sur le corps à 4 mois, dans l'espace externe vers 6 mois
  (Begum Ali, Spence & Bremner 2015) ; remappage postural 6,5–10 mois (Bremner, Holmes & Spence 2008) ; atteinte guidée 3–5 mois (von Hofsten).
- Objets : principe de CONTACT (Spelke 1990).

**Débattu**
- Distinction soi/externe néonatale (Rochat & Hespos 1997, petit échantillon).
- Ce que reçoit le système « monde » : la décharge corollaire atteint des aires perceptives (Sommer & Wurtz 2008) mais pour ANNULER le
  mouvement auto-produit ; pour un objet poussé, la cause physique est l'ÉTAT DU CORPS (main, contact), pas la commande.
  Lecture raisonnable : commande -> modèle du corps ; monde <- état du corps (+ annulation de la réafférence).

## 2. Séparation des sens
**Établi**
- Aires primaires distinctes ; voies dorsale/ventrale (classique). Colliculus supérieur du chaton : AUCUN neurone multisensoriel les
  ~10 premiers jours ; intégration plus tard et dépendante de l'expérience (Wallace & Stein 1997 ; Stein, Stanford & Rowland 2014).
- Intégration statistique « optimale » très tardive : visuo-haptique vers 8–10 ANS, un sens domine avant (Gori et al. 2008) ;
  même stéréo+texture pas avant ~12 ans (Nardini, Bedford & Mareschal 2010) — un sens CALIBRE l'autre.
- Nuance : relations AMODALES (synchronie, contingence) détectées très tôt. Schéma : estimations par sens -> liens par
  synchronie/contingence -> fusion statistique tardive.

**Apprentissage machine (vérifié)** — compétition / paresse de modalités : Wang, Tran & Feiszli 2020 (le meilleur unimodal bat souvent
le multimodal) ; Huang et al. 2022 (l'encodeur n'apprend qu'un sous-ensemble des modalités) ; Peng et al. 2022 (OGM-GE) ; Du et al. 2023 ;
fusion intermédiaire par goulot d'attention > fusion naïve (Nagrani et al. 2021). Transposé à notre cas : la main (contrôlable, dense,
prévisible) est le facteur DOMINANT, l'objet (rare, ne bouge que poussé) le perdant. Analogues directs corps/monde :
Choi et al. 2019 (un modèle de dynamique attentif localise l'agent sans supervision) ; Iso-Dream (Pan et al. 2022 : branches séparées
contrôlable / non contrôlable, meilleures prédictions).

## 3. Profondeur / 3D
**Établi**
- Indices cinétiques/looming dès ~1 mois (évitement robuste plus tard, débattu) ; stéréopsie en moyenne à 16 semaines
  (Held, Birch & Gwiazda 1980) ; indices picturaux vers 5–7 mois (Granrud & Yonas 1984). Stéréopsie ≈ début de l'atteinte dirigée
  (lien Granrud 1986, non revérifié).
- Ségrégation des objets par l'ESPACE et le MOUVEMENT, pas par couleur/forme : mouvement commun -> unité à 4 mois
  (Kellman & Spelke 1983) ; deux objets séparés en profondeur ou en mouvement relatif = deux unités, ADJACENTS = un bloc
  (von Hofsten & Spelke 1985).
- ⇒ Une main en contact avec l'objet, même profondeur, sans mouvement relatif, est fusionnée même par un vrai bébé. Ce qui sépare main
  et objet : d'abord le MOUVEMENT RELATIF, ensuite la profondeur. La stéréo aide pour l'occlusion/l'ordre en profondeur, pas au contact.

**ML** : SAVi++ (Elsayed et al. 2022) — la profondeur aide l'émergence d'objets, mais comme CIBLE supervisée (LiDAR) = raccourci non
fidèle. Version fidèle : deux « yeux » décalés, disparité en ENTRÉE, vers « 4 mois » ; vérifier qu'à 32×32 la disparité
péripersonnelle fait plusieurs pixels.

## 4. Recommandations (impact / coût / fidélité)
1. **Séparer modèle du CORPS et modèle du MONDE** — fort / moyen / fidèle. Corps (« cervelet ») : prédit bras, toucher, main visuelle à
   partir de la commande, entraîné en premier. Monde : ne reçoit PAS la commande ; reçoit l'état du corps (main, contact) et ce que le
   corps n'explique pas (réafférence). Seul chemin causal vers l'objet = le contact (Spelke). Analogue : Iso-Dream.
2. **Encodeurs par sens (poids distincts), fusion tardive par goulot** — moyen à fort / faible à moyen / fidèle. Diagnostic préalable
   possible (ablation, attention).
3. **Masquage naturel de modalités entières** (main hors champ, obscurité) — moyen / faible / fidèle.
4. **Stéréo à deux yeux vers « 4 mois »** — moyen / faible en simulation / fidèle ; utile pour l'occlusion, pas au contact ; le mouvement
   relatif d'abord (déjà fourni par les blocs).
5. **À éviter (non fidèles)** : profondeur en cible supervisée ; modulation de gradient par modalité ; masques/identités d'objets fournis ;
   commande donnée au modèle du monde « pour aider ».

## Règles de conception
1. Le corps d'abord (modèle du corps entraîné tôt, par contingence).
2. La commande ne va qu'au corps ; le monde reçoit l'état du corps.
3. Soustraire le soi : le monde apprend sur ce que le corps n'explique pas.
4. Un objet ne bouge que par contact (via l'état du corps ou d'autres objets).
5. Un encodeur par sens, fusion tardive par goulot.
6. Garder les estimations par sens lisibles ; les conflits entre sens sont un signal d'apprentissage.
7. Surveiller la compétition : mesurer ce que chaque facteur (main, objet) et chaque sens apporte.
8. Mouvement relatif avant profondeur.
9. Stéréo vers « 4 mois », en entrée jamais en cible, disparité exploitable à 32×32.
10. Corps dans l'espace externe plus tard (remappage).
11. Étiqueter chaque ajout : « rattaché au bébé (réf.) » ou « raccourci d'ingénierie » ; les seconds interdits dans la version de référence.

Références non revérifiées en ligne : Granrud 1986 ; von Holst & Mittelstaedt 1950 ; Spelke 1990 ; Baillargeon / Piaget ;
Mishkin & Ungerleider ; Milner & Goodale.
