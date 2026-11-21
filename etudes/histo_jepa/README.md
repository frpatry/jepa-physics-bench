# Étude histo_jepa — un JEPA sur des images de cancer (Kaggle)

**Question.** Sur le jeu Kaggle *Histopathologic Cancer Detection* (220 025 images 96×96 avec étiquette, ~57 000 sans),
quel AUC obtient un petit ViT pré-entraîné **sans étiquettes** par un JEPA (appris sur ces images seulement, aucun poids
venu d'ailleurs), puis ajusté avec les étiquettes ? Et, quand les étiquettes sont rares (1 % et 10 %), que gagne-t-il sur
le même modèle appris **de zéro** ? C'est cette deuxième comparaison qui mesure ce qu'apporte le JEPA : avec toutes les
étiquettes la tâche est facile et presque tout modèle correct arrive haut.

**État (21 nov. 2026).** Essai de tuyauterie **réussi sur Kaggle sans GPU** (notebook `histo-essai`, `--mini 2000`, 137 s) pour
les bras pré-entraînés ; le bras « de zéro », ajouté ensuite, n'a pas encore tourné. **Pas encore vérifié** : le chemin GPU
(demi-précision, deux T4, un processus par GPU), la reprise entre sessions, le modèle à sa vraie taille et donc son coût.
Vérifié en local : la syntaxe, et les parties sans réseau (séparation, sous-ensembles, AUC, liste des entraînements, rapport).

## Protocole

1. **Séparation** faite une fois, graine fixe : 70 % entraînement / 15 % validation / 15 % test, enregistrée dans `split.csv`.
   Par **lame** si un fichier public « image → lame » est ajouté au kernel ; sinon au hasard par classe (limite écrite dans le rapport).
   Le test Kaggle officiel (sans étiquettes) ne sert pas à mesurer.
2. **Modèle** : ViT, patchs de 8 px → 144 jetons (taille dans `config.toml`).
3. **Pré-entraînement de type I-JEPA** : blocs masqués tirés au hasard **partout** dans l'image (aucune zone privilégiée),
   cibles = latents d'un encodeur-cible (moyenne glissante), **aucune perte en pixels**, aucune étiquette.
   Images : notre lot d'entraînement + les images de test Kaggle ; jamais notre validation ni notre test.
4. **Avec étiquettes** : `jepa_ft` (ajustement complet) et `jepa_gel` (encodeur gelé + petite tête, pour juger les latents bruts),
   aux fractions 1 %, 10 %, 100 % (emboîtées), plusieurs graines ; `zero` (le même modèle sans pré-entraînement, mêmes données,
   même nombre de pas, même recherche de taux) à 1 % et 10 % seulement. Une requête apprise lit les jetons : le modèle apprend seul où regarder.
5. **Augmentations** : les 8 retournements/rotations de 90° et un léger changement de couleur. L'image reste entière.
6. **Réglages** : 3 taux d'apprentissage essayés (graine 0), le meilleur **sur la validation** sert aux autres graines ; arrêt sur la validation.
   L'AUC du test n'est calculée qu'à la fin, dans le rapport, pour les réglages choisis.
7. **Résultats** : `resultats.md`, `resultats.csv`, `courbe.png` ; AUC d'une vue, de la moyenne des 8 vues, et de la moyenne des graines.
   `submission.csv` (prédictions du test Kaggle) est écrit mais **jamais envoyé par le code**.

## Lancer

Tout passe par `histo_jepa.py <étape>` ; chaque étape reprend où elle s'était arrêtée.

| étape | rôle |
|---|---|
| `prep` | décode les images une fois, fait la séparation |
| `chrono` | chronomètre ~60 pas de chaque sorte, estime les heures du plan complet |
| `pre` | pré-entraînement JEPA |
| `ft` | entraînements avec étiquettes (un processus par GPU) |
| `rapport` | tableau, courbe, `submission.csv` |
| `tout` | `prep` → `pre` → `ft` → `rapport` |

Sur Kaggle (kernels `histo-a` / `histo-b` qui se relaient, la sortie de l'un est l'entrée de l'autre) — voir l'en-tête de `kaggle_histo_push.sh` :

1. essai de tuyauterie **sans GPU** : `bash kaggle_histo_push.sh a tout 0 1 0 "--mini 2000 --out /kaggle/working/histo_mini"`
2. chronométrage (~15 min de GPU) : `bash kaggle_histo_push.sh a chrono 0 1 1`, puis fixer `pre.pas_total` (et la taille du modèle) dans `config.toml`
3. vrai run : `bash kaggle_histo_push.sh a tout 0 11 1`, puis `b tout 1 11 1`, etc. jusqu'à « TOUT EST FINI »

Le kernel clone le dépôt GitHub : le code doit y être poussé avant.

## Reproductibilité

Graines fixées ; `split.csv`, `config_utilisee.json`, le commit du code et l'empreinte de la config sont enregistrés avec chaque résultat.
Changer `config.toml` en cours d'étude est refusé (il faut un autre dossier de sortie). Le calcul sur GPU n'est pas reproductible au chiffre près.

## Limites connues

- Notre test est pris dans le « train » Kaggle : le test Kaggle vient d'autres lames et donne en général un score plus bas.
  Seule une soumission tardive donnerait un chiffre comparable au classement (non prévue sans accord).
- Les AUC de 1.000 du classement viennent probablement d'une fuite des étiquettes du test (non vérifié).
- Le repère `zero` est un ViT appris de zéro, faible quand les étiquettes sont rares : un écart dit « le pré-entraînement aide
  ce modèle », pas « il bat tout » (un réseau à convolutions ferait sans doute mieux que lui à 1 %). Il n'est pas lancé à 100 %
  (`ft.fractions_zero`), pour garder le budget GPU au pré-entraînement.
- Compétition de 2019 : le classement ne sert pas de repère, et avec toutes les étiquettes les écarts entre bons modèles sont de quelques millièmes.
- L'étiquette ne décrit que le carré central de 32×32 px ; rien dans le code ne le dit au modèle.
