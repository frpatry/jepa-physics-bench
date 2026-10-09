# Carte des fichiers

Tout le code est à la racine du dépôt : les scripts s'importent entre eux par leur nom
et les notebooks Colab les appellent par chemin après un `git pull`. Les fichiers ne sont
donc **pas** rangés en dossiers ; cette carte les regroupe par **époque / axe de
recherche**, dans l'ordre chronologique.

Légende : **actuel** = utilisé dans la phase en cours ; **socle** = ancien mais encore
importé par le code actuel (ne pas supprimer) ; **historique** = expérience close,
gardée pour la trace et la reproductibilité ; **instrument** = outil de mesure ou de
diagnostic (jamais une méthode d'apprentissage).

Les notebooks `*_colab.ipynb` commencent tous par une cellule qui clone ou met à jour le
dépôt (`git pull`), puis lancent les scripts.

---

## 7. Avenue multisensorielle « bébé » (`av_*`, octobre 2026) — ACTUEL

Un bébé artificiel qui apprend à voir, entendre, sentir et agir, puis planifie.
Détails : [ETAT_DU_PROJET.md](ETAT_DU_PROJET.md).

**Phase 0 « mes mains » et boîtes-objets — le cœur actuel**

| fichier | rôle | statut |
|---|---|---|
| `av_world0.py` | monde de phase 0 : 4 sens + commande, étapes 0a → 0e, monde lisible, table qui déborde | actuel |
| `av_phase0.py` | entraîneur du bébé de phase 0 (encodeur 4 sens, prédicteur conditionné par la commande, examens de surprise) | actuel |
| `av_plan0.py` | imagination décodée + planification « pousse l'objet sur la cible » ; `--obj_diag` | actuel |
| `av_slots0.py` | boîtes-objets au-dessus de l'œil figé de la phase 0 (essai 3 en cours) | actuel |
| `av_inv_diag.py`, `av_mob_diag.py` | diagnostics « deviner son geste » et « mobile » sur un instantané de phase 0 | instrument |
| `recherche_bebe_mains.md`, `recherche_bebe_objets.md` | recherches sourcées sur le nourrisson (fondements de la phase 0 et des boîtes) | actuel |
| `conception_boites_objets.md` | conception des boîtes-objets | actuel |

**Phase 1 (perception : vision + ouïe) et phase 2 (action + toucher) — octobre 2026**

| fichier | rôle | statut |
|---|---|---|
| `av_jepa.py` | monde AV (disques à masse/matériau cachés, son synthétique) + AV-JEPA ; fournit le rendu audio et les masques | socle |
| `av_dev.py` | phase 1 développementale (son seul → flou → net), tests adaptés à l'âge | socle |
| `av_dev_long.py` | run long développemental en flux continu ; recette stable (SIGReg faible + lr cosinus + encodeurs séparés) ; lecteur de positions | socle |
| `av_world5.py` | monde v5 varié (formes, rotation, objets au repos, main qui babille) ; formes réutilisées par la phase 0 | socle |
| `av_act.py` | phase 2 : prédicteur conditionné par l'action, toucher, planification MPC (monde à frottement) | historique (importé par des diagnostics) |
| `av_vjepa2.py`, `av_fusion.py` | V-JEPA 2 gelé comme « cortex visuel » ; JEPA de fusion vision + ouïe | historique |
| `av_loc_vj2.py`, `av_fair_eval.py` | localisation avec V-JEPA 2 ; évaluation équitable à peu d'étiquettes | historique |
| `av_world_check.py`, `av_test_check.py`, `av_bind_check.py` | plafonds : l'information existe-t-elle dans le monde ? les tests sont-ils solubles ? le liage est-il calculable ? | instrument |
| `av_dev_diag.py`, `av_vis_std.py`, `av_hand_diag.py`, `av_locality.py`, `av_layer_diag.py`, `av_res_test.py` | diagnostics de la perception (localisation, effondrement, main, localité des tokens, précision par couche, résolution) | instrument |
| `av_jepa_colab.ipynb` | notebook des premières expériences AV | historique |

---

## 6. Push-T (`pusht_*`, juillet 2026) — historique

Tentative de se mesurer à DINO-WM (taux de succès 0,90) sur la tâche Push-T (pousser un
bloc en T sur une cible).

| fichier | rôle | statut |
|---|---|---|
| `pusht_data.py` | collecte de données de jeu depuis gym-pusht | historique |
| `pusht_plan.py` | planification MPC (protocole DINO-WM), oracle ; importé par les autres scripts Push-T | historique (socle de l'époque) |
| `pusht_readout.py`, `pusht_pose.py` | décodeur riche sur slots gelés ; sonde de pose (x, y, angle) | historique / instrument |
| `pusht_posewm.py` | world model fondé sur la pose (perception géométrique, dynamique et planification en nombres) | historique |
| `pusht_jepa.py`, `pusht_jepa2.py` | JEPA + SIGReg officiel conjoint, puis en 2 étapes (encodeur gelé + dynamique) | historique |
| `pusht_vjepa2.py` | ablation : V-JEPA 2 de Meta gelé à la place de l'encodeur maison, dynamique + planificateur | historique (résultats de planification non rapportés) |
| `pusht_jepa_colab.ipynb`, `pusht_pose_colab.ipynb`, `pusht_posewm_colab.ipynb` | notebooks | historique |

---

## 5. Objets émergents : slots (`slots*`, juillet 2026) — historique, recette de référence

Découverte non supervisée des objets (Slot Attention), puis dynamique, puis action.
La recette de `slots.py` est celle qu'on réutilise dans les boîtes-objets.

| fichier | rôle | statut |
|---|---|---|
| `slots.py` | Slot Attention « peel » : un slot par objet sans supervision (erreur 0,048) ; importé par les autres `slots_*` | historique (référence) |
| `slots_dyn.py` | dynamique sur les slots, chocs, rollout | historique |
| `slots_act.py` | System-2 sur slots : toy-Push, MPC dans l'espace décodé | historique |
| `slots_pusht.py` | la recette portée sur Push-T 96×96 | historique |
| `slots_sharp.py`, `slots_orient.py` | bacs à sable : netteté (objets qui se touchent), orientation (le T tourné) | historique / instrument |
| `slots_colab.ipynb`, `slots_dyn_colab.ipynb`, `slots_act_colab.ipynb`, `slots_pusht_colab.ipynb`, `slots_sharp_colab.ipynb`, `slots_orient_colab.ipynb` | notebooks | historique |

---

## 4. Monde d'objets et apprentissage actif (fin juin – début juillet 2026) — historique

| fichier | rôle | statut |
|---|---|---|
| `objects.py` | disques qui rebondissent et se percutent + V-JEPA + sondes de compréhension | historique |
| `objects_plan.py` | System-2 : traverser des obstacles mobiles en imaginant leur futur | historique |
| `explore.py` | apprentissage actif (curiosité) contre passif, à données égales, depuis les pixels | historique |
| `explore_state.py` | même question sur l'état exact (sans pixels) | instrument |
| `objects_colab.ipynb`, `explore_colab.ipynb` | notebooks | historique |

---

## 3. Vidéo réelle, V-JEPA et conduite (juin 2026) — historique

| fichier | rôle | statut |
|---|---|---|
| `vjepa.py` | **V-JEPA fidèle** (encodeur sur tokens visibles, prédicteur attentionnel, masques en tubelets) et **SIGReg** officiel ; importé partout, y compris par la phase 0 | **socle** |
| `lejepa_video.py` | LeJEPA sur vraie vidéo (UCF101), chargeurs vidéo | socle de l'époque (importé par `driving_*`) |
| `drive.py`, `drive_model.py`, `drive2.py`, `drive2_model.py` | jouets de conduite 2D : monde, world model JEPA, conducteur appris | historique |
| `drive_plan.py` | planificateur MPC latent (System-2) sur la conduite jouet | historique |
| `driving_transfer.py`, `driving_rollout.py`, `driving_risk_viz.py` | dashcam réelle (Nexar) : détection de danger par transfert, prédiction de collision par rollout, visuel de risque | historique |
| `drive_colab.ipynb`, `lejepa_video_colab.ipynb`, `vjepa2_real_colab.ipynb` | notebooks | historique |

---

## 2. Banc SSL anti-effondrement sur physique jouet (juin 2026) — historique

| fichier | rôle | statut |
|---|---|---|
| `cl_sslbench.py` | JEPA sur trajectoires de balles sous gravité ; compare `none` / `vicreg` / SIGReg ; sonde de la gravité `g` | historique (première expérience publiée du dépôt) |
| `cl_sslbench_colab.ipynb` | notebook | historique |

---

## 1. Apprentissage continu (`cl_*`, juin 2026) — historique

Point de départ du projet : l'oubli catastrophique, d'abord sur un modèle de langage à
routage gelé, puis sur des JEPA.

| fichier | rôle | statut |
|---|---|---|
| `cl_scale.py` | modèle de langage, routage gelé top-K contre dense (vrai texte) ; chargeurs de données réutilisés | historique |
| `cl_jepa.py`, `cl_jepa_text.py` | apprentissage continu sur un JEPA (données continues, puis texte) | historique |
| `cl_jepa_demo.py`, `cl_barlow.py`, `cl_multiview.py` | Barlow-JEPA, replay, multi-vue « 4 réseaux » | historique |
| `cl_jepa_probe.py`, `cl_5probe.py` | métrique d'oubli par sonde gelée ; sonde des 5 composants du multi-vue | instrument |

---

## Autres fichiers

| fichier | rôle |
|---|---|
| `README.md` | présentation et état actuel |
| `docs/` | cette documentation |
| `session_handoff.md` | ancienne note de reprise (juillet 2026, époque slots) ; conservée pour l'histoire, **ne reflète plus l'état actuel** |
| `.gitignore` | ignore les sorties générées (journaux, poids, caches, `runs_*/`, `*.json`) |

## Doublons et fichiers à surveiller (rien n'a été supprimé)

- `cl_scale 2.py` (non suivi par git) : copie **plus ancienne** de `cl_scale.py` (noms de
  jeux de données Hugging Face non « namespacés », sans le domaine code). Aucun script ne
  l'importe. Doublon probable laissé par une copie de fichier.
- Les notebooks `cl_*_colab.ipynb` non suivis (sauf `cl_sslbench_colab.ipynb`) datent
  d'avant le dépôt git : ils recopient le code des scripts dans des cellules
  `%%writefile`, au lieu de faire un `git pull`. Ils peuvent diverger des scripts suivis.
- `lejepa_video.py` contient un JEPA simplifié remplacé par `vjepa.py`, mais il est
  encore importé (chargeurs vidéo) par les scripts `driving_*`.
