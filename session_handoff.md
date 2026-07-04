# Session Handoff — JEPA / World Model / Object-Centric

Reprise propre du projet. Lire ceci en premier.

---

## But du projet
Explorer des **world models façon JEPA** (vision LeCun) : perception → prédiction → action/planification (System-2).
Parti d'un banc SSL/physique, évolué vers : *un world model peut-il comprendre les objets + leur dynamique, puis agir ?*
Récemment : recherche d'un **edge de recherche** et construction d'une **fondation object-centric émergente**.

Repo GitHub : **frpatry/jepa-physics-bench** (public). Tout tourne sur **Colab GPU** (CPU local trop lent).

---

## OÙ ON EN EST (immédiat)
**✅ FONDATION OBJECT-CENTRIC ACQUISE (run GPU 12).** `slots.py` découvre les objets sans supervision :
**erreur position 0.048** (≈2 px/48), séparation 0.95, **sur scènes VARIABLES** (1-4 objets, couleurs au hasard).
Un slot par disque quel que soit leur nombre, prises excédentaires vides (le comptage émerge), fond au reliquat.

**Recette gagnante** (12 runs, chaque échec a fermé une échappatoire économique) :
`--mode peel` (épluchage récursif, IDÉE DE L'UTILISATEUR : le cerneur prend un objet, scope×(1−masque), suivant)
+ inits apprises distinctes par round (sinon rounds clones) + `--loss mix` (vraisemblance de mélange par pixel,
interdit le partage doux) + reliquat = couleur unie apprise (ferme la porte dérobée) + décodeur SBD 1×1 faible
+ goulot slot_dim 16 (plancher : 8 dégénère) + `--vary 1` (scènes variables — LE déclencheur final : sans ça,
des masques-gabarits indépendants de l'image paient autant que percevoir).

Commande du run gagnant : `python -u slots.py --n 5000 --n_obj 4 --H 48 --K 5 --steps 20000 --bs 64 --mode peel --loss mix --vary 1`

**Limite résiduelle** : deux disques qui se touchent fusionnent parfois dans une prise — le statique est
fondamentalement ambigu là-dessus ; c'est le mouvement qui désambiguïse (common fate) → prochaine étape.

---

## LES 3 MURS DIAGNOSTIQUÉS (résultats clés, honnêtes)
1. **Perception** : lire une position **précise** depuis un latent compressé/poolé, auto-supervisé, sans échafaudage = **mur**. (Le pooling dilue la position — récurrent : dashcam, objets, explore.)
2. **Prédiction honnête** : le fameux « 0.70 » d'objects.py était l'encodeur **bidirectionnel qui TRICHAIT** en voyant le futur ; en causal honnête (`context_latents`) ça **s'effondre**. Prédire le latent brut → collapse vers la moyenne.
3. **Exploration** : la **curiosité naïve** (argmax du désaccord d'ensemble, `explore_state.py`) est **PIRE que le hasard** (s'obsède sur quelques objets, couverture pauvre).

---

## INSIGHTS STRATÉGIQUES
- **L'efficacité-données est le vrai gap** (un enfant apprend la physique en ~mois, pas en millions d'heures) → l'échelle est une **béquille**, pas la solution. L'edge doit venir de **COMMENT on apprend**, pas de la taille.
- **Standard de l'utilisateur** : émergent, **sans échafaudage/triche**, transférable. (A rejeté CoordConv : casse l'équivariance par translation, pas transposable au réel.)
- **V-JEPA 2.1** (Meta) **A résolu** la perception dense/spatiale précise et **émergente** à l'échelle → notre mur est **réel mais pas fondamental**. Valide l'idée « **V-JEPA pré-entraîné = les yeux** ».
- **Chemin choisi** : object-centric **émergent** (Slot Attention) = le seul respectant le standard « émergent, sans triche » à notre échelle jouet.

---

## FICHIERS PRINCIPAUX
- **`vjepa.py`** — V-JEPA fidèle : encodeur-contexte sur tokens **visibles**, prédicteur **attentionnel**, masquage tubelets+temporel, SIGReg, sans EMA. + `probe`, `attentive_probe`, `context_latents` (encodage honnête préfixe-seul).
- **`objects.py`** — monde d'objets (disques colorés, rebonds/collisions) + V-JEPA SSL + sondes de compréhension (position, prédiction du futur). `--bounce 0` = ligne droite ; `--viz` = GIF réel vs imaginé.
- **`objects_plan.py`** — System-2 : agent navigue parmi obstacles mobiles via futur imaginé. **Résultat : ne bat pas System-1** (mur perception/prédiction).
- **`drive_plan.py`** — System-2 conduite jouet (MPC). Prédicteur trivial (`g` ≈ identité).
- **`driving_transfer.py`** — transfert UCF→dashcam Nexar (meilleur **~0.78 à H=32** ; H=96 dilue via mean-pool ; sonde attentive a échoué/surappris).
- **`driving_rollout.py`** — prédiction collision par rollout (au hasard = mur prédiction honnête).
- **`explore.py` / `explore_state.py`** — actif (curiosité) vs passif. **Résultat : curiosité naïve ≤ hasard.**
- **`slots.py`** — **ACTUEL, VALIDÉ** : découverte d'objets non supervisée, err 0.048 sur scènes variables. Modes : `par` (Slot Attention classique), `seq` (explaining away), **`peel` (gagnant)** ; `--loss mix`, `--bg const`, `--vary 1`, inits par round. Métrique : erreur position Hungarian + séparation des masques.
- **Notebooks dédiés** (chacun : cellule 1 = git pull) : `drive_colab.ipynb`, `objects_colab.ipynb`, `explore_colab.ipynb`, `slots_colab.ipynb`.

---

## WORKFLOW (important)
- **Colab GPU** obligatoire. Cycle : *je push → l'utilisateur re-roule la cellule 1 (git pull) → lance la cellule d'expérience.*
- ⚠️ **Piège récurrent** : après `git pull`, le **notebook OUVERT reste périmé** (Colab garde l'ancienne version en mémoire). L'utilisateur a plusieurs fois roulé les **anciens args** → toujours **coller la commande explicite** avec tous les args, ou recharger le notebook.
- **Notebooks dédiés courts** par expérience (ne pas enterrer les cellules).
- **Déléguer les smoke-tests / recherches verbeuses à des sous-agents** (préserver le contexte principal — préférence utilisateur). Smoke Slot Attention sur CPU local = LENT (~18 min) → préférer un **check de formes** local rapide + le vrai run sur GPU.
- **Sauver l'état en mémoire** à chaque étape clé (`~/.claude/projects/-Users-...-JEPA-type-learning/memory/`). Index dans `MEMORY.md`.
- Commits : finir par `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## MÉMOIRES CLÉS (persistent entre sessions, voir MEMORY.md)
`cl-object-centric-foundation` (actuel), `cl-active-learning-edge`, `cl-objects-world`, `cl-vjepa-faithful-and-system2`, `cl-drive-plan-next`, `cl-workflow-use-subagents`, `cl-git-colab-workflow`, `cl-ssl-bench-ami`.

---

## PROCHAINES ACTIONS
1. **Dynamique sur les slots** : séquences du monde qui bouge (disques avec vitesses/rebonds, cf. objects.py),
   encoder frame t → slots → prédire slot_t+1 (petit MLP partagé) → décoder t+1 avec la même recette.
   Double gain attendu : (a) prédiction honnête du futur sur état object-centré (là où le latent global
   s'effondrait) ; (b) le common fate devrait séparer les objets qui se touchent (limite résiduelle du statique).
2. Puis re-tester **l'edge** (apprentissage actif, world model sur slots, planner) sur cette fondation saine.

Objectif affiché de l'utilisateur : produire de la **connaissance réutilisable** (ligne LeCun), pas sur-scaler Meta.
