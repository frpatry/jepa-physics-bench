# jepa-physics-bench — a JEPA "baby" that learns by playing

A personal research project on **JEPA-style world models** (Joint-Embedding Predictive
Architecture), in the spirit of Yann LeCun's program (LeJEPA, V-JEPA 2, AMI Labs): learn
without labels by **predicting in latent space**, then **act by planning in imagination**.

The repository name comes from its first experiment (an SSL benchmark on toy physics).
The project has moved a long way since; see the [history](docs/HISTORIQUE.md) (in French).

> Language note: this README is in English. Code comments, commit messages, logs and the
> older documents under `docs/` are in French.

## Goal

Build an artificial "baby", 100 % self-supervised, that:

1. learns to **see, hear, feel (touch, arm sense) and move** in a small simulated 2-D world
   (one hand, a few objects), through free play, never receiving a label;
2. builds a world model from it ("if I make this gesture, here is what I will see, hear
   and feel");
3. then solves a **task it was never trained on** (Push-T: push a T-shaped block onto a
   target) by **imagining** the consequences of candidate gestures and picking the best
   (MPC in imagination), with the model frozen.

The thread is **developmental**: each choice (order of stages, blurry vision first, the
hand before objects…) should be relatable to what is known about infants. Sourced notes:
[recherche_bebe_mains.md](recherche_bebe_mains.md),
[recherche_bebe_objets.md](recherche_bebe_objets.md),
[recherche_bebe_corps_monde.md](recherche_bebe_corps_monde.md).

## Ground rules

These constraints are deliberate; results should be read with them in mind.

- **Latent targets only.** No loss derived from pixels (optical flow, change maps…).
  Pixels, positions and oracles are **measurement instruments**, never training signals.
- **No imitation, no labels, no training on Push-T.** No reflex head "state + goal → action".
  A small reader trained on free play that reads the *imagined* latents is allowed.
- **A sense may be missing, never incoherent.** Eyes closed or a single sense is fine; an
  object that moves and a felt contact with no visible hand is not.
- **General, not specialised.** Anything added must make sense beyond the toy world
  (real video + sound is the long-term target). Hand-coded sub-goals are out.
- **Method over score.** Push-T is judged first on whether the baby *plays* with the object
  (contacts, pushes toward the goal), then on coverage (success = coverage ≥ 95 %).

## The model

- **World** (`av_world0.py`): 32×32 view, stereo audio, touch on the 4 sides of the hand,
  arm sense. Inert objects with friction; they never move on their own unless thrown.
- **Baby** (`av_phase0.py`): patches of 4 px → 64 cells per frame, 16 frames, width 192.
  One encoder per sense, fused in a causal predictor; EMA target encoder; SIGReg
  anti-collapse (official [LeJEPA](https://github.com/rbalestr-lab/lejepa)). The motor
  **command enters the predictor only**.
- **Curriculum**: 0a arm → 0b hands → 0c mobile tied to the gesture → 0d contact →
  0e sharp vision and curious free play (up to 60k steps), then **imagining objects**.
- **Per-object, step-by-step imagination** (`ObjImag`, in `av_objimag.py` and inside the
  baby): one gesture = one imagined frame. Cells are grouped by "what moves together";
  each group is *transported*, the background is copied, and a learned "clock" accounts
  for the time-dependent part of the latents. Purely latent.
- **Planner** (`av_plan0.py`): samples gesture sequences, rolls them out in imagination,
  scores them against the goal image.

## Where the project stands (17 November 2026)

**Short version: the baby now moves objects in its imagination; it does not yet solve
Push-T (0 % success).**

Exam used throughout: an object really displaced by 4–8 px over 8 steps — how much of that
displacement is imagined (amplitude), in which direction (cosine), and how much a
resting object drifts. Target: ≥ 50 % / ≥ 0.80 / ≤ 0.50 px.

| Model | Amplitude | Direction | Resting drift |
|---|---|---|---|
| Original "jump" predictor (270k steps) | ≈ noise | — | 2.3 px |
| Separate module on a frozen eye (v7) | 50–70 % | +0.80 to +0.96 | 0.15–0.44 px |
| Baby v2 (module inside, from scratch) | ≈ 30 % (plateau) | — | — |
| **Baby v3c**, 90k steps | **57 %** | **+0.62** | **0.54 px** |

Baby v3c is still training (target 110k steps, imagined touch/sound switched on at 96k).

### What was learned

1. **A predictor that jumps to a future frame never transported objects.** In imagination
   the object was frozen from the first step (2.7 px imagined for 6.0 px real, noise 2.0).
   Three targeted extra losses and a push-only fine-tune changed nothing: the limit was the
   construction of the imagination, not data or loss weights.
2. **Why: the push signal sits below the noise of the targets.** Two frames of a *resting*
   object already differ by 0.42–0.47 in its latents. About three quarters of that is a
   "clock" identical across scenes (cosine 0.97, predictable); the rest is a jitter of
   ≈ 0.15, against +0.03 for a 1–2 px displacement.
3. **Transport by object, step by step, fixes it.** Details that mattered: a long warm-up
   without displacement (otherwise the module cheats with a global blur that merely
   denoises), then chaining one step at a time instead of predicting a far horizon.
4. **Calm latents come first.** Trained inside the baby from scratch (v2), the same module
   plateaued at ≈ 30 %. The cause was measured, not guessed: the eye's latents jitter more
   at rest (0.251 vs 0.178 for the older lineage at 60k steps), visible from 5k steps.
   Many hypotheses were refuted on the way (sliding objects, grouping quality,
   thing/background split, training length, pixel-derived losses…). The culprits were three
   early ingredients *together* — extra weight on cells whose latents change, the multi-step
   rollout loss, and "guess from a single sense" (jitter at 5k steps: 0.135 without the
   three, 0.17–0.22 with any of them). Baby v3c drops them: 0.168 at 60k.
5. **Real senses are good, imagined ones are the open question.** Real touch and audio
   say "it slid" at 0.94–0.99 AUC; in the older lineage their imagined versions fell to
   chance at contact. v3c is the first run where they are imagined from the same imagined
   scene, switched on only once objects move.
6. **Push-T is not solved.** Best so far with a frozen model: the block is pushed, sometimes
   toward the goal (coverage up to 26–28 % max, finals 9–21 %), then the hand leaves and
   does not come back; no going around the block. The image-goal cost tends to rank plans
   that touch the block *worse*, and two identical runs can differ by 10 points — so any
   claim needs fixed seeds and more episodes.

Abandoned along the way: object "boxes" (slot attention) — worked on the toy world but
judged too specialised; hand-coded sub-goals; hiding the hand from the image.

## Running it

Training runs on a GPU (Colab A100 ≈ 1.5 steps/s; Kaggle T4 ≈ 0.3 steps/s with
`--accum 4`). Everything is resumable: rerun the same command.

```bash
git clone https://github.com/frpatry/jepa-physics-bench.git && cd jepa-physics-bench
pip install torch numpy scipy matplotlib scikit-learn
pip install git+https://github.com/rbalestr-lab/lejepa        # official SIGReg

python av_world0.py                                            # look at the world (local, seconds)

# Baby v3c: full curriculum from scratch, 110k steps, exams printed every 5k steps
bash run_bebe3.sh runs/bebe3c.pt 110000 --motion_w 0 --roll_w 0 --solo_share 0

# Does the eye jitter at rest? (frozen checkpoints, same scenes for all)
python av_motion_probe.py --ckpts runs/bebe3c_0e_60k.pt

# Push-T with the frozen baby, planning in its own imagination
python av_plan0.py --ckpt runs/bebe3c.pt --objimag self --thing_cost 1 --task pusht --goal_img 1 \
    --slide 1 --propose 0 --diag 0 --episodes 10 --plan_steps 300 --plan_h 8 --pols= --reach_diag 1 --gif 6
```

Kaggle background sessions (two kernels relaying each other from the latest checkpoint):
`kaggle_bebe3c.py`, `kaggle_bebe3c_push.sh`.

| File | Role |
|---|---|
| `av_world0.py` | the simulated world and its senses |
| `av_phase0.py` | the baby: model, curriculum, losses, exams |
| `av_objimag.py` | per-object step-by-step imagination (stand-alone module on a frozen eye) |
| `av_plan0.py` | planning in imagination, Push-T, GIFs |
| `run_bebe2.sh`, `run_bebe3.sh` | full training recipes |
| `av_*_diag.py`, `av_motion_probe.py` | frozen-model diagnostics (one question each, stated in the docstring) |

All scripts sit at the root and import each other by name; notebooks call them by path
after a `git pull`. That is why the code is not split into packages.

## Documentation

- [conception_bebe_v2.md](conception_bebe_v2.md) — design of the from-scratch, gradual training (French).
- [docs/HISTORIQUE.md](docs/HISTORIQUE.md) — project stages since June 2026 and lessons (French).
- [docs/ETAT_DU_PROJET.md](docs/ETAT_DU_PROJET.md), [docs/CARTE_DES_FICHIERS.md](docs/CARTE_DES_FICHIERS.md) —
  detailed state and file map **as of 9 October 2026** (French; superseded by this README for the current state).
- Commit messages are long and descriptive (`git log --oneline`): they are the detailed lab notebook.

## Caveats

Toy world, single researcher, mostly single seeds; several past results varied between
seeds. Numbers above are from the project's own exams, not from an external benchmark.
The comparable external reference for Push-T is DINO-WM (success ≈ 0.90, with a
pretrained DINOv2 encoder and offline data); this project is nowhere near it.
