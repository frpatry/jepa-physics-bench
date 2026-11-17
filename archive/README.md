# archive — earlier stages of the project

Scripts and Colab notebooks from before the multisensory "baby" (June–September 2026):
continual learning (`cl_*`), driving world models (`drive*`, `driving_*`), object worlds
and slots (`objects*`, `slots*`), earlier Push-T attempts (`pusht_*`), active exploration
(`explore*`). Nothing here is used by the current code. See
[docs/HISTORIQUE.md](../docs/HISTORIQUE.md) (French) for what each stage taught.

They import each other by name, and a few import `vjepa.py` from the repository root, so
run them from this folder with the root on the path:

```bash
cd archive && PYTHONPATH=.. python slots_dyn.py
```

The notebooks were written when these files sat at the repository root; their paths need
the same adjustment.
