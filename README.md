# Baseline v0.1 — Stage 0/1

This directory implements only Stage 0 (scaffold and contracts) and Stage 1 (data
ingestion and isolation) from `baseline_v0.1_implementation_spec.md`.

The checked-in 20-source/10-target records under `data/pilot/input/` are synthetic
contract fixtures. They verify Gate 1 mechanics and are not ACL or NC_Physics research
data.

Run the one-command pilot preparation with the required local Python environment:

```powershell
& 'D:\AnacondaData\envs_dirs\py3.13\python.exe' -m src.cli.prepare_pilot
```

It writes normalized sources, the three physically separate target namespaces, Target
Evidence Packs, and `artifacts/audits/gate1.json`. A failed Gate 1 returns a nonzero exit
status.

Run the focused verification suite:

```powershell
& 'D:\AnacondaData\envs_dirs\py3.13\python.exe' -m unittest discover -s tests -v
```

No Experience Compiler or Writer implementation exists in this stage.
