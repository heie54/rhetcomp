# Baseline v0.1 — Experience Pipeline

This repository implements the frozen pilot pipeline from
`baseline_v0.1_implementation_spec.md` through **Stage 5** (10-target pilot + Gate 5
freeze). The checked-in 20-source/10-target records under `data/pilot/input/` are
synthetic contract fixtures: they verify gate mechanics and are **not** ACL or NC_Physics
research data.

## Implemented stages

| Stage | Scope | Evidence |
|---|---|---|
| 0 | Scaffold: configs, schemas, artifact ids/hashes, logging, model-adapter + tokenizer/budget interfaces | `tests/test_stage0_contracts.py` |
| 1 | Ingestion + isolation: source normalization, target `VISIBLE/EVidence/GOLD` separation, Target Evidence Pack, Gate 1 audit | `src/cli/prepare_pilot.py`, `artifacts/audits/gate1.json` |
| 2 | Experience Compiler: single-pass extraction → exact-span validation → blind verifier → embedding pair retrieval → semantic adjudication → canonicalization → Stable Core/Supported Rare → fixed-budget library | `src/cli/compile_experience.py`, `artifacts/audits/gate2.json` |
| 3 | Baseline compilers: Raw / Summary / Guideline / Experience wrappers, shared corpus hash, cost logging, 4k budget | `src/cli/compile_representations.py`, `artifacts/audits/gate3.json` |
| 4 | One shared Writer for all 5 conditions, condition-invariant prompts, hash-equality gate | `src/cli/generate_pilot.py`, `artifacts/audits/gate4.json` |
| 5 | 10 targets × 5 conditions = 50 generations, pilot review (leakage, source-domain leak, distinctness, actionability, token/compute, format/citations), Gate 5 freeze | `src/cli/run_pilot.py`, `artifacts/evaluations/pilot_review.json`, `artifacts/audits/gate5.json` |

## Run the pipeline (mechanics mode)

Mechanics mode uses a deterministic mock adapter — it verifies the pipeline machinery,
never a formal experimental result. A real model run requires credentials + real data.

```powershell
$py = 'D:\AnacondaData\envs_dirs\py3.13\python.exe'
& $py -m src.cli.prepare_pilot                 # Gate 1 (Stage 1 artifacts)
& $py -m src.cli.compile_experience            # Gate 2 (Experience library)
& $py -m src.cli.compile_representations       # Gate 3 (all baseline representations)
& $py -m src.cli.generate_pilot --target ncphysics_fixture_001   # Gate 4 (one target, 5 conditions)
& $py -m src.cli.run_pilot                     # Gate 5 (50 generations + pilot review + freeze)
```

Run the verification suite:

```powershell
& $py -m unittest discover -s tests -v
```

## Role-scoped gold isolation

`src/runtime/` provides `CompilerDataAccess` / `WriterDataAccess` / `EvaluatorDataAccess`
with role-scoped roots. Compiler and writer packages never reference gold storage;
`TARGET_GOLD` is reachable only through the evaluator role. The Gate 1 audit
(`artifacts/audits/gate1.json`) enforces this statically and at runtime.

## Evaluation hooks preserved

Generation IDs are content-addressed hashes (`gen_…`) and do not reveal the condition;
`artifacts/generations/manifest.json` maps `target × condition → generation_id`.
`src/evaluation/review.py` keeps the pilot diagnostics (leakage, copy-rate, grounding,
tier, token/compute, compression) required for the later blinded evaluation stages.
