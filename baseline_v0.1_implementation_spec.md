# Baseline v0.1 Implementation Spec

> Status: **FROZEN FOR PILOT**
>
> Purpose: implement the smallest reproducible experiment that can falsify the current
> Scientific-Writing Experience hypothesis. Do **not** expand this into a multi-agent
> workspace, long-term memory product, full-paper writer, or retrieval product.

---

## 1. Research objective

### Main research question

When target factual evidence, source corpus, writer inference, and conditioning budget
are controlled, does a provenance-grounded, cross-document library of rhetorical writing
strategies induced from scientific Introductions in one field improve Introduction writing
in a held-out field beyond:

1. raw source exemplars;
2. matched abstractive summaries;
3. source-conditioned generated writing guidelines?

### Frozen experiment paradigm

- **Source field:** ACL/NLP
- **Target field:** Physics
- **Source corpus:** ACL 2024 main-conference Introduction exemplars
- **Target set:** NC_Physics official test set
- **Primary paradigm:** reusable cross-field library
- **Do not implement target-specific Experience extraction in v0.1.**

The experiment studies the representation used to condition the Writer, not reference
retrieval quality.

---

## 2. Hard experimental invariants

These are not optional implementation details.

1. Every condition receives the **same Target Evidence Pack** for a target paper.
2. Gold Introduction is physically separated from compiler/writer-accessible data.
3. Writer model, system prompt, decoding parameters, output limit, and number of calls are
   identical across conditions.
4. Writing-conditioning budget is identical across:
   - Raw
   - Summary
   - Generated Guideline
   - Experience
5. Default frozen budgets:
   - Target Evidence Pack: **8k tokens**
   - Writing-conditioning representation: **4k tokens**
6. Summary / Guideline / Experience are compiled from the **same source corpus**.
7. Generated Guideline should use the same compiler model and approximately matched
   source-reading / LLM-call compute as Experience.
8. All source spans stored as evidence must be programmatically checkable against the
   original normalized Introduction text.
9. Do not expose source-domain factual content through the Target Evidence Pack.
10. Record compiler-side cost as well as Writer-side cost.

---

## 3. Non-goals for v0.1

Do not implement:

- multi-agent orchestration;
- research workspace UI;
- vector knowledge base as a product feature;
- long-term user memory;
- target-specific reference retrieval;
- full-paper generation;
- Related Work generation;
- experience graphs / hierarchy;
- fixed style / structure / argument taxonomy;
- mandatory rationale / anti-pattern / negative examples;
- numerical self-reported confidence;
- majority voting as a main-system component;
- 3-run extraction as the default pipeline.

---

## 4. Pilot before full experiment

### Pilot scope

Start with:

- **20 ACL source Introductions**
- **10 NC_Physics targets**
- all 5 mandatory conditions

Purpose of the pilot:

- validate data isolation;
- validate exact-span provenance;
- validate compiler outputs;
- validate token-budget matching;
- detect leakage;
- ensure the Writer harness is condition-invariant;
- inspect whether outputs are meaningfully different before paying for the full run.

The pilot may fix implementation bugs and prompts, but must not silently change the
research question or baseline definitions.

### Full v0.1 scope after pilot gate

- **200 ACL 2024 main-conference source papers**
- **100 NC_Physics official test targets**

---

## 5. Mandatory experimental conditions

### B0 — Evidence-only

Input to Writer:

```text
Task
+ Target Evidence Pack
```

No additional writing reference.

This is the real control condition. Do not call it "No Reference" in code or reports.

### B1 — Raw Source Exemplars

```text
Task
+ Target Evidence Pack
+ token-budgeted raw ACL/NLP exemplar Introduction text
```

### B2 — Matched Abstractive Summary

```text
ACL source corpus
→ Summary Compiler
→ 4k-token source-corpus summary
```

Writer receives:

```text
Task
+ Target Evidence Pack
+ Summary
```

The Summary objective is source information compression, not writing-strategy induction.

### B3 — Compute-Matched Generated Guideline

```text
ACL source corpus
→ Guideline Compiler
→ conditional scientific-writing guidelines
```

Writer receives:

```text
Task
+ Target Evidence Pack
+ Generated Guideline
```

This is the most important adversarial baseline. It tests whether Experience is merely a
more expensive generated writing manual.

### Ours — Provenance-Grounded Writing Experience

```text
ACL source corpus
→ Experience Compiler
→ verified / consolidated Experience Library
```

Writer receives:

```text
Task
+ Target Evidence Pack
+ fixed-budget Experience Library
```

---

## 6. Core domain objects

### 6.1 SourcePaper

```json
{
  "source_id": "acl2024_xxx",
  "title": "...",
  "authors": ["..."],
  "venue": "ACL 2024",
  "track": "...",
  "introduction": {
    "normalized_text": "...",
    "paragraphs": [
      {
        "paragraph_id": 1,
        "sentences": [
          {
            "sentence_id": 1,
            "text": "...",
            "char_start": 0,
            "char_end": 120
          }
        ]
      }
    ]
  },
  "document_hash": "..."
}
```

Only the compiler-side pipeline needs the source Introduction.

### 6.2 TargetPaper

Keep three namespaces physically separate.

```text
TARGET_VISIBLE/
TARGET_EVIDENCE/
TARGET_GOLD/
```

Suggested logical form:

```json
{
  "target_id": "ncphysics_xxx",
  "visible": {
    "title": "...",
    "abstract": "..."
  },
  "evidence": {
    "non_intro_sections": {},
    "reference_metadata": []
  },
  "gold": {
    "introduction": "..."
  }
}
```

`gold` must never be passed to compilers or Writer.

### 6.3 TargetEvidencePack

```json
{
  "target_id": "ncphysics_xxx",
  "budget_tokens": 8000,
  "content": "...",
  "source_fields": [
    "title",
    "abstract",
    "non_intro_body",
    "reference_metadata"
  ],
  "input_hash": "..."
}
```

Every condition must reuse the exact same serialized pack for a target.

---

## 7. Frozen Experience semantic schema

```json
{
  "experience_id": "exp_001",
  "observed_pattern": "A source-local descriptive account of the rhetorical action observable in the cited span, without asserting author intention.",
  "strategy": "An actionable generalized writing strategy inferred from the observed pattern.",
  "applicable_when": "The writing conditions under which the strategy is expected to be useful.",
  "evidence": [
    {
      "source_id": "acl2024_xxx",
      "location": {
        "section": "Introduction",
        "paragraph": 3,
        "sentence_start": 1,
        "sentence_end": 2
      },
      "span": "Exact verbatim source text.",
      "support_relation": "instantiates_observed_pattern"
    }
  ],
  "grounding_status": "unverified"
}
```

Allowed `grounding_status` values:

```text
unverified
span_verified
support_verified
rejected
```

### Important epistemic rule

The system is allowed to claim:

```text
span supports observed_pattern
```

and, after bounded verification:

```text
observed_pattern is a reasonable basis for strategy induction
```

It must **not** claim:

```text
source span logically entails a universal writing strategy
```

### Derived metadata — keep outside semantic schema

```json
{
  "experience_id": "exp_001",
  "distinct_source_count": 3,
  "cluster_id": "cluster_014",
  "tier": "stable_core",
  "verifier_result": "...",
  "verifier_score": null,
  "run_support": null
}
```

Do not add these to the semantic Experience object.

---

## 8. Experience Compiler pipeline

Frozen main pipeline:

```text
Scientific Introduction source corpus
        ↓
single-pass open atomic extraction
        ↓
deterministic exact-span validation
        ↓
blind support verifier
        ↓
verified candidate pool
        ↓
embedding candidate-pair retrieval
        ↓
semantic equivalence adjudication
        ↓
canonicalization + provenance union
        ↓
Stable Core / Supported Rare
        ↓
fixed-budget Experience Library
```

### 8.1 Single-pass open atomic extraction

Requirements:

- no fixed content taxonomy;
- one candidate ≈ one rhetorical decision / pattern;
- return `observed_pattern`, `strategy`, `applicable_when`, and exact evidence;
- do not request numerical confidence;
- do not request author intention unless directly observable.

### 8.2 Deterministic exact-span validation

No LLM.

Check:

- source_id exists;
- paragraph/sentence coordinates exist;
- evidence span occurs exactly in normalized source text;
- coordinates and span agree.

State transition:

```text
unverified → span_verified
```

Failure:

```text
→ rejected
```

### 8.3 Blind support verifier

Verifier input only:

```text
evidence span
observed_pattern
strategy
applicable_when
```

Do not provide extractor rationale / hidden reasoning.

Verifier must separately judge:

1. Does the span support the `observed_pattern`?
2. Is the `strategy` a reasonable, bounded generalization from that observation?

Recommended structured output:

```json
{
  "observation_support": "supported | partial | unsupported",
  "strategy_generalization": "reasonable | overgeneralized | unsupported",
  "notes": "short evidence-based explanation"
}
```

Only candidates passing the frozen admission rule become `support_verified`.

### 8.4 Cross-document candidate pairing

Embedding is only a **candidate retrieval mechanism**.

Do not automatically merge on cosine threshold.

### 8.5 Semantic equivalence adjudication

At minimum distinguish:

```text
equivalent
a_subsumes_b
b_subsumes_a
related_but_distinct
contradictory
unrelated
```

Merge only when the adjudicator says the candidates are semantically compatible for a
single canonical strategy and their applicability conditions do not materially conflict.

### 8.6 Canonicalization

Create one canonical Experience and union all provenance evidence.

No hierarchy or graph.

### 8.7 Stable/Rare tiers

```text
Stable Core:
grounding_status == support_verified
AND distinct_source_count >= 2
```

```text
Supported Rare:
grounding_status == support_verified
AND distinct_source_count == 1
```

Both may remain in the library. Tier is diagnostic metadata, not a truth label.

---

## 9. Representation artifacts

Every compiler must emit a common wrapper:

```json
{
  "representation_id": "...",
  "type": "raw | summary | guideline | experience",
  "source_corpus_hash": "...",
  "compiler_model": "...",
  "compiler_prompt_version": "...",
  "compiler_input_tokens": 0,
  "compiler_output_tokens": 0,
  "compiler_calls": 0,
  "content": "...",
  "content_tokens": 0,
  "content_hash": "..."
}
```

For Experience, `content` is the serialized fixed-budget library and the full structured
objects are stored separately.

---

## 10. Budget controller

Implement one shared token-budget module.

It must:

- tokenize with a versioned tokenizer;
- enforce Target Evidence budget separately from Writing-conditioning budget;
- never let one condition silently exceed the Writing-conditioning limit;
- store pre-truncation and post-truncation token counts;
- use deterministic selection / ordering once frozen.

Default:

```yaml
target_evidence_tokens: 8000
writing_condition_tokens: 4000
```

Do not give Ours a longer hidden system prompt.

---

## 11. Writer harness

One Writer implementation for all conditions.

Input structure:

```text
[SYSTEM PROMPT — identical]
[TASK — identical]
[TARGET EVIDENCE PACK — identical for target]
[WRITING CONDITION — the only experimental variable]
```

Configuration must be shared:

```yaml
model: SAME
temperature: SAME
top_p: SAME
max_output_tokens: SAME
seed: SAME_IF_SUPPORTED
citation_format: SAME
```

Desired Introduction length should be derived once from the target dataset distribution
and frozen before the full run.

Generation artifact:

```json
{
  "generation_id": "...",
  "target_id": "...",
  "condition": "evidence_only | raw | summary | guideline | experience",
  "writer_model": "...",
  "writer_prompt_hash": "...",
  "target_evidence_hash": "...",
  "representation_hash": "...",
  "input_tokens": 0,
  "output_tokens": 0,
  "latency_ms": 0,
  "text": "..."
}
```

---

## 12. Leakage and reproducibility audits

Before generation, automatically fail a sample if any hard audit fails.

Minimum audits:

- gold Introduction path inaccessible to compiler/writer;
- target Introduction exact/near-copy absent from conditioning artifacts;
- source/target duplicate title / DOI / document-hash checks;
- evidence-pack equality across all conditions;
- writing-budget equality;
- prompt hash equality except representation content;
- source span exactness;
- generated representation copy-rate diagnostics;
- all model/prompt/tokenizer versions recorded.

Store audit results as machine-readable JSON.

---

## 13. Recommended repository layout

```text
baseline-v01/
├─ configs/
│  ├─ dataset.yaml
│  ├─ compiler.yaml
│  ├─ budget.yaml
│  ├─ writer.yaml
│  └─ evaluation.yaml
├─ data/
│  ├─ source/
│  │  ├─ raw/
│  │  └─ normalized/
│  ├─ target/
│  │  ├─ visible/
│  │  ├─ evidence/
│  │  └─ gold/
│  └─ pilot/
├─ src/
│  ├─ domain/
│  │  ├─ models.*
│  │  └─ schemas.*
│  ├─ ingest/
│  ├─ evidence_pack/
│  ├─ compilers/
│  │  ├─ raw.*
│  │  ├─ summary.*
│  │  ├─ guideline.*
│  │  └─ experience/
│  │     ├─ extract.*
│  │     ├─ span_validate.*
│  │     ├─ verify.*
│  │     ├─ pair_retrieval.*
│  │     ├─ adjudicate.*
│  │     └─ canonicalize.*
│  ├─ budget/
│  ├─ writer/
│  ├─ audits/
│  ├─ evaluation/
│  └─ cli/
├─ artifacts/
│  ├─ representations/
│  ├─ experiences/
│  ├─ generations/
│  ├─ audits/
│  ├─ costs/
│  └─ evaluations/
└─ tests/
```

Language / framework is intentionally not frozen by the research design. Prefer the
smallest stack that makes JSON artifacts, reproducibility, tests, and batch execution easy.

---

## 14. Implementation stages and gates

### Stage 0 — Scaffold and contracts

Implement:

- config loading;
- schemas;
- artifact IDs/hashes;
- logging;
- model adapter interface;
- tokenizer/budget interface.

**Gate 0**

- schema tests pass;
- artifacts round-trip serialize;
- configs are versioned;
- no business logic yet.

### Stage 1 — Data ingestion and isolation

Implement:

- ACL source normalization;
- NC_Physics target normalization;
- sentence/paragraph coordinates;
- `TARGET_VISIBLE / TARGET_EVIDENCE / TARGET_GOLD` separation;
- Target Evidence Pack builder.

**Gate 1**

For 20 source + 10 target pilot:

- all source evidence can be addressed deterministically;
- gold paths are not imported by compiler/writer modules;
- same Evidence Pack hash is reused for all conditions of a target.

### Stage 2 — Experience Compiler

Implement main frozen pipeline only.

**Gate 2**

On the 20-source pilot:

- extractor produces atomic candidates;
- ≥95% of retained evidence spans pass exact deterministic span check by construction;
- rejected spans are traceable;
- verifier output is structured;
- canonicalization never merges without semantic adjudication;
- full Experience Library serializes within the configured budget.

Do not implement 3-run consensus yet.

### Stage 3 — Baseline compilers

Implement:

- Raw;
- Summary;
- Compute-Matched Guideline;
- Experience.

**Gate 3**

- all representation artifacts use the same source corpus hash;
- Summary/Guideline/Experience compiler costs are logged;
- Guideline and Experience compute can be compared;
- all Writer-facing representations respect the same 4k budget.

### Stage 4 — Writer harness

Implement one shared Writer.

**Gate 4**

For one target:

- run all 5 conditions;
- Target Evidence Pack hash identical;
- system/task prompt hash identical;
- only representation hash differs;
- output and cost artifacts are complete.

### Stage 5 — 10-target pilot

Run:

```text
10 targets × 5 conditions = 50 generations
```

Pilot review should inspect:

- leakage;
- factual hallucinations;
- obvious source-domain content leakage;
- whether Guideline and Experience are meaningfully distinct;
- whether Experience schema contains actionable strategies rather than summaries;
- token and compute matching;
- catastrophic output-format or citation problems.

**Gate 5**

Freeze prompts/configs after pilot corrections.

Only then scale to full 100-target run.

---

## 15. Evaluation hooks required in code

The pilot does not need full final human evaluation infrastructure, but the code must
preserve everything required for it.

Primary outcomes to support later:

1. **Blinded expert rhetorical pairwise preference**
2. **Evidence faithfulness + citation validity**
3. **Target content coverage**

Store anonymized generation IDs so evaluators do not know the condition.

Diagnostics to preserve:

- Experience grounding precision;
- source count / tier;
- copy rate;
- token usage;
- compiler calls;
- latency;
- compression ratio;
- strategy count;
- extraction stability when ablation is run.

---

## 16. Mandatory ablations — implement only after main pilot works

### A1. Grounding

```text
full Experience
vs
Experience without verifier
```

### A2. Cross-document consolidation

```text
canonicalized library
vs
independent verified candidates
```

### A3. Extraction consensus

On a fixed source subset only:

```text
1-run
vs
3-run
```

3-run extraction is **not** part of the main system.

### A4. Evidence × Experience factorial

```text
T
T + E
T + X
T + E + X
```

Where:

- `T` = title/abstract/task-visible target information
- `E` = Target Evidence Pack
- `X` = same frozen Experience Library

This tests whether Experience adds writing knowledge rather than acting as a surrogate for
target content.

---

## 17. Success and kill criteria

Do not redefine success after seeing the results.

### Evidence supporting continuation

The strongest success pattern is:

- Ours beats Raw, Summary, and Generated Guideline on blinded expert rhetorical preference;
- especially Ours > Summary and Ours > Guideline;
- factual/citation faithfulness does not decline;
- target content coverage does not decline.

The frozen report suggests a confirmatory target of approximately:

```text
Ours pairwise win rate vs Summary and Guideline ≥ 55%
and paired-bootstrap 95% CI lower bound > 50%
```

with correction for multiple primary comparisons.

### Kill / simplification rules

```text
Experience ≈ Generated Guideline
→ core Experience hypothesis fails.
```

```text
Experience > Raw but Experience ≈ Summary
→ likely compression / noise-removal effect.
```

```text
No NLP → Physics gain
→ no evidence for held-out-field transfer.
```

```text
Target Evidence + Experience ≈ Target Evidence
→ Experience has no independent value when factual evidence is sufficient.
```

```text
3-run ≈ 1-run
→ delete consensus.
```

```text
full verifier ≈ no verifier
→ simplify/remove verifier.
```

```text
canonicalized ≈ unmerged verified candidates
→ simplify/remove semantic consolidation.
```

```text
generic/random academic advice ≈ source-derived Experience
→ source-derived strategy induction is not providing specific utility.
```

```text
LLM judge gain without blinded human preference gain
→ treat as NO-GO for the writing-quality claim.
```

---

## 18. First coding task

Start with **Stage 0 + Stage 1 only**.

Do not start the LLM compiler before these invariants are testable:

1. stable SourcePaper addressing;
2. stable TargetPaper separation;
3. deterministic Target Evidence Pack serialization;
4. hashing/version metadata;
5. budget interface;
6. artifact storage;
7. automated test proving `TARGET_GOLD` is unavailable to compiler/writer code paths.

### Definition of done for the first coding task

Given a local pilot fixture containing a few source and target documents, one CLI command
must produce:

```text
normalized source JSON
normalized target-visible JSON
normalized target-evidence JSON
separately stored target-gold JSON
Target Evidence Pack
audit report
```

and unit/integration tests must verify the separation and deterministic hashes.

Only after this gate passes should implementation proceed to the Experience Compiler.
