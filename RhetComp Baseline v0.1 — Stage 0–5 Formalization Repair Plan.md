# RhetComp Baseline v0.1 — Stage 0–5 Formalization Repair Plan

## 0. 修复目标

当前仓库已经完成 Stage 0–5 的 **mechanics implementation**：

```text
Synthetic fixtures
+ deterministic mock
→ Experience Compiler
→ Baselines
→ Writer
→ 50-generation pilot
```

本轮修复不重新设计研究方案，也不删除现有 mechanics mode。

目标是增加第二条真正的实验路径：

```text
真实 ACL / NC_Physics 数据
+
真实 Provider
+
DeepSeek V4 Flash
+
Qwen Embedding
        ↓
Experience / Baselines / Writer
        ↓
Real Pilot
        ↓
Formal Pilot Freeze
```

最终必须严格区分：

```text
Mechanics PASS
≠
Formal Experiment PASS
```

现有 `gate2.json ~ gate5.json` 继续保留为 mechanics evidence，但不能作为正式研究结果。

---

# 1. 全局冻结的模型与 Provider 决策

## 1.1 生成式模型

所有生成式角色统一使用：

```yaml
provider: deepseek
model: deepseek-v4-flash
protocol: openai_chat_completions
```

包括：

```text
Experience Extractor
Blind Verifier
Semantic Adjudicator
Summary Compiler
Guideline Compiler
Writer
```

禁止：

```text
某个 baseline 换模型
Verifier 偷换更强模型
自动 fallback 到其他模型
失败后切换 Provider
```

### Compiler Profile

用于：

- Experience Extraction
- Verification
- Semantic Adjudication
- Summary
- Guideline

冻结为：

```yaml
profile_id: deepseek_compiler_v1
model: deepseek-v4-flash
thinking:
  enabled: true
  reasoning_effort: high
stream: false
```

思考模式下不要向 Provider 发送：

```text
temperature
top_p
seed
```

如果内部统一 `ModelRequest` 仍保留这些字段，应改为 optional。

### Writer Profile

Writer 独立配置：

```yaml
profile_id: deepseek_writer_v1
model: deepseek-v4-flash
thinking:
  enabled: false
temperature: 0.0
top_p: null
seed: null
stream: false
```

这样五种 Writer condition 都使用完全相同的生成配置。

不要继续把 `seed=12345` 当成实验确定性保证；如果 Provider 没有明确支持某个参数，就不要发送，只记录为 `null`。

---

## 1.2 Embedding Model

冻结：

```yaml
provider: qwen
model: qwen3.7-text-embedding
dimensions: 1024
output_type: dense
```

它只用于：

```text
Experience candidate
→ vector
→ candidate-pair retrieval
```

绝不能用于：

```text
自动判断两个 Experience 相等
自动 merge
Verifier
最终评分
```

Embedding 只是高召回候选检索器。

固定 embedding 输入：

```text
strategy
+
applicable_when
```

不要加入：

```text
evidence span
source_id
paper title
ACL topic vocabulary
```

避免向量相似度主要被论文主题而不是 rhetorical strategy 支配。

最终 merge 仍必须：

```text
Qwen embedding
→ candidate pairs
→ DeepSeek semantic adjudication
→ canonicalization
```

---

# 2. Provider Foundation — 作为 Stage 0 修复的一部分

不要新增 LangChain、LangGraph 或 LiteLLM。

当前需求只需要两个薄 Adapter：

```text
ChatModelAdapter
EmbeddingAdapter
```

结构建议：

```text
src/adapters/
├─ chat/
│  ├─ base.py
│  └─ deepseek.py
├─ embedding/
│  ├─ base.py
│  └─ qwen.py
├─ records.py
└─ mock/
```

---

## 2.1 DeepSeek Adapter

实现真实：

```text
ModelRequest
→ DeepSeek ChatCompletions
→ ModelResponse
```

必须支持：

- system prompt；
- user prompt；
- max output tokens；
- thinking enabled / disabled；
- reasoning effort；
- JSON structured output；
- non-stream request；
- usage；
- latency；
- Provider 返回 model；
- Provider request id；
- system fingerprint（Provider 返回时记录；允许为 `null`）；
- finish reason。

禁止 Adapter：

```text
自动换模型
自动降级
自动缩短 prompt
自动切换 thinking mode
```

允许：

```text
transport error
429
5xx
```

做有限重试，例如最多 3 次。

但重试必须：

```text
完全相同 request
+
完整记录 retry_count
```

---

## 2.2 Qwen Embedding Adapter

新增独立接口，例如：

```python
EmbeddingRequest
EmbeddingResponse
EmbeddingAdapter
```

而不是硬塞进现有 `ModelAdapter.generate()`。

返回至少包含：

```text
vectors
model
dimensions
input_tokens
request_id
latency_ms
```

Embedding cache key：

```text
embedding_model
+
dimensions
+
embedding_text_hash
```

同一个 Experience 文本不得重复计费生成 embedding。

---

## 2.3 Provider Call Artifact

每一次正式调用保存：

```json
{
  "call_id": "...",
  "run_id": "...",
  "role": "extractor | verifier | adjudicator | summary | guideline | writer | embedding",

  "provider": "...",
  "requested_model": "...",
  "returned_model": "...",
  "provider_profile_hash": "...",

  "thinking_mode": "...",
  "reasoning_effort": "...",

  "prompt_hash": "...",
  "input_hash": "...",
  "response_hash": "...",

  "input_tokens": 0,
  "output_tokens": 0,
  "latency_ms": 0,

  "system_fingerprint": "... | null",
  "provider_request_id": "...",
  "retry_count": 0,

  "status": "success | failed"
}
```

不要保存或依赖模型的完整 chain-of-thought。

---

## 2.4 Provider 配置

新增：

```text
configs/providers.yaml
```

类似：

```yaml
config_version: baseline-v0.1-providers-1

deepseek_compiler:
  provider: deepseek
  model: deepseek-v4-flash
  thinking: enabled
  reasoning_effort: high

deepseek_writer:
  provider: deepseek
  model: deepseek-v4-flash
  thinking: disabled
  temperature: 0.0

qwen_embedding:
  provider: qwen
  model: qwen3.7-text-embedding
  dimensions: 1024
```

Secrets 仅来自环境变量：

```text
RHETCOMP_DEEPSEEK_API_KEY
RHETCOMP_DEEPSEEK_BASE_URL

RHETCOMP_QWEN_API_KEY
RHETCOMP_QWEN_BASE_URL
```

`.env` 不进入 Git。

增加：

```text
.env.example
```

但只能包含变量名。

---

## 2.5 Provider Smoke Test

新增：

```powershell
python -m src.cli.check_providers
```

它只做最小 live request：

```text
DeepSeek → 简单 JSON
Qwen → 一个 embedding
```

输出：

```text
DEEPSEEK_PROVIDER=PASS
QWEN_EMBEDDING=PASS
```

普通 unit tests 不允许调用收费 API。

Live tests 必须显式：

```text
--live
```

---

## Stage 0R Gate

Stage 0 formal repair 通过条件：

- DeepSeek Adapter 单测通过；
- Qwen Embedding Adapter 单测通过；
- mock 与 real adapter 共用上层接口；
- secrets 不进入 Git；
- provider profiles 有版本；
- thinking 参数不会和 temperature/top_p 混用；
- call artifact 可 round-trip；
- `check_providers --live` 可独立运行；
- 不引入 LangChain / LangGraph；
- mechanics mode 仍然全部通过。

---

# 3. Stage 1 Repair — 从 Synthetic Data 进入 Real Data

Stage 1 不删除 synthetic fixture。

变成两条数据线：

```text
Synthetic fixture
→ mechanics regression

Real dataset
→ formal experiment
```

---

## 3.1 Source Corpus

Formal Pilot：

```text
20 real ACL 2024 Introductions
```

Full Experiment：

```text
200 real ACL 2024 Introductions
```

新增：

```text
data/raw/acl/
data/manifests/acl_pilot.json
data/manifests/acl_full.json
```

GitHub 中尽量只保存：

```text
paper id
metadata
source URL / acquisition metadata
hash
processing status
```

不要默认重新分发完整论文文件。

Real-data CLI：

```powershell
# 仅在操作人审阅并明确接受适用许可/署名要求后执行
python -m src.cli.prepare_acl_corpus --profile pilot --accept-source-licenses
python -m src.cli.prepare_nc_physics_targets --profile pilot --accept-source-licenses
python -m src.cli.audit_gate1_formal
```

负责：

```text
raw source
→ Introduction detection
→ normalization
→ paragraph/sentence coordinates
→ SourcePaper
→ source corpus hash
```

---

## 3.2 Target Corpus

Formal Pilot 不使用 final 100-paper test set 调 prompt。

使用：

```text
NC_Physics train/validation pool
→ 固定选择 10 篇
→ real development pilot
```

为避免 Dataset Viewer 返回内容无法严格绑定到指定 revision，Stage 1R 不通过 Viewer
直接取这 100 条。数据获取固定为：

```text
NC_Physics pinned commit revision
→ 该 commit 下的 NC_Physics_trainval.jsonl 原文件
→ 校验完整原文件记录数与 SHA-256
→ 固定切片 train[10000:10100]
→ 校验 100 条快照 SHA-256
→ 固定 seed 选择 10 篇
```

原文件、切片快照、获取元数据及两者哈希必须进入 `nc_physics_pilot.json`，Gate 1R
需从固定原文件重新推导切片并逐条比对。任何 revision、URL、记录数、原文件哈希、
快照哈希或派生 target/evidence/gold/evidence-pack 哈希不一致均 FAIL。

ACL 与 NC_Physics 的 manifest 还必须记录适用许可标识/审阅 URL 及人工确认布尔值；
即使重新计算 manifest 哈希，确认值不是 `true` 时 Gate 1R 仍然 FAIL。公开可下载仅代表
可以访问，不自动替代许可与署名要求的人工确认。

最终正式 Full Experiment 才：

```text
NC_Physics official test 100
```

这样避免：

```text
先看 test 10 篇
→ 修改 prompt
→ 再把这 10 篇算进最终 100
```

造成 test contamination。

---

## 3.3 TARGET_GOLD

继续保持：

```text
TARGET_VISIBLE
TARGET_EVIDENCE
TARGET_GOLD
```

Compiler / Writer 仍然完全没有 Gold capability。

Formal Pipeline 再增加一个规则：

```text
任何 Writer 配置
不得由 official test Gold Introduction 推导
```

特别修复现在 Stage 5 中：

```text
desired_introduction_length
← synthetic gold distribution
```

Formal 模式下改成：

```text
NC_Physics dev/train aggregate length statistics
→ length_stats.json
→ Writer config
```

只允许使用统计量，例如：

```text
median words
median tokens
P25 / P75
```

Compiler / Writer 不读取那些 Introduction 本身。

---

## 3.4 Formal Token Budget

当前：

```text
DeterministicRegexTokenizer
```

继续用于 mechanics regression。

正式实验增加：

```text
DeepSeekFormalTokenizer
```

尽量使用与 DeepSeek V4 Flash 对齐、固定版本的 tokenizer。

记录：

```text
tokenizer_name
tokenizer_revision
tokenizer_hash
```

Formal 4k / 8k budget 都使用它。

同时每次真实 API 调用保存 Provider 实际：

```text
usage.prompt_tokens
usage.completion_tokens
```

用于验证本地 budget tokenizer 与实际调用的偏差。

---

## Stage 1R Gate

Real Pilot 数据准备通过条件：

```text
20 real ACL source
10 real NC_Physics dev targets
```

并满足：

- 所有 source span 可以确定性定位；
- real data manifest 固定；
- source corpus hash 固定；
- target IDs 固定；
- TARGET_GOLD 仍不可被 Compiler/Writer 访问；
- official test 100 没有被 prompt tuning pipeline 读取；
- Evidence Pack deterministic；
- 8k formal budget 正常；
- synthetic Gate 1 regression 仍通过。

---

# 4. Stage 2 Repair — Real Experience Compiler

保留当前：

```text
deterministic mode
```

作为 regression。

新增：

```text
formal mode
```

要求：

```text
20 real ACL
+
DeepSeek V4 Flash
+
Qwen embedding
```

真实执行：

```text
ACL Introduction
        ↓
DeepSeek Extraction
        ↓
Exact Span Validation
        ↓
DeepSeek Blind Verifier
        ↓
Verified Pool
        ↓
Qwen Embedding
        ↓
Candidate Pair Retrieval
        ↓
DeepSeek Semantic Adjudication
        ↓
Canonicalization
        ↓
Stable Core / Supported Rare
        ↓
4k Experience Library
```

---

## 4.1 Extraction

必须：

```text
single pass
open extraction
atomic candidates
no fixed taxonomy
```

现有 prompt 可以作为 v1 起点。

真实模型输出如果：

```text
invalid JSON
```

允许最多一次 format repair。

禁止：

```text
real LLM failed
→ silently fallback deterministic candidate
```

失败必须进入 trace。

---

## 4.2 Span Validation

保持确定性。

要求不是 ≥95% 即可继续保留，而是：

```text
只有 exact span verified candidate
才允许进入 verifier
```

无法精确定位：

```text
→ rejected
→ trace
```

---

## 4.3 Blind Verifier

真正调用 DeepSeek V4 Flash。

Verifier 只能看：

```text
span
observed_pattern
strategy
applicable_when
```

不能看到：

```text
extractor reasoning
extractor prompt history
embedding score
cluster information
```

---

## 4.4 Qwen Candidate Retrieval

删除 formal mode 中当前：

```text
deterministic_feature_hash
```

改为：

```text
qwen3.7-text-embedding
1024 dimensions
cosine similarity
```

仍保留 deterministic hash 作为 mechanics backend。

Formal candidate retrieval：

```text
top_k = 20
```

第一版不使用 aggressive cosine threshold 自动删除候选。

Embedding 只负责 candidate recall。

---

## 4.5 Semantic Adjudication

仍由 DeepSeek 决定：

```text
equivalent
a_subsumes_b
b_subsumes_a
related_but_distinct
contradictory
unrelated
```

只有：

```text
equivalent
a_subsumes_b
b_subsumes_a
```

且 applicable_when 不冲突时才允许 merge。

---

## Stage 2R Gate

Formal Experience Compiler 必须满足：

- `adapter_mode = model:deepseek-v4-flash`；
- real ACL source corpus；
- Qwen real embedding；
- deterministic fallback count = 0；
- retained spans exact = 100%；
- verifier 输出结构合法；
- 每个 rejected candidate 可追踪；
- embedding 调用完整记录；
- merge 都有 adjudication artifact；
- Stable/Rare tier 可复现；
- Experience Library ≤ 4k formal tokens；
- provider model/request/profile metadata 完整；`system_fingerprint` 为可选字段，Provider 返回时如实记录；
- 3-run consensus 仍然不实现。

当前 synthetic `Gate 2` 保留。

新增：

```text
Gate 2R
```

不要覆盖原 Gate 2。

---

# 5. Stage 3 Repair — Real Baselines + Compute Matching

四种 representation：

```text
Raw
Summary
Generated Guideline
Experience
```

全部基于完全相同：

```text
ACL source corpus hash
```

---

## 5.1 Raw

不调用 LLM。

只：

```text
real ACL exemplars
→ deterministic selection
→ 4k budget
```

---

## 5.2 Summary

使用：

```text
DeepSeek V4 Flash
deepseek_compiler_v1
```

目标：

```text
压缩 source corpus 的内容
```

不能主动转写成：

```text
actionable writing guidelines
```

否则 Summary 与 Guideline 混淆。

---

## 5.3 Generated Guideline

这是最重要 baseline。

同样：

```text
DeepSeek V4 Flash
same ACL source
same compiler profile
```

但直接要求：

```text
生成可执行 scientific-writing guidelines
```

不要：

- provenance；
- exact spans；
- verifier；
- Experience schema；
- semantic consolidation。

---

## 5.4 Compute Match

先运行 Experience Compiler 得到：

```json
{
  "llm_calls": ...,
  "input_tokens": ...,
  "output_tokens": ...
}
```

再给 Guideline compiler 一个：

```text
compute envelope
```

目标：

```text
Guideline input tokens
≈
Experience input tokens

Guideline calls
≈
Experience calls
```

Gate 建议冻结：

```text
call-count difference <= max(1, 10%)
input-token difference <= 15%
```

如果匹配失败：

```text
Gate 3R FAIL
```

不能只写：

```text
calls = 0 vs 0
→ PASS
```

---

## 5.5 Writing Condition Budget

四个 representation 均：

```text
≤ 4000 DeepSeek formal tokens
```

并保存：

```text
pre_budget_tokens
post_budget_tokens
compression_ratio
included_items
excluded_items
```

---

## Stage 3R Gate

要求：

- Raw/Summary/Guideline/Experience source corpus hash 完全相同；
- Summary / Guideline / Experience 均使用 `deepseek-v4-flash`；
- real provider call count > 0；
- Guideline 与 Experience compute envelope 达标；
- 四种 representation ≤ 4k；
- Experience 与 Guideline 内容不是 trivially identical；
- 所有 provider call artifacts 完整；
- synthetic Gate 3 regression 仍通过。

---

# 6. Stage 4 Repair — Real DeepSeek Writer

Stage 4 不再只验证 Writer class。

必须至少真正跑：

```text
1 real NC_Physics dev target
×
5 conditions
```

Writer 固定：

```text
deepseek-v4-flash
thinking = disabled
temperature = 0.0
same system prompt
same task prompt
same max output
```

五组唯一允许变化：

```text
writing-conditioning representation
```

---

## 6.1 Prompt 中明确 Evidence Boundary

公共 Writer prompt 必须强调：

```text
Target Evidence Pack
= 唯一事实来源
```

而：

```text
Raw
Summary
Guideline
Experience
```

仅用于：

```text
rhetorical / organizational guidance
```

禁止把 ACL/NLP source 的：

```text
事实
模型
benchmark
结论
citation
```

带进 Physics Introduction。

Citation 只能来自：

```text
Target Evidence Pack.reference_metadata
```

---

## 6.2 Model Identity

每次五条件 batch 检查：

```text
requested_model
returned_model
provider_profile_hash
provider_request_id
system_fingerprint（可选）
```

如果 Provider 返回了 `system_fingerprint`，且同一 target 五个条件中的非空 fingerprint 发生改变：

```text
该 target batch 标记 INVALID
```

重新完整跑五个条件。

不要只重跑某一个 condition。

---

## 6.3 Condition Order

每个 target 的五组调用顺序：

```text
本地固定随机种子 shuffle
```

保存 order manifest。

避免始终：

```text
Evidence → Raw → Summary → Guideline → Experience
```

导致时间顺序和 provider 状态产生系统偏差。

这里的 seed 是：

```text
本地实验排序 seed
```

不是模型 generation seed。

---

## Stage 4R Gate

同一 real target：

- 五组全部成功；
- 同一 DeepSeek model profile；
- target evidence hash 完全相同；
- base prompt hash 完全相同；
- prompt template hash 相同；
- representation hash 不同；
- Provider fingerprint 为可选；若存在多个非空值，则这些值必须完全一致；
- 无 deterministic Writer fallback；
- Citation 只来自 target evidence；
- 完整 cost/call artifacts。

---

# 7. Stage 5 Repair — Formal 10-target Pilot

当前：

```text
Synthetic 10 × 5
```

重新定义为：

```text
Stage 5M = Mechanics Pilot
```

不要删除。

新增：

```text
Stage 5R = Real Formal Pilot
```

使用：

```text
20 real ACL source
10 real NC_Physics development targets
5 conditions
DeepSeek V4 Flash
Qwen qwen3.7-text-embedding
```

得到：

```text
10 × 5 = 50 real generations
```

---

## 7.1 Real Pilot 检查目标

这一次仍然不是正式论文结果。

只检查：

```text
真实 API 是否稳定
Extractor JSON 是否稳定
Verifier 是否过度放行
Qwen embedding pair retrieval 是否合理
Experience merge 是否失控
Experience 与 Guideline 是否实际不同
4k budget 是否合理
Writer 是否把 ACL 内容泄漏到 Physics
Citation 是否正常
输出长度是否合理
Guideline / Experience compute 是否匹配
```

允许根据 Real Pilot 修：

```text
prompt bug
JSON schema bug
明显 pipeline bug
budget bug
citation bug
```

但不能：

```text
看哪组赢
→ 专门修改 Experience prompt
```

---

## 7.2 Pilot 与 Final Test 隔离

Formal Pilot：

```text
NC_Physics train/validation dev subset
```

Final Experiment：

```text
NC_Physics official test 100
```

两者不得重叠。

Stage 5R 完成以后，official test 100 才首次进入正式实验 pipeline。

---

## 7.3 Freeze

区分两个 Freeze：

```text
Mechanics Freeze
Formal Pilot Freeze
```

现有：

```text
gate5.json
```

只代表：

```text
mechanics_freeze
```

新增：

```text
gate5_formal.json
```

Formal Pilot Freeze 后冻结：

```text
provider profiles
model IDs
thinking modes
embedding model
embedding dimensions
tokenizer revision
source corpus manifest
prompts
JSON schemas
budgets
writer settings
compute-match rules
evaluation IDs
```

Freeze manifest 还必须记录所有正式输出文件的真实 SHA-256，而不只是目录名或逻辑 ID。
至少覆盖三份数据 manifest、四份上游 Gate audit、10 个 evidence pack、50 个 Writer
call artifact、50 个 generation artifact、10 个 condition-order manifest、Writer cost、
本次运行 manifest、evaluation 以及本次被接受的 compiler/representation artifacts；同时
冻结全部实际执行的 `src/**/*.py` 源码与正式配置文件。每个 Stage 的付费调用与
产物使用独立 attempt namespace；Freeze 只能引用上游 Gate 接受的 attempt，不得递归吸收
旧尝试残留。

Full Experiment 中不得再调整。

---

# 8. Artifact Namespace 修复

避免 mechanics 和 formal 相互覆盖。

建议：

```text
artifacts/
├─ mechanics/
│  └─ ...
│
├─ formal_pilot/
│  └─ <run_id>/
│     ├─ calls/
│     ├─ embeddings/
│     ├─ experiences/
│     ├─ representations/
│     ├─ generations/
│     ├─ costs/
│     ├─ audits/
│     └─ evaluations/
│
└─ full/
```

所有 artifact 都带：

```text
run_id
run_mode
config_hash
data_manifest_hash
provider_profile_hash
```

---

# 9. 修复执行顺序

不要一次性全部修改。

严格：

```text
Repair 1
Stage 0 Provider Foundation
        ↓
Gate 0R
STOP + REVIEW

Repair 2
Stage 1 Real-data Bootstrap
        ↓
Gate 1R
STOP + REVIEW

Repair 3
Stage 2 Real Experience Compiler
        ↓
Gate 2R
STOP + REVIEW

Repair 4
Stage 3 Real Baselines
        ↓
Gate 3R
STOP + REVIEW

Repair 5
Stage 4 Real Writer
        ↓
Gate 4R
STOP + REVIEW

Repair 6
Stage 5 Real 10-target Pilot
        ↓
Gate 5R
Formal Pilot Freeze
STOP
```

不要让 Codex 一次跑完所有 Repair。

每个 Gate 推送 GitHub 后进行独立 Review。

---

# 10. 本轮明确不做

修复 Stage 0–5 时禁止扩展：

- LangChain；
- LangGraph；
- multi-agent orchestration；
- RAG framework；
- vector database；
- pgvector；
- Neo4j；
- UI；
- workspace；
- 3-run consensus；
- Stage 6 human evaluation；
- Full 100-target experiment；
- final statistics。

Qwen embedding 暂时仅保存在本地 artifact/cache。

不需要向量数据库。

---

# 11. 最终 Stage 0–5 应达到的状态

修复前：

```text
Synthetic data
+
Mock models
→ mechanics PASS
```

修复后：

```text
                    ┌─ DeepSeek V4 Flash
Real ACL papers ────┤
                    └─ Qwen Embedding
                           ↓
                 Experience Library
                           ↓
      Raw / Summary / Guideline / Experience
                           ↓
                    DeepSeek Writer
                           ↓
              Real NC_Physics Dev Targets
                           ↓
                  50 Real Generations
                           ↓
                  Formal Pilot Freeze
```

到这里，我们才可以说：

> RhetComp 的实验系统已经经过真实论文与真实模型验证，可以进入 Stage 6 Evaluation Harness 和后续 Full Experiment。

---

# 12. 初始 Kickoff 任务（已完成的历史边界）

本计划最初批准时只执行：

```text
Stage 0 Provider Foundation Repair
```

该边界仅约束最初的 Stage 0R kickoff。后续 `/goal` 已明确授权按依赖顺序继续完成
Stage 1R–5R；当前执行状态与外部阻塞以
`planning-files/formalization-repair/task_plan.md` 和 `progress.md` 为准，仍不得越过任何 Gate
或提前进入 official-test / Stage 6。

Definition of Done：

1. 实现真实 DeepSeek `deepseek-v4-flash` Chat Adapter；
2. 实现 Qwen `qwen3.7-text-embedding` Embedding Adapter；
3. 增加 provider profiles；
4. thinking / temperature 参数语义正确；
5. 增加 ProviderCallArtifact；
6. 增加 mock provider tests；
7. 增加显式 `--live` provider smoke test；
8. secrets 仅使用环境变量；
9. 保证现有 51 个 mechanics tests 不回归；
10. Gate 0R 通过后停止，不开始 Real-data Bootstrap。
