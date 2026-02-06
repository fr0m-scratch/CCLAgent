# CCL Agent 实施手册 v2（任务级落地）

目标：把当前 CCL Agent 升级为 **Scientific Agent（Agentic Whitebox Tuning）**，满足 SC/AAAI 顶会 best-paper 级门槛，并具备 Nature 级 scientific methodology 的可迁移范式。

本文是“可执行实施规范”，不是概念阐述。每个任务都包含：
1. 目标与当前缺口
2. 精确代码改造点（文件 + 函数）
3. 数据结构/schema 变更
4. 统计与实验设计
5. 测试与验收命令
6. 风险与回滚方案

---

## 0. 总体执行原则

## 0.1 Whitebox Contract（硬约束）

每个 step 必须形成完整 `decision_bundle_v2`，且满足：
1. `why_selected[].refs` 非空。
2. `why_rejected[].refs` 非空（至少 top-2 反事实候选）。
3. 如果 `llm_decision_support.output` 为空，则 `used_in_decision=false`。
4. LLM advice 不允许跨 step 直接决策复用（只归档，不参与 action 选择）。
5. `decision.select_action` trace 事件必须含 `refs`。
6. `search.prune` 必须带标准化 drop reason taxonomy。
7. stop 事件必须包含统计证据（window、effect size、CI/p-value 或 bootstrap 置信区间）。

## 0.2 成果分层标准

1. SC 层：端到端性能、开销、稳定性、可扩展性。
2. AAAI 层：决策质量、解释忠实度、反事实有效性、置信度校准。
3. Nature 层：可迁移 scientific loop + 跨域验证 + 可审计可复现协议。

## 0.3 里程碑与 Gate

1. Gate A（Week 4）：证据链与 schema 硬化完成。
2. Gate B（Week 10）：科学决策核心能力（校准+归因+stop 证据）完成。
3. Gate C（Week 16）：TUI 审计能力 + replay 引擎 + CI 门禁完成。
4. Gate D（Week 24）：大规模实验与论文材料冻结完成。

---

## 1. 当前系统关键缺口（基于现状代码）

## 1.1 决策记录证据缺失

现状：`src/agent/analyzer.py::_write_decision_record` 将 refs 固定空数组。

影响：
1. 无法证明“为什么这个 action 被选中”。
2. TUI 只能显示文本 claim，无法跳证据。

## 1.2 LLM 输出破损会被静默吞掉

现状：`src/agent/online_advisor.py::_parse_llm_json` 失败时返回 `{}`；
`src/llm/schemas.py::validate_online_decision_support` 未强制必填键。

影响：
1. `output={}` 但可能仍被标记 `used_in_decision=true`。
2. 解释可信度失真。

## 1.3 Async advice step 对齐漂移

现状：`src/agent/analyzer.py` 会把晚到旧 step advice 当作当前 step fallback signal。

影响：
1. 决策-证据时间错位。
2. replay 无法严格复现。

## 1.4 context_pack 语义不够强

现状：`src/agent/context_pack.py` 只有基本观察与检索字段。

影响：
1. 缺少 candidate-level evidence。
2. 缺少 constraint snapshot 与 decision preconditions。

## 1.5 trace 缺少因果关系字段

现状：`src/trace/events.py` 无 `event_id/parent_event_id/span_id/causal_refs`。

影响：
1. 难做 graph replay。
2. 难做跨模块因果审计。

---

## 2. Target Architecture（AWT-v1.1）

## 2.1 六层体系

1. Evidence Registry
2. Trace Graph
3. Decision Contract
4. Modeling & Calibration
5. Agent Policy Orchestration
6. Observability & Replay

## 2.2 核心新对象

1. `context_pack_v2`：决策前上下文。
2. `decision_bundle_v2`：决策输出主文档。
3. `candidate_trace_v2`：候选全生命周期。
4. `llm_prompt_pack_v2`：LLM 调用可审计记录。
5. `stop_evidence_v2`：停止统计证据。
6. `attribution_report_v1`：收益归因。
7. `replay_report_v1`：重放一致性报告。

## 2.3 Evidence ID 规范

统一格式：`<namespace>:<scope>:<id>`

建议命名：
1. `metric:<step>:primary`
2. `metric:<step>:derived:<name>`
3. `rule:<uuid>`
4. `rag:<doc_id>:<chunk_id>`
5. `candidate:<step>:<candidate_id>`
6. `llm:call_<uuid>`
7. `tool:<step>:<call_id>`
8. `stop:<step>:claim_<idx>`
9. `model:<context_hash>:<version>`

---

## 3. Phase A（Week 1-4）证据链硬化（Foundation）

## A1. Trace Schema v2 + 因果图字段

### A1.1 目标

把 trace 从“事件流水”升级为“可回放因果图”。

### A1.2 修改文件

1. `src/trace/events.py`
2. `src/trace/emitter.py`
3. `src/trace/writer.py`
4. `doc/Design/trace_schema.md`
5. `tests/test_trace_thread_safety.py`（扩展）
6. 新增 `src/trace/validator.py`

### A1.3 精确实现

#### 1) `TraceEvent` 新字段

在 `src/trace/events.py` 的 dataclass 增加：
1. `event_id: str`
2. `parent_event_id: Optional[str]`
3. `span_id: Optional[str]`
4. `causal_refs: List[str]`
5. `quality_flags: List[str]`

`TraceEvent.now()` 中自动生成：
1. `event_id=uuid4()`
2. 默认 `quality_flags=[]`

#### 2) `TraceEmitter.event()` 签名扩展

在 `src/trace/emitter.py`：
1. 新增可选参数 `parent_event_id`, `span_id`, `causal_refs`, `quality_flags`。
2. 向后兼容：调用点不传时使用默认值。

#### 3) Trace 校验器

新增 `src/trace/validator.py`，提供：
1. `validate_event_schema(event)->list[str]`
2. `validate_event_refs(event)->list[str]`
3. `validate_trace_file(path)->Report`

规则：
1. `decision.select_action`、`search.prune`、`stop.decision`、`postrun.distill.rule` 必须 `refs` 非空。
2. `llm.call` 必须有 `refs` 且含 `llm:call_*`。

### A1.4 验收命令

```bash
python3 -m pytest tests/test_trace_thread_safety.py
python3 scripts/validate_artifacts.py --run-dir artifacts/<run_id> --check trace
```

### A1.5 DoD

1. trace schema 合规率 100%。
2. 关键事件 refs 非空率 >= 99%。

---

## A2. Online LLM 严格输出契约

### A2.1 目标

杜绝 “invalid JSON / empty output” 被当作有效建议。

### A2.2 修改文件

1. `src/llm/schemas.py`
2. `src/agent/online_advisor.py`
3. `src/agent/analyzer.py`
4. `tests/test_llm_schemas.py`
5. 新增 `tests/test_llm_output_strict_validation.py`
6. 新增 `tests/test_online_advice_alignment.py`

### A2.3 精确实现

#### 1) 强化 `validate_online_decision_support`

在 `src/llm/schemas.py`：
1. 要求必填顶层键：`hypotheses`, `numeric_guidance`, `tool_request`, `action_preference`, `convergence`。
2. `tool_request.name` 必须 in allowlist。
3. `convergence.decision` 必须 in `continue|stop`。
4. `convergence.confidence` 必须 `0<=x<=1`。
5. `hypotheses[].patch` 如存在必须 <=4 keys 且参数合法。

#### 2) JSON 完整性检测

在 `src/agent/online_advisor.py` 中新增：
1. `raw_is_valid_json: bool`
2. 解析失败时 `parse_errors += ["invalid_json"]`
3. 如果仅提取到 partial object（截断 JSON），新增 `parse_errors += ["partial_json"]`

#### 3) 决策使用门控

在 `src/agent/analyzer.py`：
1. `advice_used = advice is not None and advice.output not empty and parse_errors empty`。
2. 写 `step_k_llm_decision_support.json` 时新增字段：
   - `raw_is_valid_json`
   - `schema_passed`
   - `decision_eligible`

#### 4) 跨 step advice 禁用

在 `src/agent/analyzer.py` 的 late advice 处理逻辑：
1. `late.step != current_step` 时只 `persist_late_llm_advice`。
2. 不再赋值给 `advice`。

### A2.4 验收命令

```bash
python3 -m pytest tests/test_llm_schemas.py tests/test_llm_output_strict_validation.py tests/test_online_advice_alignment.py
```

### A2.5 DoD

1. `output={}` 且 `used_in_decision=true` 为 0。
2. advice step 与 call trace step 一致率 100%。

---

## A3. Decision Bundle v2（替换旧 decision_record）

### A3.1 目标

让每步决策具备可审计证据与反事实。

### A3.2 修改文件

1. `src/agent/analyzer.py`
2. 新增 `src/agent/decision_bundle.py`
3. `doc/Design/artifact_schema.md`
4. 新增 `tests/test_decision_bundle_refs.py`

### A3.3 数据结构（`decision_bundle_v2`）

路径：`steps/step_<k>_decision_bundle.json`

```json
{
  "schema_version": "2.0",
  "step": 5,
  "context_ref": "steps/step_5_context_pack.json",
  "chosen_action": {
    "kind": "numeric|hypothesis|rollback|stop",
    "rationale": "...",
    "call_chain": ["event:...", "llm:call_...", "candidate:5:5_2"]
  },
  "candidates_considered": [
    {
      "candidate_ref": "candidate:5:5_2",
      "score_breakdown": {
        "pred_time_ms": 1823.4,
        "uncertainty": 81.2,
        "risk_score": 0.12,
        "feasibility": 1.0,
        "final_rank_score": 0.77
      },
      "status": "selected|rejected",
      "reject_reason": "risk_too_high|dominated|...",
      "refs": ["metric:4:primary", "model:<...>"]
    }
  ],
  "why_selected": [{"claim": "...", "refs": ["..."] , "confidence": 0.83}],
  "why_rejected": [{"claim": "...", "refs": ["..."] , "confidence": 0.71}],
  "counterfactuals": [
    {
      "candidate_ref": "candidate:5:5_1",
      "expected_delta_ms": 41.2,
      "risk_delta": 0.08,
      "why_not": "dominated"
    }
  ],
  "constraints_snapshot": {
    "risk_max": 0.7,
    "sla_max_iteration_time": null,
    "budget_remaining_steps": 4
  },
  "rollback_plan": {"last_known_good_ref": "metric:2:primary"},
  "quality_flags": []
}
```

### A3.4 实现步骤

1. 在 `src/agent/decision_bundle.py` 实现 builder：
   - `build_decision_bundle(...) -> dict`
   - `validate_decision_bundle(...) -> list[str]`
2. 在 `src/agent/analyzer.py::_write_decision_record` 替换为调用 builder。
3. 保留旧文件 `step_k_decision_record.json` 一段时间（兼容 TUI），同时写新文件。
4. 新字段回填到 trace `decision.select_action` payload：
   - `decision_bundle_path`
   - `chosen_candidate_ref`

### A3.5 验收命令

```bash
python3 -m pytest tests/test_decision_bundle_refs.py
python3 scripts/validate_artifacts.py --run-dir artifacts/<run_id> --check decision-bundle
```

### A3.6 DoD

1. `why_selected/why_rejected refs` 非空率 >= 99%。
2. 每 step 至少 2 条 counterfactual（若候选数 >=3）。

---

## 4. Phase B（Week 5-10）科学决策核心（Scientific Core）

## B1. Numeric Search 白盒化 v2

### B1.1 目标

把“候选结果可见”升级为“候选生命周期可重放”。

### B1.2 修改文件

1. `src/agent/numeric.py`
2. `src/search/coordinate_descent.py`
3. 新增 `tests/test_pruning_reason_taxonomy.py`
4. `doc/Design/artifact_schema.md`

### B1.3 标准化 pruning taxonomy

枚举：
1. `duplicate`
2. `invalid_config`
3. `risk_too_high`
4. `dominance_pruned`
5. `uncertainty_guard`
6. `budget_guard`
7. `eval_timeout`
8. `tool_unavailable`

### B1.4 实现步骤

1. 在 `src/agent/numeric.py` 为每个 stage 添加统一结构：
   - `stage_name`
   - `input_snapshot`
   - `thresholds`
   - `status`
   - `reason`
   - `refs`
2. 输出 `step_k_candidates_trace.json` 升级为 v2。
3. `step_k_pruning_summary.json` 新增：
   - `dropped_by_stage`
   - `dropped_by_reason`
   - `threshold_snapshot`

### B1.5 验收命令

```bash
python3 -m pytest tests/test_pruning_reason_taxonomy.py
```

### B1.6 DoD

1. 所有 dropped candidate 具有标准 reason。
2. 可从 trace + candidates_trace 重建最终 selected。

---

## B2. Surrogate 校准与模型治理

### B2.1 目标

让 surrogate 的不确定性可量化、可比较、可用作风险控制。

### B2.2 修改文件

1. `src/models/surrogate.py`
2. `src/models/training.py`
3. 新增 `src/models/calibration.py`
4. 新增 `tests/test_surrogate_calibration.py`

### B2.3 具体算法

最小可落地方案：
1. 基模型：RF（现有）+ GBDT。
2. 不确定性：
   - RF 使用 tree variance。
   - GBDT 使用分位回归（p10/p50/p90）。
3. 校准：isotonic calibration 对 `|error|` 进行后验拟合。
4. 选择策略：按验证集 NLL/ECE 选主模型。

### B2.4 新产物

1. `memory/models/surrogate_<ctx>_<ts>.json` 增加：
   - `calibration_metrics`（ECE/NLL/PICP）
   - `model_rank`
2. `online/surrogate_predictions_step_k.json` 增加：
   - `calibrated_confidence`
   - `quantiles`

### B2.5 验收命令

```bash
python3 -m pytest tests/test_surrogate_calibration.py
python3 scripts/export_run_report.py artifacts/<run_id>
```

### B2.6 DoD

1. PICP 接近名义覆盖率（例如 90% 区间覆盖 85%-95%）。
2. 高置信误判率较旧版下降。

---

## B3. Causal Attribution（参数/动作贡献归因）

### B3.1 目标

把“最终提升”拆解为可解释贡献。

### B3.2 修改文件

1. 新增 `src/agent/attribution.py`
2. `src/agent/core.py`（postrun 挂钩）
3. `src/tui/workbench.py`（Attribution tab）
4. 新增 `tests/test_attribution_consistency.py`

### B3.3 归因方法

两级归因：
1. Step-level：
   - `delta_i = metric(step_{i-1}) - metric(step_i)`。
   - 按 action lane 汇总（hypothesis/numeric/rollback）。
2. Param-level：
   - 近似 leave-one-out（LOO）在 surrogate 上估计单参数贡献。
   - 记录符号与置信区间。

### B3.4 新产物

1. `postrun/attribution_report.json`
2. `postrun/attribution_summary.md`

### B3.5 DoD

1. 报告可解释“哪 3 个参数贡献最大”。
2. 与 counterfactual 结论无明显冲突（自动一致性检查）。

---

## B4. Stop/Convergence 统计证据化

### B4.1 目标

停止决策必须基于统计证据，而非纯启发式阈值。

### B4.2 修改文件

1. `src/agent/stop_policy.py`
2. `src/agent/analyzer.py`
3. 新增 `tests/test_stop_policy_statistics.py`

### B4.3 算法

1. 基础判定：plateau + patience（现有）。
2. 统计增强：
   - window 内最佳与最近均值比较。
   - bootstrap CI 检查 improvement 是否包含 0。
3. 输出 stop claim：
   - `effect_size`
   - `ci_low/ci_high`
   - `evidence_refs`

### B4.4 新产物

`steps/step_k_stop_decision.json` 升级字段：
1. `statistics.effect_size`
2. `statistics.bootstrap_ci`
3. `statistics.window_values`
4. `claims[].refs`

### B4.5 DoD

1. stop 决策证据完整率 100%。
2. 误停率在验证集下降。

---

## 5. Phase C（Week 11-16）可观测与重放（Observability + Replay）

## C1. TUI 审计级升级

### C1.1 目标

让评审可在 UI 中逐步追踪“决策 -> 证据 -> 反事实 -> 结果”。

### C1.2 修改文件

1. `src/tui/workbench.py`
2. `src/tui/live_monitor.py`（少量适配）
3. 新增 `tests/test_tui_evidence_render.py`

### C1.3 新增 Tab

1. `Evidence Chain`
2. `Counterfactual`
3. `Attribution`
4. `Replay Diff`

### C1.4 具体实现

1. `Evidence Chain`：显示 decision_bundle 的 claims + refs + 可点击 artifact path。
2. `Counterfactual`：显示 top-2 未选候选与差值。
3. `Attribution`：展示 postrun 归因图表（文本 + 表格）。
4. `Replay Diff`：加载 replay_report 并高亮 mismatch。

### C1.5 DoD

1. 任一步的“why selected”可在 30 秒内追到原始 evidence。
2. 不存在无 refs 的核心决策说明。

---

## C2. Replay Engine v2

### C2.1 目标

将 `scripts/replay_trace.py` 从日志打印器升级为一致性验证器。

### C2.2 修改文件

1. `scripts/replay_trace.py`
2. 新增 `src/trace/replay.py`
3. 新增 `tests/test_replay_determinism.py`

### C2.3 replay 逻辑

输入：`run_dir`

流程：
1. 加载 `trace/events.jsonl`。
2. 按 step 读取 `decision_bundle/candidates_trace/context_pack`。
3. 重建每一步候选排序与选择。
4. 对比原始 action、selected candidate、stop reason。

输出：`replay_report.json`

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "summary": {
    "step_count": 8,
    "action_match_rate": 1.0,
    "candidate_match_rate": 0.99,
    "stop_reason_match": true
  },
  "mismatches": []
}
```

### C2.4 DoD

1. showcase run replay 一致率 >= 99%。
2. mismatch 可定位具体 step + event_id。

---

## C3. CI 门禁与质量治理

### C3.1 目标

把“研究系统”变成“可持续交付系统”。

### C3.2 修改文件

1. 新增 `scripts/validate_artifacts.py`
2. 新增 `scripts/ci_quality_gate.sh`
3. 更新 `scripts/run_tests.sh`

### C3.3 gate 规则

1. `schema_gate`：artifact/trace schema 合规。
2. `explain_gate`：decision refs/coverage 达标。
3. `replay_gate`：一致率达标。
4. `perf_regression_gate`：关键指标不退化。

### C3.4 DoD

1. PR 未通过任一 gate 不允许合并。
2. quality gate 报告自动产出。

---

## 6. Phase D（Week 17-24）SOTA 实验与论文产线

## D1. SC 实验协议（系统性能）

### D1.1 实验矩阵

1. workload：LLM/CV/communication-heavy mixed。
2. topology：单机、多机 IB、异构。
3. scale：8/16/32/64 GPU。
4. repetitions：每 setting 至少 5-10 次（按预算）。

### D1.2 对比基线

1. Native NCCL tuner。
2. AutoCCL 风格 baseline。
3. 消融：
   - no-LLM
   - no-memory
   - no-pruning
   - no-calibration
   - no-counterfactual gating

### D1.3 统计方法

1. 主指标：`iteration_time_ms`。
2. 统计检验：Mann-Whitney U + bootstrap CI。
3. 报告：均值、p50、p95、CI、effect size。

### D1.4 SC 门槛

1. 对最强基线显著优于（p<0.05 + 有效 effect size）。
2. tuning overhead 在阈值内（例如 <=8%，场景可变）。
3. 稳定性不低于基线（p95 与失败率可控）。

---

## D2. AAAI 实验协议（Agent Intelligence）

### D2.1 指标定义

1. Evidence Coverage：
   - `non_empty_refs_decision / all_decisions`
2. Faithfulness：
   - 删除关键证据后 action 变化率与模型说明一致。
3. Counterfactual Validity：
   - top-2 未选候选预测与真实差值排序一致率。
4. Tool Utility：
   - tool request accepted 后收益边际分布。
5. Calibration：
   - ECE/Brier/NLL。

### D2.2 评估产物

1. `artifacts_eval/aaai_metrics.json`
2. `artifacts_eval/faithfulness_casebook.md`

### D2.3 AAAI 门槛

1. 解释覆盖率 >= 99%。
2. faithfulness 显著优于 prompt-only 解释。
3. calibration 与不确定性质量优于未校准版本。

---

## D3. Nature 范式扩展

### D3.1 目标

证明方法不是“只对 NCCL tuning 有效”。

### D3.2 跨域试点（至少 1 个）

建议：
1. 存储系统参数调优
2. 网络栈参数调优

要求：复用同一 Whitebox Contract：
1. evidence registry
2. decision bundle
3. counterfactual
4. replay

### D3.3 交付

1. `doc/Design/scientific_agent_methodology.md`
2. 跨域 case study artifacts
3. 审计与伦理章节（风险、可追责、失败处理）

---

## 7. 任务排程（WBS）

## Week 1-2（Sprint S1）

1. A2 全量完成（LLM strict contract + advice alignment）。
2. A3 最小版（decision_bundle v2 + refs 非空）。

## Week 3-4（Sprint S2）

1. A1 完成（trace v2 + validator + schema 文档）。
2. Gate A 评审。

## Week 5-6（Sprint S3）

1. B1 完成（candidate/pruning v2）。
2. B4 基础统计 stop。

## Week 7-8（Sprint S4）

1. B2 完成（calibration pipeline）。
2. B3 MVP（step-level attribution）。

## Week 9-10（Sprint S5）

1. B3 参数级归因完善。
2. Gate B 评审。

## Week 11-12（Sprint S6）

1. C2 replay v2。
2. C3 CI gates。

## Week 13-16（Sprint S7-S8）

1. C1 TUI 审计视图全部上线。
2. Gate C 评审。

## Week 17-20（Sprint S9-S10）

1. D1/D2 全量实验。
2. ablation + 统计报告。

## Week 21-24（Sprint S11-S12）

1. D3 跨域 case study。
2. Gate D 与论文冻结。

---

## 8. 逐任务测试矩阵（必须新增）

## 8.1 Unit tests

1. `tests/test_llm_output_strict_validation.py`
2. `tests/test_online_advice_alignment.py`
3. `tests/test_decision_bundle_refs.py`
4. `tests/test_pruning_reason_taxonomy.py`
5. `tests/test_surrogate_calibration.py`
6. `tests/test_stop_policy_statistics.py`
7. `tests/test_attribution_consistency.py`

## 8.2 Integration tests

1. `tests/test_replay_determinism.py`
2. `tests/test_trace_validator_integration.py`
3. `tests/test_tui_evidence_render.py`

## 8.3 Golden-run regression

新增固定种子 run（dry-run profile）：
1. 输出固定 artifact hash 白名单（忽略 timestamp/event_id）。
2. 检查 action 序列、stop reason、核心 KPI 不退化。

---

## 9. 每个任务的验收命令（统一模板）

## 9.1 开发阶段

```bash
python3 -m pytest tests/test_llm_schemas.py \
  tests/test_llm_output_strict_validation.py \
  tests/test_online_advice_alignment.py \
  tests/test_decision_bundle_refs.py
```

## 9.2 运行与审计

```bash
python3 -m src.runner \
  --mode headless \
  --config configs/agentic_showcase_kimi.json \
  --workload workload/benchmarks/llama3.1-8b-agentic-showcase.json \
  --provider fireworks \
  --model accounts/fireworks/models/kimi-k2p5 \
  --dry-run --simulate-workload

python3 scripts/validate_artifacts.py --run-dir artifacts/<run_id> --check all
python3 scripts/replay_trace.py artifacts/<run_id> --out artifacts/<run_id>/replay_report.json
```

## 9.3 门禁

```bash
bash scripts/ci_quality_gate.sh --run-dir artifacts/<run_id>
```

---

## 10. SOTA 声称的证据要求（必须满足）

任何“state-of-the-art”结论必须同时满足：
1. 与最强基线在相同硬件/预算/协议下对比。
2. 至少 3 组不同场景（workload/topology/scale）显著获胜。
3. 报告包含统计显著性 + effect size + 置信区间。
4. 解释质量指标（coverage/faithfulness/calibration）显著优于对照。
5. 全部 artifacts、replay、validator 可公开复现。

禁止：
1. 只报告单场景最优值。
2. 混用不同预算或不同 early-stop 规则做“伪公平”比较。
3. 只给文字解释不提供 refs。

---

## 11. 风险与回滚计划（任务级）

## 11.1 LLM 不稳定

策略：
1. 严格建议制（advice != action）。
2. advice 失效时 deterministic fallback。
3. parse/schema 失败写 quality flag 并上报。

## 11.2 复杂度膨胀

策略：
1. schema-first。
2. 一切新增字段都要 validator + test。
3. 每 phase 后做技术债清理（2-3 天）。

## 11.3 性能收益与解释质量冲突

策略：
1. 双目标优化：performance + explainability。
2. 通过 ablation 找 Pareto front。

---

## 12. 立即执行包（下一个 PR）

建议按以下顺序提交 3 个 PR：

PR-1（A2）
1. strict online schema
2. parse_errors 强化
3. advice step 对齐修复

PR-2（A3）
1. decision_bundle_v2 builder
2. analyzer 接入
3. refs 非空 gate

PR-3（A1）
1. trace event v2 字段
2. trace validator
3. schema 文档与 CI

每个 PR 都必须包含：
1. 单测
2. 一个 dry-run artifact 示例
3. 验收脚本输出摘要

---

## 13. 交付清单（最终）

1. 实施手册：`doc/Design/impl_plan_sc_aaai_nature.md`
2. 代码与测试（A/B/C/D 全阶段）
3. 评测脚本与统计报告
4. 论文图表可复现 pipeline
5. 跨域 scientific agent case study

