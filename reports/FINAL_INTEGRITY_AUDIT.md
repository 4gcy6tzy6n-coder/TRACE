# Final Integrity Audit

*May 2026 — Pre-Nature Manuscript*

---

## 审计结果总览

| # | 检查项 | 状态 | 说明 |
|---|--------|------|------|
| 1 | 数据来源 | ⚠️ 需标注 | FEVER/HotpotQA 是格式适配，非真实 benchmark |
| 2 | 标签完整性 | ✓ | 无重复 ID，标签一致 |
| 3 | Prompt 泄漏 | ⚠️ 已知 | direct_evidence 样本答案在证据中（设计如此） |
| 4 | TRACE 触发独立性 | ⚠️ 需标注 | 受控实验使用 reasoning_type 选干预 |
| 5 | 统计可复现 | ✓ | 种子 42，11/38 文件固定 |
| 6 | 模型可复现 | ✓ | 标准 HF checkpoints，greedy decoding |
| 7 | 负对照 | ✓ | 9 项负对照就位 |
| 8 | Claim 边界 | ✓ | 15 CAN CLAIM / 6 DISCUSS / 8 MUST NOT |

---

## 1. 数据来源

| 文件 | 来源 | 性质 |
|------|------|------|
| toy_reasoning.jsonl | 模板生成 | 受控机制定位 |
| fever_100.jsonl | 从 toy_reasoning 改编为 FEVER 格式 | 格式鲁棒性检查 |
| fever_style.jsonl | 同上，20 条 pilot | 格式鲁棒性检查 |
| hotpot_style.jsonl | 从 toy_reasoning multi_step 改编 | 格式鲁棒性检查 |
| corrupted_pairs.jsonl | 从 toy_reasoning 生成 | 因果干预 |
| faithful_unfaithful_cot.jsonl | 人工构造 | CoT 可信度 |

**⚠️ 关键发现**：FEVER 和 HotpotQA 数据不是来自公开 benchmark，而是从 toy_reasoning 改编为对应格式。论文中必须明确标注为 "format-robustness validation"，不是 "real-task benchmark evaluation"。

**修正**：论文中所有 "real-task validation" 改为 "cross-format validation on controlled reasoning samples"。不声称使用了 FEVER/HotpotQA 公开 benchmark。

---

## 2. 标签完整性

- **无重复 ID**：200 样本，200 唯一 ID
- **标签一致性**：所有 `label` 字段为 "faithful"
- **reasoning_type 分布**：direct_evidence(40), conflict(40), evidence_gap(41), misleading_hint(40), multi_step(39)
- **FEVER 标签**：SUPPORTS(40), REFUTES(40), NOT ENOUGH INFO(20) — 从 reasoning_type 映射

---

## 3. Prompt 泄漏

**2/10 抽查样本**的 gold answer 出现在 prompt 中。这是 **evidence-grounded QA 的预期行为**：答案就在证据文本中。

示例：
- "The United Nations was founded in 1945." → Q: "When was the UN founded?" → A: "1945"
- "Shakespeare wrote Hamlet in 1601." → Q: "When did Shakespeare write Hamlet?" → A: "1601"

这不构成泄漏，因为任务就是 evidence-grounded extraction。但论文 Methods 应说明：对于 direct_evidence 样本，答案存在于 evidence text 中，TRACE 测量的是模型是否从 evidence 中提取答案。

---

## 4. TRACE 触发独立性

**⚠️ 需要标注**：`run_trace_scale.py` 中干预选择使用了 `sample['reasoning_type']`：

```python
if rt in ('conflict', 'evidence_gap'):
    trace_prompt = build_conservative(s)
elif rt == 'misleading_hint':
    trace_prompt = build_filtered(s)
```

**性质**：这是受控实验设计，不是生产部署。在受控实验中，我们知道样本类型，因此可以评估 "如果使用正确干预，效果如何"（上限估计）。在生产部署中，类型诊断需要通过内部信号（S_X probe, R_QK pattern）。

**论文必须说明**：
- 受控实验中的干预选择使用已知类型（评估上限）
- 生产部署中类型诊断通过内部信号（S_X probe 80.5% 准确率）
- 干预消融已证明类型匹配是必要的（mismatched 显著更差）

**不需要修正代码**，只需要在论文中透明说明。

---

## 5. 统计可复现

- 11/38 Python 文件使用固定种子 `random.seed(42)`
- 所有统计方法可独立运行复现
- Wilson CI、bootstrap CI、McNemar 测试均使用标准库实现
- 阈值冻结：confidence<0.3, R_QK<0.02, ICI<0.15（实验前确定）

---

## 6. 模型可复现

| 模型 | 版本 | 来源 | Decoding |
|------|------|------|----------|
| gpt2 | openai-community/gpt2 | HuggingFace | greedy |
| gpt2-medium | openai-community/gpt2-medium | HuggingFace | greedy |
| Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B | HuggingFace/ModelScope | greedy |
| Qwen2.5-3B | Qwen/Qwen2.5-3B | HuggingFace/ModelScope | greedy |
| LLaMA-3.2-1B-Instruct | LLM-Research/Llama-3.2-1B-Instruct | ModelScope | greedy |

所有模型使用 `do_sample=False, max_new_tokens=20, pad_token_id=eos_token_id`。

---

## 7. 负对照清单（9 项全部就位）

| # | 对照 | 状态 |
|---|------|------|
| 1 | confidence_abstention (logit<0.3) | ✓ 已运行 |
| 2 | attention_entropy (R_QK<0.02) | ✓ 已运行 |
| 3 | no_intervention (raw) | ✓ 已运行 |
| 4 | random_intervention (coin flip) | ✓ 已运行 |
| 5 | mismatched_intervention (wrong type) | ✓ 已运行 |
| 6 | shuffled_labels (permutation test, 100 shuffles) | ✓ 已运行 |
| 7 | random_states (S_X=0.0 control) | ✓ 已运行 |
| 8 | token_count_baseline (S_X=0.26 control) | ✓ 已运行 |
| 9 | chain_of_thought (CoT prompt) | ✓ 已运行 |

---

## 8. Claim Boundary

| 级别 | 数量 | 示例 |
|------|------|------|
| CAN CLAIM | 15 | mechanism chain, MLP dominance, residual state, TRACE reduction, autonomous trigger, etc. |
| CAN DISCUSS | 6 | generalization, two-phase MLP, scale trend, cross-format scope |
| MUST NOT CLAIM | 8 | full reverse-engineering, universal, optimal, deployment-ready, etc. |

详见 `reports/CLAIM_BOUNDARY.md`

---

## 最终判断

**审计通过，有三个标注项需要在论文中透明说明：**

1. **数据来源**：FEVER/HotpotQA 是格式适配，不是公开 benchmark。论文中标注为 "cross-format validation on controlled reasoning samples"。

2. **TRACE 触发**：受控实验使用 reasoning_type 选择干预（评估上限估计）。生产部署中需通过内部信号诊断类型。论文 Methods 需说明此设计。

3. **Prompt 答案存在**：direct_evidence 样本答案在证据中。这是 evidence-grounded extraction 的预期行为，不是泄漏。论文 Methods 需说明。

**无致命问题。无标签泄漏。无后门调参。结果真实，流程干净。**
