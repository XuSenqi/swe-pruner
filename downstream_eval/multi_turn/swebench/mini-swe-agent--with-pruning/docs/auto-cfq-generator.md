# 小模型自动生成 CFQ — 设计与改动说明

## 一、背景与动机

**问题**：SWE-Pruner 依赖 `context_focus_question`（CFQ）来裁剪大段 shell 输出。部分主模型（如 GLM-4.6）不会主动写 CFQ，导致 pruner 从不触发。

**GLM-4.6 vs GLM-5.2（astropy-7606 轨迹对比）**：

| 模型 | 步数 | 主动写 CFQ | Pruner 效果 |
|------|------|-----------|------------|
| GLM-4.6 | 13 | 0 次 | 未触发 |
| GLM-5.2 | 13 | 读代码步骤有 CFQ | 裁剪有效 |

**方案**：主模型执行命令后，若未提供 CFQ，则用**小模型**根据其 `reasoning_content` 自动生成 CFQ，再交给 pruner。

---

## 二、整体流程（当前实现）

```
主模型 → command (+ 可选 CFQ)
  → 执行 shell 命令，得到 text（原始 stdout 字符串）
  → 按 CFQ 来源选择门槛做 small_output / reread 检查
  → 无 CFQ 且 len(text) >= cfq_generator.min_output_chars 且 should_generate()？
      → 小模型从 reasoning_content (+ prior context) 生成 CFQ
  → 有 CFQ → 调用 pruner
      → fallback（low_score / low_keep_ratio / small_output）？→ 原文返回
      → 否则返回裁剪结果
  → 返回给主模型
```

### 2.1 双门槛设计（核心）

主模型自带 CFQ 时，它已明确表达裁剪意图，可对**较短**输出 prune；auto-CFQ 由小模型推断意图，更保守，只对**较长**输出触发。

| CFQ 来源 | 配置项 | 默认值 | 含义 |
|---------|--------|--------|------|
| 主模型自带 | `pruner.min_output_chars` | **1600** | `len(text) ≥ 1600` 才 prune |
| 小模型 auto-CFQ | `cfq_generator.min_output_chars` | **3200** | `len(text) ≥ 3200` 才生成 CFQ + prune |

**中间区间 1600–3200 字符**：只有主模型写了 CFQ 才会 prune；auto-CFQ 路径不触发。

```
len(text)
  ├─ 有大模型 CFQ
  │    ├─ < 1600 → small_output，跳过
  │    └─ ≥ 1600 → 直接 prune
  └─ 无大模型 CFQ
       ├─ < 3200 → 不生成 CFQ，不 prune，原文返回
       └─ ≥ 3200 → 尝试 auto-CFQ → prune
```

### 2.2 提前跳过（在生成 CFQ 之前）

| 条件 | 含义 | 使用的门槛 |
|------|------|-----------|
| `small_output` | `len(text) < min_output_chars` | 有 agent CFQ → `pruner.min_output_chars`；否则 → `cfq_generator.min_output_chars` |
| `reread` | 同一 `.py` 文件第 2 次及以上读取 | 与门槛无关，直接跳过 |

主模型自带 CFQ 时，若触发上述跳过，仍会在 `cfq_stats` 中记录 `prune_skipped`。

### 2.3 触发 auto-CFQ 的条件（全部满足）

1. 配置了 `cfq_generator` 且 pruner 已启用
2. 输出未触发 `small_output` / `reread` 提前跳过
3. 主模型**未写** CFQ（`refine_existing: false` 时不覆盖已有 CFQ）
4. 有 `reasoning_content`，或历史步骤有可用的 prior reasoning
5. `len(text) >= cfq_generator.min_output_chars`（当前默认 **3200**）
6. `should_generate()` 为真：
   - 命令是读代码类：`cat` / `grep` / `sed -n` / `head` / `tail` / `find` / `nl -ba`
   - 排除：`sed -i`、git、python/pytest、ls/pwd/wc、提交命令等

### 2.4 Pruner 调用后的 Fallback（P0 保护）

即使已生成 CFQ 并调用 pruner，以下情况仍**回退原文**：

| `fallback_reason` | 触发条件 | 典型场景 |
|-------------------|---------|---------|
| `low_score` | `score < threshold`（默认 0.5） | CFQ 与输出不匹配（如读错文件） |
| `low_keep_ratio` | `left_token_cnt / origin_token_cnt < min_keep_ratio`（默认 0.35） | 裁太狠，agent 会反复重读 |
| `small_output` | `origin_token_cnt < effective_min_output_chars // 4` | 原文 token 过小 |

`effective_min_output_chars` 按 CFQ 来源取值：agent CFQ 用 `pruner.min_output_chars`，auto-CFQ 用 `cfq_generator.min_output_chars`。

### 2.5 Prior context 补全

当前步 `reasoning_content` 过短（< `min_reasoning_chars`，默认 150）时，附加最近 N 步（默认 4 步）的历史 reasoning，避免 CFQ 过于泛化。

---

## 三、改动文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/minisweagent/utils/cfq_generator.py` | **新增** | CFQ 生成器核心逻辑，含 `min_output_chars` |
| `src/minisweagent/agents/default.py` | **修改** | 双门槛集成、P0 保护、轨迹字段 |
| `src/minisweagent/utils/pruner.py` | **修改** | `PrunerConfig`，agent CFQ 门槛 |
| `templates/swe-pruner.yaml` | **修改** | 真实测试配置 |
| `templates/pruner.yaml` | **修改** | 模板配置 |
| `src/minisweagent/config/extra/swebench.yaml` | **修改** | 默认 swebench 配置 |
| `src/minisweagent/run/extra/swebench.py` | **修改** | CLI 参数 |
| `scripts/test_cfq_integration.py` | **新增** | 轨迹回放 + 联调测试 |
| `tests/utils/test_cfq_generator.py` | **新增** | CFQ 生成器单元测试 |
| `tests/agents/test_pruner_guards.py` | **新增** | P0 保护 + 双门槛单元测试 |

---

## 四、核心实现要点

### 4.1 `cfq_generator.py`

- 调用 OpenAI 兼容 API：`http://10.10.10.181:8000/v1/chat/completions`
- 模型：`Qwen3.5-35B-A3B-FP8`，`enable_thinking: false`
- 输入：**reasoning_content** + command（不用 THOUGHT 文本）
- 输出：一条完整 CFQ，或 `SKIP`（返回 `None`）
- `_normalize_output()`：过滤含行号 / file hint 的低质量 CFQ

主要方法：

- `CFQGenerator.should_generate(command, output, *, min_output_chars)` — 门槛由 agent 传入（来自 `cfq_generator.min_output_chars`）
- `CFQGenerator.build_reasoning_block(current, prior)` — 必要时附加 prior context
- `CFQGenerator.generate(...)` — 调用小模型 API

### 4.2 `default.py` 集成点（`_apply_pruner`）

执行顺序：

1. 读取主模型 CFQ（若有），确定 `agent_min_output_chars` / `auto_cfq_min_output_chars`
2. **`_get_prune_skip_reason()`** — 按 CFQ 来源选门槛，检查 `small_output` / `reread`
3. **`cfq_generator.generate()`** — 仅无 agent CFQ 且 `len(text) ≥ auto_cfq_min_output_chars` 时
4. **`pruner_client.prune()`**
5. **`_prune_fallback_reason(..., min_output_chars=effective)`** — 决定是否回退原文

辅助方法：

- `_extract_reasoning_content(action)` — 从 `reasoning_content` 字段提取
- `_collect_prior_reasoning()` — 收集最近 N 步 assistant reasoning
- `_extract_read_paths(command)` / `_record_file_reads()` — 重读检测

### 4.3 轨迹字段

**`cfq_stats`（user message 上）**：

```json
{
  "source": "cfq_generator",
  "context_focus_question": "...",
  "used_prior_context": true,
  "prune_skipped": "small_output"
}
```

- `source`：`agent` 或 `cfq_generator`
- `prune_skipped`：仅提前跳过时出现（`small_output` / `reread`）

**`pruned_stats`（user message 上）**：

```json
{
  "score": 0.998,
  "origin_token_cnt": 2132,
  "left_token_cnt": 768,
  "model_input_token_cnt": 2231,
  "fallback": true,
  "fallback_reason": "low_keep_ratio"
}
```

- `fallback: true` 表示最终给 agent 的是**原文**，不是裁剪结果

---

## 五、配置说明（`swe-pruner.yaml`）

```yaml
agent:
  pruner:
    url: http://10.10.10.39:6001/prune
    timeout: 120
    retries: 3
    min_output_chars: 1600   # 主模型自带 CFQ 时的 prune 门槛
    min_keep_ratio: 0.35
    skip_prune_on_reread: true
    chunk_overlap_tokens: 50
    threshold: 0.5
  cfq_generator:
    url: http://10.10.10.181:8000/v1/chat/completions
    model: Qwen3.5-35B-A3B-FP8
    timeout: 60
    retries: 2
    max_tokens: 256
    temperature: 0.0
    refine_existing: false
    min_output_chars: 3200   # auto-CFQ 触发门槛（更保守）
    min_reasoning_chars: 150
    max_prior_reasoning_steps: 4
    chat_template_kwargs:
      enable_thinking: false

model:   # 主模型，与小模型分开配置
  model_name: claude-4-5-sonnet
  model_kwargs:
    api_base: ${OPENAI_BASE_URL}
    api_key: ${OPENAI_API_KEY}
    drop_params: true
    temperature: 0.0
```

### 5.1 参数说明

| 参数 | 位置 | 默认值 | 含义 |
|------|------|--------|------|
| **`min_output_chars`** | `pruner` | **1600** | 主模型自带 CFQ 时，`len(text) ≥ 此值` 才 prune |
| **`min_output_chars`** | `cfq_generator` | **3200** | auto-CFQ 路径，`len(text) ≥ 此值` 才生成 CFQ + prune |
| `min_keep_ratio` | `pruner` | 0.35 | 裁剪后保留比低于此值 → fallback 原文 |
| `threshold` | `pruner` | 0.5 | pruner score 低于此值 → fallback 原文 |
| `skip_prune_on_reread` | `pruner` | true | 同一文件重复读取时跳过 CFQ/prune |
| `min_reasoning_chars` | `cfq_generator` | 150 | 当前 reasoning 过短时附加历史 context |
| `max_prior_reasoning_steps` | `cfq_generator` | 4 | 最多收集几步历史 reasoning |

**注意**：`min_output_chars` 使用 Python `len(text)`，即命令 stdout 的**原始字符串长度**（含换行、空格、行号），不是 token 数。

估算：`1600 字符 ≈ 400 tokens`，`3200 字符 ≈ 800 tokens`（代码场景 `// 4` 粗算）。

**为何双门槛**：

- **1600（agent CFQ）**：主模型主动写 CFQ，说明它知道要裁什么，可对中等长度输出 prune
- **3200（auto-CFQ）**：小模型推断 CFQ 有误判风险，只对大段输出触发，避免对较短输出误裁引发重读

### 5.2 CLI 参数（`swebench.py`）

| 参数 | 作用 |
|------|------|
| `--cfq-generator-url` | 覆盖小模型 endpoint |
| `--disable-cfq-generator` | 关闭自动 CFQ |
| `--disable-pruner` | 关闭 pruner（同时关闭 cfq_generator） |
| `--pruner-url` | 覆盖 pruner endpoint |

### 5.3 单独重跑一个 instance

```bash
uv run mini-extra swebench \
  --subset verified --split test \
  --filter "astropy__astropy-12907" \
  --workers 1 \
  -c ./templates/swe-pruner.yaml \
  -m openai/zai-org/glm-4.6 \
  --pruner-url http://10.10.10.39:6001/prune \
  -o runs/retry-12907-run1
```

覆盖已有结果时加 `--redo-existing`；否则 `preds.json` 中已有该 instance 会跳过。

---

## 六、测试命令

工作目录：

```bash
cd downstream_eval/multi_turn/swebench/mini-swe-agent--with-pruning
```

### 6.1 轨迹回放 + 联调

```bash
python scripts/test_cfq_integration.py
```

脚本使用 `PRUNER_MIN_OUTPUT_CHARS=1600`、`CFQ_MIN_OUTPUT_CHARS=3200` 与当前配置一致。

### 6.2 单元测试

```bash
uv run python -m pytest tests/utils/test_cfq_generator.py tests/agents/test_pruner_guards.py -v
```

- `test_cfq_generator.py`：should_generate、prior reasoning、API 解析、SKIP、默认 `min_output_chars=3200`
- `test_pruner_guards.py`：small_output 提前跳过、双门槛（agent 2000 字符可 prune / auto-CFQ 2000 不触发）、reread、low_score fallback

### 6.3 完整 SWE-bench

```bash
export OPENAI_API_KEY=xxxxxx
export OPENAI_BASE_URL=https://api.modelverse.cn/v1

uv run mini-extra swebench \
  --subset verified --split test --slice 0:20 --workers 4 \
  -c ./templates/swe-pruner.yaml \
  -m openai/zai-org/glm-4.6 \
  --pruner-url http://10.10.10.39:6001/prune \
  -o runs/with-pruner-GLM-4.6-CFQ-dual-threshold
```

对照组（关闭 auto-CFQ）：

```bash
uv run mini-extra swebench \
  -c ./templates/swe-pruner.yaml \
  -m openai/zai-org/glm-4.6 \
  --disable-cfq-generator \
  -o runs/with-pruner-GLM-4.6-no-cfq
```

### 6.4 检查轨迹

```bash
python -c "
import json
traj = json.load(open('runs/.../astropy__astropy-12907.traj.json'))
for m in traj['messages']:
    if m.get('cfq_stats') or m.get('pruned_stats'):
        print('cfq:', m.get('cfq_stats'))
        print('pruned:', m.get('pruned_stats'))
        print('---')
"
```

成功裁剪：`source=cfq_generator` 且 `pruned_stats` 无 `fallback`，且 `left_token_cnt < origin_token_cnt`。

---

## 七、主模型环境变量

| 组件 | 配置位置 | 环境变量 |
|------|---------|---------|
| 主模型 | `model:` 段 | `${OPENAI_API_KEY}` / `${OPENAI_BASE_URL}`，或 `MSWEA_MODEL_API_KEY` / `MSWEA_MODEL_NAME` |
| CFQ 小模型 | `agent.cfq_generator` | 默认内网 URL，一般不需 key |
| Pruner | `agent.pruner` | 无 key |

**注意**：

- 模型名需带 provider 前缀，如 **`openai/zai-org/glm-4.6`**（不是 `zai-org/glm-4.6`）
- `SWEBENCH_MODEL` **不被识别**
- yaml 中 `api_key` / `api_base` 需用 `${...}` 占位符，或写死真实值

---

## 八、实验结论摘要（GLM-4.6，500 实例全量）

### 8.1 总体指标对比

`runs/with-pruner-GLM-4.6`（baseline，关闭 auto-CFQ）vs `runs/with-pruner-GLM-4.6-CFQ-v5-all`（CFQ dual，P0 + 1600/3200 双门槛）。

| 指标 | baseline（无 CFQ） | CFQ dual（1600/3200） | 差异 |
|------|-------------------|----------------------|------|
| Submitted | 459/500 (91.8%) | **466/500 (93.2%)** | +7 |
| 平均步数 | 52.7 | 53.0 | +0.3 |
| 平均 api_calls/任务 | 52.7 | 53.0 | 持平 |
| 总 LLM tokens | 717,625,717 | 672,837,218 | **-6.2%** |
| 平均 tokens/任务 | 1,435,251 | 1,345,674 | **-6.2%** |
| 总 prompt_tokens | 711,490,994 | 667,083,244 | -6.2% |
| cached_tokens | 679,493,888 | 629,834,752 | -49.7M |
| reasoning_tokens | 2,930,171 | 2,713,841 | -7.4% |
| prune 触发次数 | 14 | 921 | ~66× |
| prune 成功（无 fallback） | 14 | 182 | — |
| prune fallback | 0 | 739 | — |
| prune keep_ratio | 19.6% | 9.9% | 更激进 |
| auto-CFQ（小模型生成） | 0 | 918 | — |
| agent 自带 CFQ | 0 | 32 | — |
| auto-CFQ 使用 prior context | 0 | 496 (52%) | — |

### 8.2 关键结论

1. **auto-CFQ 解锁了 pruning**：baseline 中 GLM-4.6 从不写 CFQ，prune 只触发 14 次；加了小模型 auto-CFQ 后触发 921 次。
2. **端到端 token 下降温和（~6%）**：pruning 只压缩回填进上下文的 shell 输出文本，而 prompt 里 cached 前缀占绝对大头，所以整体只省 ~6%，并非「压缩到 10% → 省 90%」。
3. **成功率略升、步数持平**：Submitted 459 → 466（+1.4pp），平均步数 52.7 → 53.0，说明压缩没有引入重读灾难。
4. **P0 fallback 大面积兜底**：921 次 prune 中有 739 次（80%）触发了 fallback 保护（原文返回），真正裁剪成功的只有 182 次；keep_ratio 9.9% 说明即使有 P0 保护，实际保留比例仍偏激进。

### 8.3 典型 positive example（dual，astropy-13398）

成功裁剪（无 fallback），`cat -n` 读代码，auto-CFQ + prune：

- `cat -n .../builtin_frames/icrs_observed_transforms.py`：**1699 → 732 tokens**（score=0.999）
- `cat -n .../builtin_frames/altaz.py`：1747 → 1672（score=1.000）
- `cat -n .../builtin_frames/hadec.py`：1742 → 1460（score=1.000）

Auto-CFQ 示例：*What is the structure and key components of the AltAz frame implementation, including its attributes, methods, and how it handles transformations?*

### 8.4 典型 fallback example（dual，astropy-12907）

触发 P0 `low_keep_ratio` 保护，最终回退原文：

- `cat -n .../modeling/separable.py`：3961 → 525 tokens，keep 比 13.3% < 0.35 → **fallback 原文**
- `cat -n .../modeling/tests/test_separable.py`：2132 → 354 tokens，keep 比 16.6% < 0.35 → **fallback 原文**

两次 score 都高达 0.997/0.998，说明「score 高」不等于「裁得合适」，`min_keep_ratio` 保护是对「裁太狠」的独立兜底。

### 8.5 加入死循环守卫：v6 全量对比

`runs/with-pruner-GLM-4.6`（baseline）vs `runs/with-pruner-GLM-4.6-CFQ-v6-all`（CFQ dual + `max_repeat_steps: 10` 死循环守卫）。

#### 8.5.1 参数差异

| 参数 | baseline | v6 |
|------|----------|-----|
| `cfq_generator` | 无 | `Qwen3.5-35B-A3B-FP8`（`min_output_chars: 3200`） |
| `pruner.min_output_chars` | `min_chars: 500`（旧键） | `1600` |
| `pruner.min_keep_ratio` | 无 | `0.35` |
| `pruner.skip_prune_on_reread` | 无 | `true` |
| `pruner.threshold` | 0.5 | 0.5 |
| `max_repeat_steps` | 无 | `10` |
| `step_limit` / `cost_limit` | 250 / 3.0 | 250 / 3.0 |
| 主模型 | glm-4.6 | glm-4.6 |

> 注：v6 的 `min_keep_ratio` 为 0.35（本地改到 0.2 的版本未被 `.gitignore` 外的 `swe-pruner.yaml` 实际加载进 v6 运行）。baseline 使用旧代码的 `min_chars: 500` 键。

#### 8.5.2 总体指标

| 指标 | baseline | v6 | 差异 |
|------|----------|-----|------|
| 实例数 | 500 | 499 | -1（v6 缺 `sympy__sympy-18199`） |
| Submitted | 459 (91.8%) | 462 (92.6%) | +3 |
| 平均步数 | 52.7 | 43.7 | **-17%** |
| 总 LLM tokens | 717,625,717 | 433,350,985 | **-39.6%** |
| 平均 tokens/任务 | 1,435,251 | 868,439 | **-39.5%** |
| prompt_tokens | 711,490,994 | 428,524,263 | -39.8% |
| completion_tokens | 6,134,723 | 4,826,722 | -21.3% |
| cached_tokens | 679,493,888 | 403,258,496 | -40.6% |
| reasoning_tokens | 2,930,171 | 2,148,331 | -26.7% |
| prune 触发次数 | 14 | 914 | ~65× |
| prune 成功（无 fallback） | 14 | 185 | — |
| prune fallback | 0 | 729 | — |
| auto-CFQ（小模型生成） | 0 | 906 | — |

#### 8.5.3 成功 / 失败实例拆分

| 分组 | baseline | v6 |
|------|----------|-----|
| Submitted 实例数 | 459 | 462 |
| Submitted 总 tokens | 388,913,855 | 347,357,386 |
| Submitted 平均 tokens/任务 | 847,307 | 751,856（**-11%**） |
| Submitted 平均调用数 | 41.7 | 40.2 |
| 失败实例数 | 41 | 37 |
| 失败总 tokens | 328,711,862 | 85,993,599（**-74%**） |
| 失败平均 tokens/任务 | 8,017,362 | 2,324,151 |
| 失败平均调用数 | 176.2 | 87.7 |

#### 8.5.4 退出状态分布

| exit_status | baseline | v6 |
|-------------|----------|-----|
| Submitted | 459 | 462 |
| `LimitsExceeded` | 26 | 4 |
| `TimeoutExpired` | 11 | 4 |
| `RepeatedAction`（死循环守卫新增） | — | 29 |
| `Error` | 4 | 0 |

#### 8.5.5 关键结论

1. **-39.6% 的绝对主力是死循环守卫**：`max_repeat_steps: 10` 把「烧到 250 步 / 超时才停」的失控实例提前判为 `RepeatedAction` 快速失败，失败实例平均 token 从 8.0M 降到 2.3M，贡献了约 240M 的节省。
2. **pruning 对成功实例的真实净收益是 -11%**（847k → 752k/task），这是 CFQ + pruner 的温和但实在的贡献（约 40M）。
3. **成功率略升**：459 → 462（91.8% → 92.6%），说明死循环守卫主要杀掉本来就跑不出来的实例，误伤不大。

### 8.6 干净 baseline（无 pruner）vs v6 全量对比

`runs/baseline-GLM-4.6`（**完全不使用 pruner**）vs `runs/with-pruner-GLM-4.6-CFQ-v6-all`（pruner + auto-CFQ + 死循环守卫）。这是最干净的对照：把「无 pruner 的主模型」和「完整 CFQ + pruner + 守卫流水线」直接对比。

#### 8.6.1 配置差异

| 参数 | baseline-GLM-4.6 | v6-all |
|------|------------------|--------|
| pruner | 无 | `min_output_chars: 1600` / `min_keep_ratio: 0.35` / `threshold: 0.5` / `skip_prune_on_reread: true` |
| cfq_generator | 无 | `Qwen3.5-35B-A3B-FP8`（`min_output_chars: 3200`） |
| `max_repeat_steps` | 无 | `10` |
| `step_limit` / `cost_limit` | 250 / 3.0 | 250 / 3.0 |
| 主模型 | glm-4.6 | glm-4.6 |

#### 8.6.2 总体指标（499 个共有实例，v6 缺 `sympy__sympy-18199`）

| 指标 | baseline-GLM-4.6 | v6-all | 差异 |
|------|------------------|--------|------|
| 实例数 | 499 | 499 | — |
| 总步数 | 28,629 | 21,797 | -6,832 |
| 平均步数 | 57.4 | 43.7 | **-23.9%** |
| 总 API 调用 | 28,634 | 21,802 | -6,832 |
| prompt_tokens | 676,578,356 | 428,524,263 | -248,054,093 |
| completion_tokens | 6,430,616 | 4,826,722 | -1,603,894 |
| cached_tokens | 644,361,856 | 403,258,496 | -241,103,360 |
| reasoning_tokens | 2,508,223 | 2,148,331 | -359,892 |
| **总 LLM tokens** | **683,008,972** | **433,350,985** | **-36.6%** |
| 平均 tokens/任务 | 1,368,755 | 868,439 | **-36.6%** |

#### 8.6.3 退出状态分布

| exit_status | baseline-GLM-4.6 | v6-all |
|-------------|------------------|--------|
| Submitted | 468 | 462 |
| `LimitsExceeded` | 28 | 4 |
| `RepeatedAction`（死循环守卫新增） | 0 | 29 |
| `TimeoutExpired` | 2 | 4 |
| `Error` | 1 | 0 |

#### 8.6.4 成功 / 失败拆分

| 分组 | 指标 | baseline-GLM-4.6 | v6-all | 变化 |
|------|------|------------------|--------|------|
| **Submitted** | 实例数 | 468 | 462 | -6 |
| | 平均 tokens/任务 | 809,812 | 751,856 | **-7.2%** |
| | 平均 API 调用 | 46.2 | 40.2 | -6.0 |
| | 总 tokens | 378,991,833 | 347,357,386 | -8.3% |
| **失败** | 实例数 | 31 | 37 | +6 |
| | 平均 tokens/任务 | 9,807,004 | 2,324,151 | **-76.3%** |
| | 平均 API 调用 | 226.2 | 87.7 | -138.5 |
| | 总 tokens | 304,017,139 | 85,993,599 | **-71.7%** |

#### 8.6.5 v6 pruner / auto-CFQ 活动

| 指标 | 数值 |
|------|------|
| auto-CFQ（小模型生成） | 906 |
| agent 自带 CFQ | 29 |
| prune 成功（真正裁短） | 184 |
| prune fallback | 729 |
| 裁剪节省（shell 输出 tokens） | 134,541 |

#### 8.6.6 关键结论

1. **-36.6% 的绝对主力是死循环守卫**：baseline 有 28 个实例烧满 250 步（`LimitsExceeded`），v6 用 `max_repeat_steps: 10` 提前判为 `RepeatedAction`（29 个）快速失败，失败侧平均 token 从 9.8M 压到 2.3M（-76%），贡献约 218M 节省。
2. **pruner + auto-CFQ 对成功实例的真实净收益是 -7.2%**（809k → 752k/task，约 32M），温和但实在——省 token 的同时平均调用数还略降（46.2 → 40.2）。
3. **成功率基本持平**：468 → 462（93.8% → 92.6%），死循环守卫主要杀掉本来就跑不出来的实例，误伤有限。

### 8.7 局限

- 无 `reasoning_content` 的模型无法生成 CFQ
- CFQ 小模型、pruner 的 token **不在**主模型 `usage` 统计里
- `stats.py` 的 `extract_token_stats` 只统计主模型 tokens，不含 shell 输出字符数
- resolve 率需 SWE-bench 官方评测，不能只看 Submitted
- 死循环守卫（`max_repeat_steps`）只对「完全相同命令」判重，读不同行号的同类命令不会命中，仍可能漏掉部分 read-only 循环

---

## 九、架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        主模型 (大模型)                        │
│  输出: THOUGHT + bash command + 可选 CFQ                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ env.execute → text
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    default.py::_apply_pruner                 │
│                                                              │
│  1. 有 agent CFQ?                                            │
│     yes → len < pruner.min_output_chars (1600)? → 跳过       │
│     no  → len < cfq_generator.min_output_chars (3200)? → 跳过│
│     reread? → 跳过                                           │
│         │ 未跳过                                             │
│  2. 主模型写了 CFQ? ──no──► cfq_generator.generate()         │
│         │                                                    │
│  3. 无 CFQ? ──yes──► 原文返回                                │
│         │ no                                                 │
│  4. pruner.prune(code, query=CFQ)                            │
│         │                                                    │
│  5. fallback (low_score / low_keep_ratio / small_output)?    │
│     ──yes──► 原文返回 (pruned_stats.fallback)                │
│     ──no───► Filtered Output 给 agent                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              user message + cfq_stats + pruned_stats
```
