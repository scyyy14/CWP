# 第八步“移动次数优先级前移”执行方案

## 1. 目标

把完整合法排程的正式词典序目标从：

```text
(makespan, split_bay_count, load_deviation, movement_count)
```

改为：

```text
(makespan, split_bay_count, movement_count, load_deviation)
```

含义是：

1. 完工时间仍是绝对第一目标；任何 `H=208` 的合法方案都优于任何 `H=209` 的方案。
2. 拆分贝位数仍是第二目标；不能为了少移动而增加拆分贝位。
3. 当工期和拆分贝位数相同时，优先选择移动次数更少的方案。
4. 只有前三项都相同时，才比较中间重载偏差。

本任务不是把移动次数加入工期，也不是把四目标改为加权和。必须继续使用严格词典序比较。

## 2. 基线与分支

- 当前工作分支：`codex/step8-improvements`
- 当前 HEAD：`3b914a7`；该提交只更新了 `output/schedule.json` 和 `output/schedule.png`。
- 当前第八步累计局部修复代码提交：`9785421 Add cumulative local Step 8 repair`。
- 新建分支建议：`codex/step8-move-priority`，从当前 HEAD 创建，以便原改进版继续作为对照。
- 工作区已有很多用户实验目录和未跟踪文件，不得删除、覆盖或批量加入提交。

修改前先记录：

```bash
git status --short --branch
git rev-parse HEAD
```

## 3. 必须修改的代码

### 3.1 正式目标键

文件：`cwp_solver.py`

修改 `_CandidateSchedule.objective_key`：

```python
return (
    self.makespan,
    self.split_bay_count,
    self.movement_count,
    self.load_deviation,
)
```

同步修改 docstring。完整候选的接受、全局最好方案、精英池排序和发布都应继续统一依赖 `objective_key`，不要分别实现另一套排序。

### 3.2 输出、记录和文档顺序

检查并同步以下位置：

- `cwp_solver.py` 的控制台“目标优先级”文字；
- `README.md` 中两处四目标说明；
- `tools/evaluate_critical_step.py` 中手工构造的 objective 数组、报告标题和元数据字符串；
- 其他通过 `rg` 找到的硬编码旧顺序。

建议执行：

```bash
rg -n "load_deviation.*movement_count|中间重载偏差.*移动次数|split_bay_count.*load_deviation" .
```

实验 JSON 中 `objective` 数组必须按新顺序保存。各具名字段仍分别保存，避免旧实验记录被误读。不要修改已有历史实验文件。

### 3.3 搜索引导与正式目标分开处理

第八步的中间状态首先要消除作业缺口并得到合法的 `H-1` 排程。因此：

- 不得把“少移动”放到缺口数、超载量、可行性之前；
- 不得因为一次准备动作暂时增加移动就禁止该动作；否则可能破坏 `209→208` 的能力；
- 最终完整合法候选的接受必须使用新的 `objective_key`；
- `_shortening_potential` 等可行性导向的中间评分保持“先可行、后次目标”。若同一中间评分中同时出现移动和负载偏差，才把移动放在负载偏差之前。

构造阶段的 UCB 奖励不是正式目标。目前奖励中负载偏差项系数为 `0.01`、移动项为 `0.002`。建议分两个提交处理：

1. 第一提交只修改正式词典序、测试、输出和文档；
2. 第二提交再将两个很小的搜索引导系数对调为移动 `0.01`、负载偏差 `0.002`，并单独做对照。

如果第二提交没有改善移动次数或降低了 `H=208` 命中能力，应回退第二提交，但保留第一提交。不要大范围提高所有 `move_penalty`，因为这可能阻止为缩短工期所需的桥吊换位。

## 4. 单元测试

修改 `tests/test_solver_improvements.py::test_four_objective_priority_order`，至少覆盖：

1. 较短工期始终胜出，即使拆分、移动、负载偏差都更差；
2. 同工期时，较少拆分始终胜出，即使移动次数更多；
3. 同工期、同拆分时，较少移动始终胜出，即使负载偏差更大；
4. 只有工期、拆分和移动都相同时，较小负载偏差才胜出；
5. `assignment_count` 和 `reversal_count` 仍不参与正式比较。

建议明确加入以下关键断言：

```python
self.assertLess(
    candidate(moves=1, deviation=10_000).objective_key,
    candidate(moves=2, deviation=0).objective_key,
)
self.assertLess(
    candidate(split=0, moves=10_000).objective_key,
    candidate(split=1, moves=0).objective_key,
)
```

然后运行完整测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

不得只运行新测试。当前基线为 29 个测试全部通过；如果测试数量发生变化，在报告中注明。

## 5. Mentor 固定实例对照实验

固定输入：

```text
experiments/mentor_step8_baseline_20260917_h209/input/instance.json
```

固定源排程：

```text
experiments/mentor_step8_baseline_20260917_h209/input/source_schedule.json
```

现有旧顺序基线记录：

```text
experiments/mentor_step8_cumulative_local_20260917/beam_trajectory_seed0_300s/records.json
```

该基线中：

- trajectory 在约 160.75 秒找到合法 `H=208`，旧顺序目标为 `[208, 3, 2198, 20]`；
- beam 运行 300 秒，没有找到候选；这不代表全局不存在合法方案。

先做短时冒烟测试，确认程序、校验器和记录格式正常。随后用完全相同的 seed、输入、源排程、模式和预算运行新版本：

```bash
.venv/bin/python tools/evaluate_critical_step.py \
  --out experiments/mentor_step8_move_priority_20260917/seed0_300s \
  --experiment direct \
  --fixed-input experiments/mentor_step8_baseline_20260917_h209/input/instance.json \
  --fixed-source experiments/mentor_step8_baseline_20260917_h209/input/source_schedule.json \
  --seeds 0 \
  --budgets 300 \
  --modes trajectory beam \
  --allow-edge-exit \
  --verbose-windows
```

trajectory 和 beam 各自拥有 300 秒，不能共享一个 300 秒预算。正式计时测试时不要画每个窗口的图；搜索结束后只给最终合法最好方案补图，避免绘图占用搜索预算。

至少报告：

| 指标 | 旧顺序 | 新顺序 |
|---|---:|---:|
| 是否找到合法改进 |  |  |
| 完工时间 |  |  |
| 拆分贝位数 |  |  |
| 移动次数 |  |  |
| 负载偏差 |  |  |
| 首次找到改进的时间 |  |  |
| 实际运行时间 |  |  |
| evaluated / calls |  |  |

注意：旧记录 `[208, 3, 2198, 20]` 使用旧字段顺序。按新顺序表达同一个方案应为 `[208, 3, 20, 2198]`，不能把 2198 误当成移动次数。

## 6. 验收标准

必须全部满足：

- 正式目标在代码、测试、控制台、README、评估脚本和新实验记录中一致为 `(makespan, split_bay_count, movement_count, load_deviation)`；
- 完整测试全部通过；
- 独立校验器确认输出方案合法；
- 任意合法 `H=208` 仍严格优于任意 `H=209`；
- 同工期、同拆分时，移动少的方案一定胜出；
- 同工期时，不能为了少移动接受更多拆分贝位；
- 不修改移动次数定义，不修改 `move_time=0` 的含义，不重新引入大窗口；
- 不宣称一次 seed 的结果证明新顺序整体更好；如 300 秒结果退化，应如实保留对照数据并定位是正式排序还是搜索引导导致。

## 7. 提交建议

建议形成两个可独立回退的提交：

1. `Prioritize crane movements before load deviation`
2. `Align search reward with movement priority`（只有完成单独对照后再保留）

提交时只加入本任务修改的源码、测试、README 和小型报告。不要加入历史实验目录、大量 PNG、现有未跟踪文件或覆盖用户的输出文件。

## 8. 可直接交给 Luna Max 的指令

> 请读取 `STEP8_MOVE_PRIORITY_LUNA_MAX_PLAN.md` 并完整执行。先从当前 HEAD 创建 `codex/step8-move-priority` 分支，保留所有用户文件和未跟踪实验。第一阶段只把正式词典序目标改为 `(makespan, split_bay_count, movement_count, load_deviation)`，同步测试、README、控制台和评估记录格式，完整运行测试。第二阶段才单独评估是否需要交换构造搜索奖励中的移动与负载偏差小权重，不得让移动目标压过工期、拆分或可行性缺口。最后在指定 mentor 固定输入和固定源排程上，用 trajectory、beam 分别独立运行 300 秒、seed=0，独立校验合法性，并与现有旧顺序记录比较工期、拆分、移动次数、负载偏差和首次改进时间。不要把 beam 超时描述为不可行证明，不要删除或提交无关实验文件。
