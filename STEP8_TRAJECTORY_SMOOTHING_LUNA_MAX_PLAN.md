# 第八步 trajectory 合法缩短后的轨迹平滑改进方案

## 1. 任务结论与目标

当前 trajectory 的主要问题不是“目标顺序写错”，而是：

> 找到第一个合法 `H-1` 方案后立即返回，没有利用剩余预算在固定的新工期内继续减少拆分、移动和短时往返。

本轮目标是在保留现有 `209→208` 能力的前提下，为 trajectory 增加“合法缩短后的同工期平滑阶段”。正式目标顺序保持：

```text
(makespan, split_bay_count, movement_count, load_deviation)
```

不再通过调整目标文字或构造阶段 UCB 权重声称改善第八步。直接第八步测试不会使用构造阶段 UCB 奖励；此前交换奖励系数没有效果，已经回退。

## 2. Git 基线和保护要求

- 当前分支：`codex/step8-move-priority`
- 当前 HEAD：`a5e8b127edf5e18d4d349e8a61ffa72b2870e07f`
- 正式目标顺序提交：`46764c7 Prioritize crane movements before load deviation`
- 当前完整测试：29/29 通过
- 新建分支建议：`codex/step8-trajectory-smoothing`

从当前 HEAD 创建新分支。工作区已有大量用户实验目录和未跟踪文件，禁止删除、覆盖、清理或批量加入提交。只提交本任务涉及的源码、测试、文档和小型结果报告。

## 3. 已确认的问题证据

固定源排程的新顺序目标为：

```text
[209, 1, 12, 2252]
```

当前 trajectory 返回：

```text
[208, 3, 20, 2198]
```

旧目标顺序版本和新目标顺序版本的1040条槽记录逐条完全一致；只是 objective 数组的字段顺序改变了。

源方案负荷：

```text
[209, 209, 186, 180, 100]
```

缩短后负荷：

```text
[208, 207, 189, 180, 100]
```

trajectory 为了给Q1、Q2卸载，增加了以下短时接力。程序时间从0开始：

- Q2：贝位9 → 时间1到贝位7工作1个时段 → 时间2回贝位9；
- Q3：贝位13 → 时间1到贝位9工作1个时段 → 时间2回贝位13；
- Q3：时间18～19到贝位12工作2个时段 → 时间20回贝位13；
- Q3：时间36～37到贝位11工作2个时段 → 时间38回贝位13。

这四段不是空跑，但都是短访往返，共引入8次位置变化。移动次数由12增至20。拆分贝位由1个增至3个：源方案只有贝位11拆分；新方案又拆分了贝位7和9。

这些时间点和指标必须变成固定回归检查，不能只凭排工图肉眼判断。

## 4. 根因

当前 `_trajectory_repair` 和 `_cumulative_local_trajectory_repair` 在获得第一个合法缩短方案时直接 `return`。正式 `objective_key` 只在存在多个完整候选时才能发挥作用；现在通常只有第一个候选，因此把移动次数提升到第三目标不会自动改变轨迹。

当前300秒测试中，trajectory 约160秒找到 `H=208` 后结束，约140秒预算没有用于同工期平滑。这部分时间应成为本轮改进的主要资源。

## 5. 总体流程

把 trajectory 拆成两个明确阶段：

```text
固定源H=209
    ↓
阶段A：现有累计小窗口修复，寻找第一个合法H=208
    ↓
独立校验并保存 first_feasible
    ↓
阶段B：固定H=208，在剩余预算内做小窗口轨迹平滑
    ↓
独立校验所有完整候选，只保留当前最好合法方案
    ↓
返回 polished_best；没有改善时安全返回 first_feasible
```

阶段A保持现有行为和随机种子语义，避免丢失已经验证的 `H=208` 命中能力。阶段B不得重新放宽到大窗口，也不得重新执行第7步或完整全局求解。

## 6. 阶段B：同工期平滑器

建议新增独立函数，例如：

```python
_refine_same_horizon_trajectory(
    W, M, starts, candidate, deadline, seed,
    *, move_time=0, attempt_trace=None,
) -> tuple[_CandidateSchedule, int]
```

硬约束：

- 输出工期必须等于输入候选工期；
- 必须完成全部工作量；
- 保留强制首槽作业；
- 始终满足安全间距、不可交叉、贝位互斥；
- 不改变 `move_time=0` 的定义；
- 任何未完成或未通过独立校验的中间状态都不能成为输出；
- 没有找到更好候选时原样返回输入候选。

最终合法候选继续按正式 `objective_key` 比较。对于正式目标完全相同的候选，可以使用下面定义的短访指标作为确定性附加 tie-break，但不得让该指标压过工期、拆分或移动次数。

## 7. 短访指标

新增只读诊断函数，例如：

```python
_trajectory_smoothness(candidate, M, short_visit_limit=2)
```

建议返回：

```text
(short_excursion_count, short_excursion_work, position_block_count)
```

定义：

- position block：同一桥吊在同一位置连续停留的最大时间段；
- short excursion：形如 `A → B → A` 的位置块，`B != A`，B段长度不超过2，并且B段至少有一次作业；
- short_excursion_work：所有短访段内的作业槽总数；
- position_block_count：所有桥吊位置块总数，用于正式目标相同时的稳定性诊断。

当前固定 `H=208` 候选应检测到至少上述4个短访段。为避免边界定义争议，测试应直接断言Q2/Q3这四段被识别，而不是只断言总数。

短访指标第一版只用于：

- 诊断和日志；
- 正式 objective 完全相同时的 tie-break；
- 引导阶段B优先选择需要处理的局部窗口。

不要第一版就加入硬性“最短停留3个时段”约束，因为这可能直接排除所有合法 `H=208` 方案。是否设为硬约束必须由实验数据决定。

## 8. 必须实现的局部算子

阶段B不能继续依赖任意单点随机修改，应优先实现面向短访的区段算子。

### 8.1 短访消除

识别 `A → B → A` 的1～2时段短访，尝试将该段位置恢复为A，同时把B段承担的工作量重新安排到：

1. 同一桥吊已有的较长B访问段；
2. 已经在B作业的另一台桥吊且仍有同工期容量的位置；
3. 相邻桥吊的一段连续访问中。

每个提案必须同时处理“位置”和“工作量”，不能只删移动而遗失作业。

### 8.2 同贝位区段合并

如果同一桥吊对同一贝位存在多个分散作业块，尝试把短块的工作合并到较长块附近，通过移动区段边界而不是新增单点访问完成。

### 8.3 连续接力批量化

当前时间1存在Q2与Q3同时向左短移的接力。为这种情况增加多桥吊联动算子：一次选择相邻2～3台桥吊，把零散的单时段接力改成连续区段转移。整个区段的每一行都检查安全间距，不能分别修改单吊后再补救冲突。

### 8.4 访问边界滑动

允许把一个访问块的开始或结束向前/后滑动，使同一位置的作业连续化。提案应以完整位置块为单位，不以孤立时间点为主要单位。

### 8.5 贝位归属简化

优先针对新增拆分贝位7和9，尝试把其中一台桥吊的少量工作完整归还给另一所有者。如果减少拆分会使H=208不可行，应保留当前合法最好方案，不得生成伪改善。

## 9. 搜索与接受规则

维护至少三个对象：

- `first_feasible`：阶段A首次找到并校验通过的 `H-1` 方案；
- `current_state`：阶段B允许退火或局部扰动使用的当前状态；
- `best_legal`：独立校验通过且按正式目标最好的完整方案。

规则：

1. `best_legal` 永远不能被更差方案覆盖；
2. 中间状态可以暂时不优，但只能在内存中用于搜索，不能发布；
3. 每得到一个完整候选，先检查工期固定，再调用独立校验器；
4. 正式目标更好时更新 `best_legal`；
5. 正式目标相同时，才比较短访指标；
6. 截止时无论当前状态如何，都返回 `best_legal`；
7. 禁止找到第一个H=208后立即结束整个 direct trajectory 测试。

可以保留一个小型同工期候选池，按位置块签名去重，建议上限8～16。不要因为一个候选负载偏差较小就淘汰移动结构明显不同的同工期候选。

## 10. 时间预算

整个 direct trajectory 仍只有用户给出的单一绝对 deadline。

- 阶段A使用现有累计局部搜索，直到首次找到合法 `H-1`；
- 阶段B使用剩余时间，但至少预留2秒用于最终校验和序列化；
- 如果首次可行方案接近截止时间，立即安全返回，不得超时；
- 搜索期间不画窗口图；结束后只画 `first_feasible` 和 `polished_best`；
- 总运行不得通过“阶段A 300秒 + 阶段B 300秒”偷偷变成600秒。

当前基线约160秒首次命中，因此理论上有约138秒可用于平滑。

## 11. 日志和输出

扩展 `tools/evaluate_critical_step.py` 的 trajectory 记录，至少增加：

```text
time_to_first_feasible
first_feasible_objective
first_feasible_smoothness
polish_seconds
polish_attempts
polish_complete_candidates
polish_legal_improvements
best_objective
best_smoothness
```

同时保存：

- `first_feasible.json`
- `polished_best.json`
- `first_feasible.png`
- `polished_best.png`

图片标题必须明确区分“首次合法缩短方案”和“同工期平滑后方案”。如果没有改善，两张图可以相同，但报告必须写明“没有观察到平滑收益”。

## 12. 单元测试

至少增加以下测试：

1. 短访识别：准确识别Q2的 `9→7→9`、Q3的 `13→9→13`、`13→12→13`、`13→11→13`；
2. 没有短访的连续长访问不会被误判；
3. 同工期平滑器永远不改变 makespan；
4. 平滑器不能丢失工作量、破坏强制首槽或安全间距；
5. 找不到改善时逐槽返回原候选；
6. 找到正式目标更好的候选时正确更新；
7. 正式目标相同时，短访更少者胜出；
8. deadline 很短时仍返回已校验的 `first_feasible`；
9. 固定seed下结果可复现；
10. 原有29个测试全部继续通过。

完整运行：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 13. 固定实例实验

输入：

```text
experiments/mentor_step8_baseline_20260917_h209/input/instance.json
```

源排程：

```text
experiments/mentor_step8_baseline_20260917_h209/input/source_schedule.json
```

当前基线记录：

```text
experiments/mentor_step8_move_priority_20260917/seed0_300s/records.json
```

基线结果：

```text
trajectory: FOUND, 160.363s, [208, 3, 20, 2198]
beam: TIMEOUT, 300.000s
```

本轮主要修改 trajectory。先做5～10秒冒烟测试，再运行一次正式300秒、seed=0的 trajectory 对照。beam代码如果未修改，只做短预算回归即可，不需要再浪费300秒证明同一件事。

建议正式命令按现有评估脚本参数执行，并输出到新目录：

```text
experiments/mentor_step8_trajectory_smoothing_20260917/seed0_300s/
```

## 14. 验收标准

硬性要求：

- 返回方案独立校验通过；
- 最终工期保持208，不能因平滑退回209；
- 最终正式 `objective_key` 不得劣于当前 `[208, 3, 20, 2198]`；
- 找不到改善时必须安全返回当前基线方案；
- 单次正式 trajectory 总预算仍为300秒；
- 不重新引入大窗口；
- 不删除或覆盖已有实验文件。

观察性目标，不能提前保证：

- 移动次数少于20；或者
- 在移动次数相同时，四个已知短访至少消除一个；或者
- 拆分贝位少于3且移动次数不出现明显恶化。

如果最终仍为同一方案，应明确报告“本次算子在300秒、seed=0下没有观察到轨迹改善”，不能用更多迭代数或新增代码代替效果证据。

## 15. 提交建议

建议拆成三个可独立回退的提交：

1. `Add trajectory short-excursion diagnostics`
2. `Refine feasible Step 8 schedules at fixed horizon`
3. `Record trajectory smoothing comparison`

诊断提交不应改变求解结果；平滑器提交必须有针对性测试；实验报告单独提交。PNG和大型 records.json 不加入Git。

## 16. 可直接交给 Luna Max 的指令

> 请读取 `STEP8_TRAJECTORY_SMOOTHING_LUNA_MAX_PLAN.md` 并完整执行。从当前 `codex/step8-move-priority` 的 HEAD 新建 `codex/step8-trajectory-smoothing`，保留所有已有文件和实验。不要再改目标顺序或构造阶段UCB权重。先实现短访诊断并验证当前候选的四段短访，再把 trajectory 改为找到首个合法H=208后利用剩余预算做固定工期的小窗口轨迹平滑，重点实现短访消除、区段合并和多桥吊连续接力算子。任何输出都必须完整合法并通过独立校验；没有改善时返回原H=208方案。完整运行测试，并在固定mentor输入、固定源、seed=0、总预算300秒下与 `[208,3,20,2198]` 对照，报告首次可行时间、平滑耗时、移动次数、拆分贝位、短访数和最终图。不要把搜索量增加描述为效果改善。
