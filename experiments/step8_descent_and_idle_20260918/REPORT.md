# 第八步持续降工期、局部接力与停工诊断实验报告

执行分支：`codex/step8-descent-and-idle`  
最终提交：`e9f2733`（上一阶段控制流程提交：`834ccbb`）  
目标顺序：`(makespan, split_bay_count, movement_count, load_deviation)`。  
本报告中的“找到”只表示候选通过独立校验器；不表示已经证明全局最优。

## 实现内容

- `cwp_solver.py` 增加逐吊停工诊断：开头等待、中途 idle、末次作业后的尾部空闲、移动槽和 offrail 分开统计，并输出 idle 区间。
- 第八步 trajectory 改为共享 absolute deadline 的多轮降工期流程：找到首个 H-1 后重新生成窗口，继续尝试 H-2、H-3；记录每轮 source/target/result/耗时。
- 为了保留已验证行为，新流程首先使用旧累计局部搜索作为“首个 H-1 种子”，找到后立即交回新流程继续下降；这不是全局求解器，也不扩大窗口。
- 固定 H 测试显式 `preserve_horizon`，只做同 H 局部改善，不把随机降工期混入固定 H 结果。
- 同时保留 `formal_best` 与 `operational_best`。候选池最多 16 个完整合法候选，按联合位置/作业轨迹去重。
- direct 实验的 plotting 不再延长搜索 deadline；beam 无候选时也保存 source/best JSON 和 PNG，并将 `best_is_source=true`。

## 回归与独立校验

```text
.venv/bin/python -m unittest discover -s tests -v
Ran 44 tests in 1.876s
OK
```

另外重新读取并调用独立校验器检查了 H=208 固定结果、H=209 三个 trajectory 结果、H=209 beam source、H=280 trajectory 和 beam best；全部通过。实验命令生成的 JSON 没有直接当作“只看图”的结果使用。

## 输入哈希

H=209：

- instance：`998a448e1452c30291d4c2e42bb712601756c42c7d6de4f95b03998a2fb06cbf`
- source：`5cfe154aa22a9511ea9af3e8984a402e4ca201266b7de59167c3f22969d5c944`
- source objective：`[209, 1, 12, 2252]`
- `move_time=0`，使用 `--allow-edge-exit` 与已有 H=209 实验保持一致。

H=280：

- instance：`77c2284d530ed7825b1e78ef7f702452c769f2652f98109d9edcfd7e06ef144d`
- source：`b5bbebc7be016cf742d80570f75eb5410d40524df21e73a4a4fe312ced31e4d6`
- source objective：`[280, 0, 9, 2332]`
- `M=4, move_time=0`。

## 300 秒实验结果

每个 nominal 300 秒实验都保留 15 秒输出预留，trajectory/beam 的搜索预算记录在各自 `records.json`。

| 实验 | 状态与首次新 H | formal_best | operational_best | 关键观察 |
|---|---|---|---|---|
| 固定 H=208，trajectory，seed 0 | `NO_SHORTENING_SAME_H_IMPROVEMENT` | `[208,1,14,1900]` | `[208,1,14,1900]` | source `[208,1,15,1900]`，移动 15→14；H 未改变 |
| H=209，trajectory，seed 0 | H=208，约 142.58 秒 | `[208,2,17,2008]` | `[208,2,17,2216]` | 没有 H=207；operational 保留较少内部 idle 的候选 |
| H=209，trajectory，seed 1 | H=208，约 183.77 秒 | `[208,3,17,2008]` | `[208,3,17,2008]` | 没有 H=207；结果对 seed 敏感 |
| H=209，trajectory，seed 2 | H=208，约 199.37 秒 | `[208,2,17,1990]` | `[208,2,17,1990]` | 没有 H=207；三次中 formal 最好 |
| H=209，beam，seed 0 | `TIMEOUT`，48/48 窗口无候选 | source `[209,1,12,2252]` | source | 没有合法修改，不等于全局无解 |
| H=280，trajectory，seed 0 | 依次出现 H=279/278/277/276 | `[276,1,11,2272]` | `[276,1,11,2272]` | 25 轮记录；相对旧 trajectory 的 279 有明显下降 |
| H=280，beam，seed 0 | 45 个窗口中 5 个候选 | `[227,2,13,1696]` | `[227,2,13,1696]` | 找到很短的合法方案，但 split 从 0 变 2；未证明全局最优 |
| H=209 消融：`--no-operational-repairs` | H=208，约 142.26 秒 | `[208,2,19,2216]` | `[208,2,19,2216]` | 关闭固定 H operational/polish；仍保留降工期 trajectory 局部修复 |

H=209 trajectory 的安全工作量下界为 177，H=280 样例为 221；因此 H=208 和 H=276 都不能称为理论最短。H=209 的三次测试均只到 H=208，且没有证据表明 H=207 在全局不存在。

## 排工图与 JSON

固定 H=208：

- JSON：`fixed_h208_300s_v3/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.json`
- 图：`fixed_h208_300s_v3/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.png`

H=209 trajectory seed 0：

- JSON：`full_h209_trajectory_seed0_300s_v5/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.json`
- formal 图：`full_h209_trajectory_seed0_300s_v5/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.png`
- operational 图：`full_h209_trajectory_seed0_300s_v5/window_plots/fixed/seed_0/budget_300s/trajectory/operational_best.png`

H=209 trajectory seed 1/2：

- seed 1：`full_h209_trajectory_seeds1_2_300s/window_plots/fixed/seed_1/budget_300s/trajectory/formal_best.png`
- seed 2：`full_h209_trajectory_seeds1_2_300s/window_plots/fixed/seed_2/budget_300s/trajectory/formal_best.png`

H=209 beam：

- source/best 图：`full_h209_beam_seed0_300s_v2/window_plots/fixed/seed_0/budget_300s/beam/source.png`、`best.png`
- 对应结果：`full_h209_beam_seed0_300s_v2/records.json`

H=280：

- trajectory 图：`h280_trajectory_seed0_300s_v2/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.png`
- beam 图：`h280_beam_seed0_300s_v2/window_plots/fixed/seed_0/budget_300s/beam/best.png`

每个实验目录还保存了 `records.json`、`manifest.json`、输入备份和代码备份；旧实验目录没有删除或覆盖。

## 结论与剩余问题

控制流程问题已经修复：H=280 证明首个 H-1 后确实会继续下降。H=209 上 trajectory 能稳定找到 H=208，但局部邻域仍不足以找到 H=207，也没有消除贝位 2 首尾回访。beam 在 H=209 无候选，在 H=280 找到 H=227，说明两个方法的搜索行为差异很大。

当前结果不能宣称“第八步已经找到最短方案”或“所有停工都可消除”。下一步若继续改进，应优先针对贝位 2 的早期/末端残余设计严格的双区段工作守恒事务，并继续保留当前分支和这些实验作为回归基线。

典型运行命令：

```bash
.venv/bin/python tools/evaluate_critical_step.py --experiment direct \
  --fixed-input experiments/mentor_step8_baseline_20260917_h209/input/instance.json \
  --fixed-source experiments/mentor_step8_baseline_20260917_h209/input/source_schedule.json \
  --seeds 0 --budgets 300 --modes trajectory beam \
  --allow-edge-exit --plot-windows \
  --out experiments/step8_descent_and_idle_20260918/<new-run>
```

