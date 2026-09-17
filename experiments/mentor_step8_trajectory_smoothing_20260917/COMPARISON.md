# Step 8 trajectory smoothing 对照报告

## 执行范围

- 分支：`codex/step8-trajectory-smoothing`
- 输入实例：`mentor_step8_baseline_20260917_h209/input/instance.json`
- 固定源：`mentor_step8_baseline_20260917_h209/input/source_schedule.json`
- seed：`0`
- trajectory 总预算：`300 s`
- 画图：只在搜索结束后生成 `first_feasible` 和 `polished_best` 两张图；搜索阶段预留 `15 s` 画图时间。
- 最终总耗时：`289.495107 s`

正式目标顺序仍为：

```text
(makespan, split_bay_count, movement_count, load_deviation)
```

短访诊断三元组为：

```text
(short_excursion_count, short_excursion_work, position_block_count)
```

## 结果

| 方案 | 目标 | 短访诊断 | 说明 |
|---|---:|---:|---|
| 固定源 | `[209, 1, 12, 2252]` | `[0, 0, 17]` | 输入源排程 |
| 旧 trajectory 基线 | `[208, 3, 20, 2198]` | `[4, 6, 25]` | `mentor_step8_move_priority_20260917/seed0_300s` |
| 首个合法 H-1 | `[208, 2, 19, 2216]` | `[2, 4, 24]` | 阶段 A 首次找到并通过独立校验 |
| 阶段 B 观察到的最平滑合法候选 | `[208, 2, 17, 2126]` | `[0, 0, 22]` | 轨迹中间候选；没有被正式目标最终选为 best |
| 最终 polished best | `[208, 1, 26, 2008]` | `[2, 4, 31]` | 按正式目标保存和返回 |

结论：

1. 硬性目标满足：最终仍为 `H=208`，且优于旧基线的正式目标（拆分贝位 `3→1`，负荷偏差 `2198→2008`）。所有发布候选都经过独立校验。
2. 阶段 A 已经改善了旧 trajectory：首次合法解的拆分贝位为 `2`、移动次数为 `19`，短访从 `4` 段降为 `2` 段。
3. 阶段 B 确实找到过更平滑的合法 H=208 候选：`[208,2,17,2126]`，短访为 `0`，但因为正式目标把拆分贝位放在移动次数之前，最终 `[208,1,26,2008]` 在形式目标上更优，因此被保留。
4. 因此本轮没有证明“最终返回图一定比首个合法图更平滑”。它证明了平滑候选可被找到、校验并记录，但正式目标与轨迹平滑之间存在明确权衡；不能把最终方案宣称为短访或移动次数的改善。

## 时间和搜索日志

- 首次合法 H-1 时间：`158.782871 s`
- 平滑阶段耗时：`119.036814 s`
- 平滑尝试数：`71`
- 通过独立校验的完整候选数：`6`
- 合法改进数：`6`
- 总评估数：`1,254,020`
- 总墙钟时间：`289.495107 s`

“评估数”只说明搜索工作量，不作为效果指标。

## 输出文件

最终实验目录：

```text
seed0_300s_verified/
```

其中包含：

- `records.json`：完整指标、阶段轨迹和候选摘要；
- `window_plots/fixed/seed_0/budget_300s/trajectory/first_feasible.json`；
- `window_plots/fixed/seed_0/budget_300s/trajectory/first_feasible.png`；
- `window_plots/fixed/seed_0/budget_300s/trajectory/polished_best.json`；
- `window_plots/fixed/seed_0/budget_300s/trajectory/polished_best.png`。

PNG 和大型 JSON 只保留在实验目录，不加入 Git 提交。
