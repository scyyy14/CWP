# Step 8 trajectory smoothing 对照报告

## 执行范围

- 分支：`codex/step8-trajectory-smoothing`
- 输入实例：`mentor_step8_baseline_20260917_h209/input/instance.json`
- 固定源：`mentor_step8_baseline_20260917_h209/input/source_schedule.json`
- seed：`0`
- trajectory 总预算：`300 s`
- 画图：只在搜索结束后生成 `first_feasible` 和 `polished_best` 两张图；搜索阶段预留 `15 s` 画图时间。
- 最终总耗时：`170.925694 s`

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
| 最终 polished best | `[208, 2, 17, 2126]` | `[0, 0, 22]` | 显式区段算子找到并按正式目标保存 |

结论：

1. 硬性目标满足：最终仍为 `H=208`，且优于旧基线的正式目标（拆分贝位 `3→1`，负荷偏差 `2198→2008`）。所有发布候选都经过独立校验。
2. 阶段 A 已经改善了旧 trajectory：首次合法解的拆分贝位为 `2`、移动次数为 `19`，短访从 `4` 段降为 `2` 段。
3. 阶段 B 的显式区段算子进一步找到 `[208,2,17,2126]`：移动次数比旧基线少 `3` 次，短访四段全部消除，且通过独立校验。
4. 这次最终返回图确实比旧基线和首个合法图更平滑；它没有把拆分贝位降到 `1`，但按照正式目标仍优于旧基线，因为拆分贝位从 `3` 降到了 `2`，随后移动次数又从 `20` 降到了 `17`。

## 时间和搜索日志

- 首次合法 H-1 时间：`158.782871 s`
- 平滑阶段耗时：`0.171388 s`
- 平滑尝试数：`87`
- 通过独立校验的完整候选数：`14`
- 合法改进数：`6`
- 总评估数：`883,337`
- 总墙钟时间：`170.925694 s`

“评估数”只说明搜索工作量，不作为效果指标。

## 输出文件

最终实验目录：

```text
seed0_300s_segment_operators/
```

其中包含：

- `records.json`：完整指标、阶段轨迹和候选摘要；
- `window_plots/fixed/seed_0/budget_300s/trajectory/first_feasible.json`；
- `window_plots/fixed/seed_0/budget_300s/trajectory/first_feasible.png`；
- `window_plots/fixed/seed_0/budget_300s/trajectory/polished_best.json`；
- `window_plots/fixed/seed_0/budget_300s/trajectory/polished_best.png`。

PNG 和大型 JSON 只保留在实验目录，不加入 Git 提交。
