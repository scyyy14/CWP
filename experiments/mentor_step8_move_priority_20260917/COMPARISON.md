# Step 8 movement-priority comparison

固定输入：`mentor_step8_baseline_20260917_h209/input/instance.json`

固定源排程：`mentor_step8_baseline_20260917_h209/input/source_schedule.json`

移动时间：`move_time=0`；seed：`0`；trajectory 和 beam 各自预算：`300s`。

最终正式目标顺序为：

```text
(makespan, split_bay_count, movement_count, load_deviation)
```

## 结果

| 阶段 | 模式 | 状态 | 实际时间 | evaluated | 结果 |
|---|---|---|---:|---:|---|
| 目标顺序改动 | trajectory | FOUND | 160.363s | 982,833 | `[208, 3, 20, 2198]` |
| 目标顺序改动 | beam | TIMEOUT | 300.000s | 108,017,693 | 未找到候选 |
| 目标顺序 + 搜索奖励交换 | trajectory | FOUND | 160.383s | 1,029,041 | `[208, 3, 20, 2198]` |
| 目标顺序 + 搜索奖励交换 | beam | TIMEOUT | 300.000s | 111,407,122 | 未找到候选 |

固定源新顺序目标为 `[209, 1, 12, 2252]`。

两阶段 trajectory 都找到同一个合法 `H=208` 方案，移动次数和负载偏差均没有改善；第二阶段 beam 只是评估量增加，没有找到合法候选。因此第二阶段搜索奖励交换已回退，最终只保留正式目标顺序改动。

`TIMEOUT` 只表示预算内没有找到候选，不表示不存在合法改进方案。

## 最终产物

- 第一阶段结果：`seed0_300s/records.json`
- 第一阶段最终图：`seed0_300s/trajectory_h208.png`
- 第二阶段对照：`reward_aligned_seed0_300s/records.json`
- 当前最终代码：分支 `codex/step8-move-priority`，正式目标顺序改动提交 `46764c7`；第二阶段尝试已由 `3e03c99` 回退。
