# 真实小例：工作/位置联合事务

这些小例不是只改桥吊位置后再让全局解码器重派工作，而是直接给出完整的 `(时间, 桥吊) -> work_bay/idle` 工作账本；两份结果都通过项目独立校验器。

| 小例 | 事务 | 结果 |
|---|---|---|
| 相邻接力 | `tail_relay`，Q1→Q2，源区间 `[1,4)`，声明区段 `[0,4)`，2 台相邻桥吊、1 个区段 | Q1 的贝位 2 工作转给相邻 Q2；区段外第 0 槽保持不变 |
| 同贝位残余交换 | `paired_residual_exchange`，Q1 贝位 2 的间隙 `[2,3)` 与末端 `[4,5)` 两个 1 槽区段 | 连续性键从 `[0,1,0,0,0]` 变为 `[0,0,0,0,0]`，消除一次回访/碎片 |

证据文件：

- [相邻接力事务 JSON](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/real_small_examples/relay_transaction.json)
- [相邻接力修改后排工图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/real_small_examples/relay_after.png)
- [残余交换事务 JSON](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/real_small_examples/residual_transaction.json)
- [残余交换修改前排工图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/real_small_examples/residual_source.png)
- [残余交换修改后排工图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/real_small_examples/residual_after.png)

对应的守恒、越界、超长区段、非相邻桥吊和 `source_hash` 错配拒绝测试在 [test_solver_improvements.py](/Users/yy/Documents/ChatGPT/CWP/tests/test_solver_improvements.py) 中。
