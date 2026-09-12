# 精英方案池与关键桥吊窗口重排复测

本轮在 `example_input.json`（39 贝位、9 台桥吊）上实施交接文档 P1/P2：

- 最多保存 16 个按贝位归属和移动路线去重的完整精英方案；普通轨迹修复轮换精英来源。
- 选择最后完成桥吊附近的 2～4 台桥吊，在晚期时间窗口用小束搜索重新排列位置行；窗口外和链外轨迹固定。
- 新增运行指标：精英方案数、精英修复数、关键桥吊修复迭代数和改进数。

60 秒求解预算、seed=20260910 的 CLI 结果：

```text
makespan=44
lower_bound=41
search_seconds=51.233
program_seconds=68.800
elite_pool_size=16
elite_repairs=20
critical_repair_iterations=772633
critical_repair_improvements=0
trajectory_repair_improvements=4
```

独立校验通过，结果在 `experiments/next_large39_60/`。相同输入此前 60 秒记录也是工期44；因此本轮新增邻域暂未显示工期改善，不能把精英池或窗口束搜索写成已突破难例。它们扩大了搜索结构，后续应继续做算子消融和更长/不同种子的测试。

小规模回归（12 秒、seed=20260910）仍得到工期19、下界19；15项自动测试通过。
