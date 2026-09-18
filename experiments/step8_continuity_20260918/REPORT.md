# Step 8 continuity experiment report

分支：`codex/step8-continuity`。已有实验目录未删除、未纳入本次提交。

## 实现与校验

- 连续性诊断新增作业时间块、贝位空档、桥吊位置块、任意长度工作回访、纯让行回访、折返和移动统计；旧 `smoothness` 字段继续保留。
- 固定 H 解码器保留绝对时间轴和 idle/offrail 槽，不通过删掉全局 idle 行来制造改善。
- 连续性提案只修改一个小区段或两个显式成对区段；候选必须通过容量检查、独立校验和首槽约束。正式目标仍是 `(makespan, split_bay_count, movement_count, load_deviation)`。
- 连续性搜索单独保存 `formal_best` 与 `continuity_best`，最多保留 16 个联合位置/作业签名候选；不再使用“接受 6 次就停止”的硬限制。
- `.venv/bin/python -m unittest discover -s tests -v`：41 项全部通过。

输入 hash：

- H=209 instance：`998a448e1452c30291d4c2e42bb712601756c42c7d6de4f95b03998a2fb06cbf`
- H=209 source：`5cfe154aa22a9511ea9af3e8984a402e4ca201266b7de59167c3f22969d5c944`
- H=280 instance：`77c2284d530ed7825b1e78ef7f702452c769f2652f98109d9edcfd7e06ef144d`
- H=280 source：`b5bbebc7be016cf742d80570f75eb5410d40524df21e73a4a4fe312ced31e4d6`

## 结果

连续性 key 的顺序为 `(work_revisit_count, bay_fragmentation, reversal_count, movement_count, load_deviation)`；它只是诊断排序，不替换正式目标。

| 实验 | source | first feasible | formal best | continuity best |
|---|---:|---:|---:|---:|
| 固定 H=208 连续性、300 s（搜索 298.08 s，预留 2 s） | `[208,2,17,2126]` | 固定源 | `[208,1,15,1900]` | `[208,1,15,1900]` |
| H=209 完整 trajectory、300 s | `[209,1,12,2252]` | `[208,2,19,2216]` | `[208,1,17,2008]` | `[208,2,15,2008]` |
| H=280 四吊 trajectory、300 s | `[280,0,9,2332]` | `[279,1,11,2320]` | `[279,0,9,1984]` | `[279,0,9,1984]` |
| H=280 四吊 beam、300 s | `[280,0,9,2332]` | — | `[227,2,13,1696]` | 复用 beam 正式结果 |

H=208 的连续性 key 从 `[5,6,9,17,2126]` 降到 `[3,3,7,15,1900]`。Q3 的贝位 11 长回访在最终方案中不再出现；但贝位 2 仍有两个作业块，原来的 `[0,1)`、`[207,208)` 变为 `[0,1)`、`[201,202)`，所以不能声称贝位 2 已经连续完成。对应的早期补做候选因贝位 7 容量不足或末端安全间距冲突被拒绝。

H=209 的正式最优和连续性最优发生冲突：正式结果少一个 split bay，连续性结果少 2 次移动且连续性 key 更好；两者均保留，不能用其中一个掩盖另一个。H=280 trajectory 没有增加 split bay，且保持 9 次移动。

原 H=209 `source_schedule.json` 在独立校验器的 t=180 存在 Q3=13、Q4=14 的距离 1 冲突。实验使用项目已有 `--allow-edge-exit` 仅对已完成边缘桥吊做输入合法化；原文件和 hash 保留，实际适配后的 source 另存为实验目录中的 `source.json`。

## 文件位置

- [固定 H=208 记录（截止安全版）](</Users/yy/Documents/ChatGPT/CWP/experiments/step8_continuity_20260918/fixed_h208_continuity_seed0_300s_v2/records.json>)
- [固定 H=208 source 图](</Users/yy/Documents/ChatGPT/CWP/experiments/step8_continuity_20260918/fixed_h208_continuity_seed0_300s_v2/source.png>)
- [H=209 trajectory 记录](</Users/yy/Documents/ChatGPT/CWP/experiments/step8_continuity_20260918/full_h209_trajectory_seed0_300s/records.json>)
- [H=280 trajectory 记录](</Users/yy/Documents/ChatGPT/CWP/experiments/step8_continuity_20260918/h280_trajectory_seed0_300s/records.json>)
- [H=280 beam 记录](</Users/yy/Documents/ChatGPT/CWP/experiments/step8_continuity_20260918/h280_beam_seed0_300s/records.json>)

H=280 beam 没有改动 beam 核心逻辑；本次 300 秒调用重新完成并独立校验了记录，图使用已有同一 beam 结果图复用，文件为 `beam_best_reused.png`。

首次未预留尾部的 H=208 记录也保留在 `fixed_h208_continuity_seed0_300s/`；v2 使用同一搜索结果但增加了 2 秒尾部预留，作为正式引用结果。

源码提交：`721a5c5`（连续性源码）、`cc34554`（回归测试）、`07a4df4`（固定 H 实验工具）、`f36ac71`（source artifact）、`8401bdb`（成对区段记录）。
