# Step 8：真实工作接力与连续性改进验收报告

执行日期：2026-09-19。实现分支：`codex/step8-work-transfer-continuity`。创建分支时基线为 `codex/step8-descent-and-idle@6ce78b4`。历史实验和旧输出均保留；本轮新输出只写入本目录。

## 结论先行

- 新增的严格固定工期事务已经实现：位置和工作归属一起修改，窗口外工作、位置和槽字段冻结；事务最多 3 台相邻桥吊、最多 2 个不重叠时间区段、每段最多 8 槽，并有完整工作账本和独立校验。
- 真实小例确实证明了两件事：相邻桥吊可以合法接力；同贝位的早/晚残余可以用两个短区段消除一次回访。49 个测试全部通过。
- 在正式 H209 源上，trajectory 三个种子和 beam 都没有找到 H208；没有把旧的全局重排结果冒充为严格局部事务结果。H209 源的 Q1 贝位 2/7 连续块没有被新事务进一步拆碎，但 Q4/Q5 的完工后空闲也没有被合法接力消除。
- 因此本轮“实现与验收”完成，但计划中的实效目标（H209→H208，或减少真实 H209 的 Q4/Q5 尾部停工）未达到，不能写成算法已经解决该实例。

## 输入、哈希与代码

H209 正式输入：

- instance：`N=21, M=5, move_time=0, sum(W)=884`，SHA-256 `998a448e1452c30291d4c2e42bb712601756c42c7d6de4f95b03998a2fb06cbf`
- source：目标 `[209,1,12,2252]`，SHA-256 `5cfe154aa22a9511ea9af3e8984a402e4ca201266b7de59167c3f22969d5c944`
- 文件：[instance.json](/Users/yy/Documents/ChatGPT/CWP/experiments/mentor_step8_baseline_20260917_h209/input/instance.json)、[source_schedule.json](/Users/yy/Documents/ChatGPT/CWP/experiments/mentor_step8_baseline_20260917_h209/input/source_schedule.json)

H280 回归输入：`M=4, move_time=0`；instance SHA-256 `77c2284d530ed7825b1e78ef7f702452c769f2652f98109d9edcfd7e06ef144d`，source SHA-256 `b5bbebc7be016cf742d80570f75eb5410d40524df21e73a4a4fe312ced31e4d6`。

本次修改文件：

- [cwp_solver.py](/Users/yy/Documents/ChatGPT/CWP/cwp_solver.py)：显式工作账本、严格事务区域校验、`paired_residual_exchange`、`tail_relay`、`idle_fill`、有界多步接力、连续性保护和候选池。
- [evaluate_critical_step.py](/Users/yy/Documents/ChatGPT/CWP/tools/evaluate_critical_step.py)：新增 `--work-transfer`、`--continuity-protection`、`--strict-local-transactions`、`--multi-relay/--no-multi-relay`、`--legacy-seed` 开关；beam 和 trajectory 共用 285 秒搜索 + 15 秒输出预留协议。
- [test_solver_improvements.py](/Users/yy/Documents/ChatGPT/CWP/tests/test_solver_improvements.py)：新增真实小事务、守恒、冻结区域、范围契约、连续性和哈希错配测试。

验证命令：

```text
.venv/bin/python -m py_compile cwp_solver.py tools/evaluate_critical_step.py
.venv/bin/python -m unittest discover -s tests -q
```

结果：`Ran 49 tests ... OK`。

## 实现验收

### 严格事务的边界

每个候选包含：源轨迹签名、`source_hash`、相邻桥吊链、1～2 个时间区段、区段内完整工作账本、边界位置和操作目标。候选生成后执行以下检查：

1. 工作账本覆盖全部 `(t,q)`，每个贝位的工作量精确等于 `W`，同一时刻不重复作业。
2. 位置必须与 `work_bay` 一致，安全间距和边界退出规则交给项目独立校验器复核。
3. 区段外的 `state/start_bay/end_bay/work_bay/move_id/move_step/move_steps` 逐槽比对，任何变化拒绝。
4. 区段最多两个、每段最多 8 槽、桥吊最多三台且必须连续相邻；区段之间的槽完全冻结。
5. 当前源轨迹签名和事务保存的 `source_hash` 都匹配后才解码；接受一个新 best 后重新从该 best 生成事务，避免旧提案套到新基线。
6. `strict_local_transactions=True` 时，固定 H polish 被拒绝的事务不会落入旧的“只改位置、再全局贪心分配工作”路径。

严格事务目前只支持 `move_time=0`。非零移动时间继续显式走已有的旧路径，不会静默按零移动时间解释；本轮正式输入全部是 `move_time=0`。

## 真实小例结果

小例说明和图在 [real_small_examples/README.md](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/real_small_examples/README.md)。

| 小例 | 修改前 | 修改后 | 验收 |
|---|---:|---:|---|
| 相邻接力，`W=[0,4,0,0,0], M=2` | `[4,0,0,8]` | `[4,1,2,4]` | Q1 贝位 2 工作交给相邻 Q2；事务外槽冻结；独立校验通过 |
| 同贝位残余交换，`M=1` | 连续性键 `[0,1,0,0,0]` | `[0,0,0,0,0]` | 两个 1 槽区段，消除一次回访和碎片；独立校验通过 |
| 反例 | — | — | 窗口外作业、超过 8 槽、4 台桥吊不相邻、`source_hash` 不匹配均被拒绝 |

相邻接力小例的正式目标变差是有意保留的：它证明的是“真实工作转移和合法冻结”，不是宣称所有接力都能改善四目标。

## 固定 H=208

计划要求的两个固定源都跑过，目标模式是 `same_horizon`，trajectory、严格工作事务、连续性保护均开启。

| 固定源 | 源目标 | formal / continuity / operational | 工作事务统计 | 结论 |
|---|---:|---:|---|---|
| 上轮 v5 `formal_best`，哈希 `6ba6f737d3d0e28c02294f8dccbb97ddaefc812be5c5b078db26a9ced376c392` | `[208,2,17,2008]` | 三者均 `[208,2,17,2008]` | 60 个完整候选、60 个事务级验证、0 接受；46 个连续性拒绝 | 没有改变；Q1 贝位 2 仍是 `[0,1)+[207,208)`，贝位 7 仍有早段小块 |
| 上轮 fixed-H v3 `formal_best`，哈希 `d1ef82b10add1539a1ad0c5ba8266bf97cbb36d50d9fd747e586061d5a9e6588` | `[208,1,14,1900]` | 三者均 `[208,1,14,1900]` | 92 个完整候选、24 个独立校验、0 接受；68 个连续性拒绝 | 没有改变；严格层没有制造新的窗口外修改 |

两组固定 H 的 source、formal、continuity、operational JSON 均独立重读并通过校验。固定 H 不能减少总非作业容量；H208 的 `M*H-sum(W)=156`，只能重新分配等待和工作归属。

图和完整记录：

- [v5 formal 排工图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/fixed_h208_v5_strict_300s/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.png)
- [v5 完整 records.json](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/fixed_h208_v5_strict_300s/records.json)
- [v3 formal 排工图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/fixed_h208_strict_300s/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.png)
- [v3 完整 records.json](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/fixed_h208_strict_300s/records.json)

## H=209 完整第八步

所有正式 H209 实验都使用同一个合法化 source，名义预算 300 秒，实际搜索预算 285 秒，另留 15 秒生成图；输出中的 `records.json`、三类 best JSON 和 PNG 都已保留。

源排工的关键结构：Q1 贝位 2 为 `[0,2)`，贝位 7 为 `[140,209)`；Q4 工作到 179 后尾部空闲 29 槽；Q5 工作到 99 后尾部空闲 109 槽；源连续性为 `work_revisit_count=1, bay_fragmentation=2`，内部等待 3 槽。

| 实验 | flags | formal best | continuity / operational | 首次可行 H | 结论 |
|---|---|---:|---:|---|---|
| trajectory seed 0 | transfer=on, guard=on, strict=on, multi=on | `[209,1,12,2234]` | 均 `[209,1,12,2234]` | 209 | 同 H 只改善 load deviation；无 H208 |
| trajectory seed 1 | 同上 | `[209,1,12,2252]` | 均 `[209,1,12,2252]` | 209 | 源方案保留；无 H208 |
| trajectory seed 2 | 同上 | `[209,1,12,2216]` | 均 `[209,1,12,2216]` | 209 | 同 H 改善 load deviation；无 H208 |
| beam seed 0 | 同上 | `[209,1,12,2252]` | 均 `[209,1,12,2252]` | — | 48 个严格事务候选均未接受，source 保留；无 H208 |

三条 trajectory 结果的 Q1 贝位 2/7 工作块仍与 source 相同；Q4/Q5 的尾部空闲也仍分别是 29/109 槽。也就是说，本轮严格事务没有再制造 Q1 回访，但也没有找到把 Q4/Q5 尾部变成有效接力的机会。

严格事务证据：

- seed 0：52 个事务候选、52 个账本验证、2 个独立候选校验，0 个接受；其余主要因连续性保护拒绝。
- seed 1：48 个候选、48 个账本验证，0 个接受。
- seed 2：58 个候选、58 个账本验证、4 个独立候选校验，0 个接受。
- beam：48 个候选均保留 source；beam 搜索本身没有找到可报告的修改方案。

图和记录：

- [H209 source 图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_trajectory_seed0_300s_v2/window_plots/fixed/seed_0/budget_300s/trajectory/source.png)
- [trajectory seed 0 formal 图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_trajectory_seed0_300s_v2/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.png)
- [trajectory seed 2 formal 图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_trajectory_seed2_300s/window_plots/fixed/seed_2/budget_300s/trajectory/formal_best.png)
- [beam source 保留图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_beam_seed0_300s_v2/window_plots/fixed/seed_0/budget_300s/beam/best.png)
- [trajectory seed 0 records](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_trajectory_seed0_300s_v2/records.json)、[seed 1 records](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_trajectory_seed1_300s/records.json)、[seed 2 records](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_trajectory_seed2_300s/records.json)、[beam records](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/full_h209_beam_seed0_300s_v2/records.json)

每个 trajectory formal/continuity/operational JSON 都包含 `idle.per_crane`、`continuity.work_blocks_by_bay` 和完整 `slots`；每次事务的声明区段、工作账本和源签名在 `window_results` 中保留。

## H209 消融

| 消融 | formal best | 解释 |
|---|---:|---|
| 关闭 work transfer，同时关闭 continuity/strict | `[209,0,11,2026]` | 这是旧的全局/位置解码路径，不属于严格新事务；可作为算法参考，不可与严格局部结果直接宣称同一范围 |
| 关闭 multi-relay | `[209,1,12,2234]` | 与 seed 0 一样，没有 H208 |
| 关闭 continuity protection | `[209,1,12,2234]` | 52 个事务都通过账本校验但 0 个正式接受；本实例失败不是保护阈值单独造成的 |

消融记录：[no work transfer](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/ablation_no_work_transfer_seed0_300s/records.json)、[no multi-relay](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/ablation_no_multi_relay_seed0_300s/records.json)、[no continuity protection](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/ablation_no_continuity_protection_seed0_300s/records.json)。

## H280 回归与范围契约审计

| 模式 | source | result | 审计结论 |
|---|---:|---:|---|
| trajectory | `[280,0,9,2332]` | `[273,2,20,2248]`；首次出现 H273 | 公开校验合法，但降 H 阶段仍使用旧的 horizon-changing local decoder；固定 H polish 才是严格事务，因此不能把 H273 当作“严格 8 槽事务找到的 H273” |
| beam | `[280,0,9,2332]` | `[227,2,13,1696]` | 公开校验合法，但 beam 实际枚举的窗口如 `[1,70)`、`[32,101)`，长度 69～70 且可含 4 吊，明显不满足本计划的 8 槽/3 吊契约；只作旧 beam 的 reachability 回归，不作严格局部性能证据 |

H280 的首次可行 trajectory 链为 H280→279→278→277→276→275→274→273。H280 beam 的 `window_results` 和最佳排工图均保留，未隐藏范围不合规事实。

图：

- [H280 trajectory formal 图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/regression_h280_trajectory_seed0_300s/window_plots/fixed/seed_0/budget_300s/trajectory/formal_best.png)
- [H280 trajectory continuity 图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/regression_h280_trajectory_seed0_300s/window_plots/fixed/seed_0/budget_300s/trajectory/continuity_best.png)
- [H280 beam best 图](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/regression_h280_beam_seed0_300s/window_plots/fixed/seed_0/budget_300s/beam/best.png)
- [H280 trajectory records](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/regression_h280_trajectory_seed0_300s/records.json)、[beam records](/Users/yy/Documents/ChatGPT/CWP/experiments/step8_work_transfer_20260919/regression_h280_beam_seed0_300s/records.json)

## 逐项验收矩阵

| 计划要求 | 状态 | 证据/说明 |
|---|---|---|
| 新分支、保留历史实验、独立输出目录 | 完成 | 本分支与 `experiments/step8_work_transfer_20260919/`；未使用 `git add .`、未删除旧目录 |
| 真实工作/位置联合事务 | 完成 | `cwp_solver.py` 显式 `work_plan` 与事务解码器；49 个测试 |
| 最多 3 吊、2 段、每段 ≤8 槽、外部冻结 | 完成 | `_normalize_work_transfer_regions`、槽字段逐项冻结测试 |
| 首尾残余交换 | 小例完成，H209未改善 | residual 小例消除一次回访；正式 H209 未发现可接受事务 |
| 相邻吊接力/有界多步接力 | 小例完成，H209未改善 | Q1→Q2 小例通过；H209 所有正式事务均未接受 |
| source hash / 当前源绑定 | 完成 | 哈希字段 + 完整轨迹签名 + 错配拒绝测试 |
| H208 固定修复 | 完成但无改善 | v5、v3 两个固定源均独立验证，均 0 接受 |
| H209 trajectory/beam、seed1/2 与三消融 | 完成 | 本报告 H209 章节和对应 `records.json` |
| H280 两模式回归 | 完成并审计 | 结果合法，但降 H 旧路径范围不满足新严格契约，已单列 |
| 非零 `move_time` | 明确不支持新事务 | 严格新事务只接受 0；非零输入显式返回/分派到旧路径，未静默当 0 |
| H209 达到 H≤208 且改善 Q4/Q5 | 未完成 | 本轮所有正式 H209 结果首次可行 H 都是 209 |

本报告没有把“负载偏差变小”“公开校验通过”写成“找到最短排工”。正式结论以四目标、连续性键、逐吊 idle 明细和独立校验共同决定。
