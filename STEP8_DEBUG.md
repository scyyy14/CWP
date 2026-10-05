# 第八步独立调试入口

导师拉取本分支后，可以在 PyCharm 中直接运行 `tools/run_step8_debug.py`。这个入口只执行最新版第八步邻域搜索，不重新执行前七步。

## PyCharm 直接运行

新建一个 Python Run/Debug Configuration：

- Script path：`tools/run_step8_debug.py`
- Parameters：留空
- Working directory：项目根目录 `CWP`
- Python interpreter：项目虚拟环境中的 Python 3.10+

留空参数时默认使用仓库内的 H208 原排工，搜索 30 秒，并将结果写入 `experiments/step8_debug/`。主要结果位于：

```text
experiments/step8_debug/schedule_artifacts/fixed/seed_0/budget_30s/trajectory/
```

其中 `source.json` 是原排工，`execution_best.json` 是以正式目标 `(全部工作实际完工时间, 移动次数)` 选择的执行方案，`recommended_best.json` 是兼顾完工同步性的推荐方案。

## 在 PyCharm 中改变参数

把下面内容填入 Run/Debug Configuration 的 Parameters：

```text
--case h280 --budget 60 --seed 1
```

可调参数：

- `--case h208|h280`：选择仓库自带的原排工；
- `--budget 秒数`：第八步搜索时间；
- `--seed 整数`：随机种子；
- `--target-mode shorten|same_horizon|auto`：缩短完工时间、固定工期优化或自动选择；
- `--local-state-limit 整数`：局部循环交换的状态上限；
- `--out 目录`：输出目录。

测试自己的原方案时，同时给出实例输入和原排工 JSON：

```text
--input D:\data\instance.json --source D:\data\source_schedule.json --budget 60
```

`--source` 必须是包含完整 `slots` 的排工 JSON，不是排工图片；它必须与 `--input` 中的 `W`、`M`、`S` 和 `move_time` 相匹配。

## 命令行运行

Windows：

```powershell
.venv\Scripts\python.exe tools\run_step8_debug.py
.venv\Scripts\python.exe tools\run_step8_debug.py --case h280 --budget 60 --seed 1
```

macOS / Linux：

```bash
.venv/bin/python tools/run_step8_debug.py
.venv/bin/python tools/run_step8_debug.py --case h280 --budget 60 --seed 1
```

绘制某个结果：

```bash
python tools/render_reference_schedule.py path/to/execution_best.json path/to/execution_best.svg
```

如需逐项开关第八步内部算子，可直接运行 `tools/evaluate_critical_step.py --help`。一键入口固定启用当前正式配置：仅局部窗口、禁止全局重构，并开启碎片修复、循环交换、阶段重排、阶段闭合、跨桥吊接力、强制前缀合并与空闲能力再平衡。
