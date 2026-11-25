# SenseWhenNeeded 模块集成指南（core-motion-data-collector）

本目录包含 **已训练好的三个模型**（WISDM 腕部 IMU，5 类活动：walking / jogging / stairs / sitting / standing）以及推理所需代码。

**App 端的 Swift 集成代码已经写好**（`Core Motion Data Collector iOS/` 目录）：

| 文件 | 作用 |
|---|---|
| `SWNFeatureExtractor.swift` | 逐行复现 `swn/features.py`：59 维窗口特征 + 突发统计 + 事件特征 + 规则 z 分支 |
| `SWNRecognizer.swift` | 20 Hz 独立采集流 + 事件驱动策略（每 1s 突发分类 / 每 3s 主分类 / 每 6s 事件检测）+ Core ML 预测封装 |
| `SWNActivityCard.swift` | 主界面识别卡片：开关、当前活动、置信度、事件状态、"自动写入日记"开关 |

策略常量与训练时一致：RF 事件阈值 0.50、规则 z 阈值 4.0、置信提交 0.85/0.60、事件触发后立即重新分类并重建参考历史。突发历史按每 6 s 间隔采样（与检测器训练分布一致）。

## 两套模型，选一套转换

| 目录 | 说明 | 何时用 |
|---|---|---|
| `artifacts/models/` | 原始 WISDM 腕表模型 | 默认基线 |
| `artifacts/models_personalized/` | **个性化模型**：WISDM + 真机录音（35–95s 下楼 / 95–180s 走平路 / 180s 后静坐）混合重训 | 手机测试推荐用这套 |

个性化模型在真机录音上 macroF1 0.462 → 1.000，WISDM 测试受试者 0.801 → 0.827（无回退），详见 `artifacts/models_personalized/personalization_report.json`。注意它只对"你的设备位置/你的步态"有效。

## 离线处理脚本（Windows/Mac 都能跑）

```bash
# 1) 对任意一次 Sensor Logger 导出的 Motion.csv 做活动状态划分
python scripts/process_recording.py /path/to/Motion.csv \
    --models-dir artifacts/models_personalized \
    --out-dir artifacts/timeline_personalized

# 2) 采集新的带真值标注数据后，重新个性化微调（见 scripts/personalize.py 文档字符串）
python scripts/personalize.py NewRecording.csv --sessions-npz <SenseWhenNeeded>/data/processed/sessions.npz
```

输出：`timeline.csv`（逐秒决策）、`segments.csv`（分段表）、`summary.json`、`timeline.html`（可视化）。

## 你唯一还需要做的事：转 Core ML 并放进工程（Mac 上一次）

App 里没有模型时功能保持可用但惰性（卡片会显示缺什么）。要让它真正识别：

```bash
cd swn-models-for-macos
pip install coremltools "scikit-learn<=1.5.1" joblib numpy
# 用个性化模型转换（把 --models 指向 artifacts/models 则用原始模型）
python swn/coreml_export.py --models-dir artifacts/models_personalized
```

生成 `artifacts/coreml/` 下三个 `.mlmodel`，把这三个文件**拖进 Xcode 工程的 "Core Motion Data Collector iOS" 目录**（文件夹同步模式会自动纳入 target 并编译为 `.mlmodelc`），重新构建即可：

- `ActivityClassifier.mlmodel` — 3s 窗、5 类（每 3s 确认）
- `BurstClassifier.mlmodel` — 1s 窗、5 类（每 1s 快速估计）
- `EventDetector.mlmodel` — 突发历史 → change/no_change（每 6s）

## 验证特征对齐（建议做一次）

Swift 端特征必须与 Python 完全一致。在 Mac 上（swn-models-for-macos 目录）跑：

```python
import numpy as np, sys; sys.path.insert(0, ".")
from swn.features import window_features
t = np.arange(20) / 20.0
acc = np.stack([0.3*np.sin(2*np.pi*1.5*t), 9.81+0.2*np.cos(2*np.pi*1.5*t), 0.1*np.sin(2*np.pi*3.0*t)], 1)
gyro = np.stack([0.05*np.sin(2*np.pi*t), np.full(20, 0.02), 0.04*np.cos(2*np.pi*t)], 1)
print(window_features(acc, gyro, 20.0)[0])
```

与 Swift 端 `SWNFeatureExtractor.selfTestWindow()` 构造的同一窗口喂给 `windowFeatures` 的输出逐项对比（误差应在 1e-6 量级）。

## 使用方式

1. App 主界面出现"活动识别（Sense When Needed）"卡片，打开开关即开始；
2. 识别使用独立的 20 Hz 传感器流，与"Collect Motion Data"采集互不影响；
3. 打开"活动变化自动写入日记"后，每次确认活动变化会自动在 Daily Log 打点（与 CSV 导出的时间戳同一时钟，可直接对齐分析）；
4. **"导入 CSV 并分析"是一体化流程**：选中 Sensor Logger 导出的 Motion.csv 后，App 会立即自动回放事件驱动策略（单位换算 g→m/s²、100Hz→20Hz 重采样、1s 突发分类 + 3s 主分类 + 6s 事件检测，与 `scripts/process_recording.py` 完全一致），直接展示活动时间线、分段结果和各活动占比——不需要单独的分析步骤。普通 CSV 则回退为表格预览。对应实现：`SWNOfflineAnalyzer.swift` + `CSVImportView.swift`。

## 诚实提醒

两个分类器是 300 棵深树的 RandomForest，Core ML 转换后单个模型约 20–50 MB。嫌大可以：只用 BurstClassifier + EventDetector（改 `SWNRecognizer.loadModels`），或回 Python 减 `n_estimators` 重训。
