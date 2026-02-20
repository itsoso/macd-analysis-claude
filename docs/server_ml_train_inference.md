# 服务器上 ML 训练与推理使用指南

基于当前代码，说明如何在**生产服务器**上充分利用训练代码，并确保推理使用最新模型。

---

## 一、推理用哪些文件（确保推理可用）

推理由 `ml_live_integration.py` 的 `MLSignalEnhancer.load_model()` 完成，从 **项目根目录下的 `data/ml_models/`** 按需加载：

| 用途 | 文件 | 说明 |
|------|------|------|
| 方向预测 (优先) | `stacking_meta.pkl` + `stacking_meta.json` | Stacking 元学习器；有则优先用 Stacking |
| 方向预测 (基模型) | `stacking_lgb_1h.txt` / `stacking_xgb_1h.json` / `stacking_lstm_1h.pt` / `stacking_tft_1h.pt` | Stacking 四基模型（由 stacking_meta.json 指名） |
| 方向预测 (回退) | `lgb_direction_model.txt` + `.meta.json` | LGB 方向；无 Stacking 时与 LSTM/TFT/跨资产加权 |
| | `lstm_1h.pt`（+ ONNX 可选） | LSTM+Attention |
| | `tft_1h.pt` + `tft_1h.meta.json` | TFT（meta 含 feat_mean/feat_std 做标准化） |
| | `lgb_cross_asset_1h.txt` + `.meta.json` | 跨资产 LGB |
| 集成配置 | `ensemble_config.json` | 权重、阈值、可选 feat_mean/feat_std |
| Regime | `vol_regime_model.txt` + `regime_config.json` 等 | RegimePredictor 所需 |
| 分位数 | `quantile_config.json` + `quantile_h5_q*.txt` / `quantile_h12_q*.txt` | QuantilePredictor 所需 |
| 可选 | `mtf_fusion_mlp.pt`、`ppo_position_agent.zip` 等 | 神经融合、PPO 等 |

**要点**：

- 所有路径相对于**进程当前工作目录**（Web 为 `/opt/macd-analysis`，即 `data/ml_models` = `/opt/macd-analysis/data/ml_models`）。
- 模型是在**进程启动时**加载的（Gunicorn worker、live_runner 启动时各加载一次），**更新模型后必须重启服务**才会用上新文件。

**确保推理可用**：

1. 保证上述需要的文件在 `data/ml_models/` 存在且可读。
2. 更新/新增模型后执行：  
   `systemctl restart macd-analysis`  
   （交易引擎会因 PartOf 自动重启，并重新加载 ML 模型。）

---

## 二、训练代码如何在服务器上“充分利用”

`train_gpu.py` **不依赖 Binance API**，只读本地 Parquet。因此可以在服务器上运行，与 H800 使用同一套代码和目录约定。

### 2.1 数据从哪来

训练需要目录与 H800 一致：

- `data/klines/{SYMBOL}/{interval}.parquet`
- `data/funding_rates/{SYMBOL}_funding.parquet`
- `data/open_interest/{SYMBOL}/{period}.parquet`
- `data/mark_klines/{SYMBOL}/{interval}.parquet`（可选但推荐）

**在服务器上**（可访问 Binance）：

- 本机拉取并写入上述目录：
  - 全量：`python3 fetch_5year_data.py`（约 15–30 分钟）
  - 或只用你已有的脚本/定时任务，按相同路径写入 Parquet
- 或从本机/H800 用 rsync/scp 同步已有 `data/klines`、`data/funding_rates`、`data/open_interest`、`data/mark_klines` 到服务器对应目录。

只要目录和文件名与 `train_gpu.py` 中 `load_klines_local` / `load_funding_local` / `load_oi_local` / `load_mark_local` 一致即可。

### 2.2 在服务器上跑哪些训练模式

- **有 GPU**：可跑全部 12 种模式（与 H800 一致），输出都在 `data/ml_models/`（及 `data/gpu_results/`）。
- **仅 CPU**（常见阿里云 ECS）：
  - **推荐**：`python3 train_gpu.py --mode lgb [--tf 1h]`  
    LightGBM 在 CPU 上可接受，产出 `lgb_direction_model.txt` 等，推理会用作方向预测（或 Stacking 基模型之一）。
  - **可选**：`--mode optuna`、`--mode cross_asset`（LGB 系）、`--mode incr_wf`（增量 Walk-Forward）等，按需跑。
  - LSTM/TFT/Stacking/MTF 等 PyTorch 模式在 CPU 上会**自动回退到 CPU**（`device='cuda' if torch.cuda.is_available() else 'cpu'`），能跑但较慢，仅建议在确有需要或测试时在服务器跑。

训练输出统一写到：

- `data/ml_models/`：所有推理用到的 `.txt`、`.pt`、`.pkl`、`.json`、`.onnx` 等。
- `data/gpu_results/`：结果与日志（可选）。

因此，**同一套训练代码在服务器上跑出来的模型，会直接落在推理读取的目录里**。

### 2.3 训练后如何“确保推理用上”

1. 训练结束：检查 `data/ml_models/` 下是否生成了你期望的文件（如 `lgb_direction_model.txt`、`stacking_meta.pkl` 等）。
2. 重启服务，让推理重新加载模型：
   - `systemctl restart macd-analysis`  
     会连带重启 `macd-engine`（PartOf），两边都会重新执行 `load_model()`，读到新文件。
3. 可选：在控制面板或 `/api/ml/status` 查看当前加载的模型列表，确认新模型已被加载。

---

## 三、推荐用法总结

| 场景 | 做法 | 推理如何用上 |
|------|------|----------------|
| **仅用 H800 训练** | H800 跑 `train_gpu.py`，模型通过 git 或 scp/rsync 回传服务器 `data/ml_models/`，再 `git pull` 或同步后重启服务 | 重启后 `load_model()` 读新文件 |
| **服务器轻量训练** | 服务器上先拉数据（`fetch_5year_data.py` 或同步），再跑 `train_gpu.py --mode lgb`（及可选 optuna/cross_asset/incr_wf），产出已在 `data/ml_models/` | 同一目录，重启即用 |
| **混合** | 大模型（LSTM/TFT/Stacking）在 H800 训，小模型（LGB/Regime/Quantile）在服务器用 `train_gpu.py` 做增量/重训 | 所有产出都在 `data/ml_models/`，一次重启统一生效 |

**“充分利用训练代码”**：在服务器上使用同一套 `train_gpu.py` + 相同数据目录约定，按机器能力选择模式（CPU 优先 lgb/optuna/cross_asset/incr_wf），产出与 H800 一致，推理无需改代码即可用。

**“确保推理也可以用”**：  
- 所有推理用到的模型和配置都放在 **`data/ml_models/`**；  
- 更新该目录后 **重启 Web + 引擎**（`systemctl restart macd-analysis`），推理即使用最新模型。

---

## 四、服务器上可执行的命令示例

```bash
# 进入项目目录（与 run 时一致）
cd /opt/macd-analysis
source venv/bin/activate   # 或 . venv/bin/activate

# 1) 数据准备（若尚未有完整数据）
python3 fetch_5year_data.py

# 2) 仅 CPU 时推荐：只训 LGB 方向模型
python3 train_gpu.py --mode lgb --tf 1h

# 3) 可选：跨资产 LGB、或增量 WF
# python3 train_gpu.py --mode cross_asset --tf 1h
# python3 train_gpu.py --mode incr_wf --tf 1h

# 4) 更新模型后重启，使推理加载新模型
systemctl restart macd-analysis
```

如需定时在服务器上重训（例如每日 LGB），可用 cron 调用上述 2）+ 4），并注意并发与磁盘空间。
