# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**MACD Analysis 六书融合策略平台** — 基于 ETH/USDT 的量化交易分析系统。

融合 7 本技术分析经典著作的策略：背离分析、均线、蜡烛图、布林带、量价分析、KDJ、海龟交易法则。提供回测优化、实盘信号检测、多周期联合决策等功能。支持 GPU 离线训练 (H800)。

## 技术栈

- **后端**: Python 3 + Flask
- **前端**: Jinja2 模板 + 原生 JS（无框架）
- **数据**: Binance API (ETH/USDT K线)，本地 Parquet 缓存 (`data/klines/`)
- **ML**: LightGBM + XGBoost (CPU/GPU), PyTorch LSTM (GPU)
- **GPU 训练**: H800 离线训练 (PyTorch + LightGBM CUDA + Optuna)
- **部署**: Gunicorn + Nginx + systemd
- **服务器**: 阿里云 47.237.191.17:22222
- **域名**: https://invest.executor.life
- **GitHub**: git@github.com:itsoso/macd-analysis-claude.git

当修改跨 Python 后端和前端模板时，确保两侧都已更新。服务器上 GitHub 不可用时使用 SCP 部署。

## 系统架构总览

系统由三大子系统组成，运行在三台不同机器上：

```
┌─────────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│  本机 (开发机)        │   │  H800 GPU 训练机  │   │  阿里云 (生产服务器)  │
│  macOS               │   │  内网，需跳板机    │   │  47.237.191.17      │
│                      │   │                  │   │                     │
│  • 代码开发           │   │  • 模型训练       │   │  • Flask Web 应用    │
│  • Binance 数据拉取   │   │  • 超参优化       │   │  • 实盘信号检测      │
│  • 回测分析           │   │  • 深度学习       │   │  • 交易执行          │
│  • 数据打包传输 →     │──→│  ← 模型回传       │──→│  • 模型推理 (CPU)    │
└─────────────────────┘   └──────────────────┘   └─────────────────────┘
```

## 本地开发

```bash
pip install -r requirements.txt

# 启动开发服务器 → http://localhost:5000
python app.py

# 运行回测优化 (耗时约6分钟)
python optimize_six_book.py

# 测试实盘信号 (单周期)
python live_runner.py --test-signal --tf 1h

# 测试多周期信号
python live_runner.py --test-signal-multi --tfs 15m,1h,4h,8h,24h

# 运行测试
pytest test_core.py
pytest test_live_system.py
pytest test_backtest_realism.py
# 单个测试
pytest test_core.py::test_signal_consistency -v
```

## 核心架构

### 信号计算管线 (关键路径)

```
binance_fetcher.py (K线 + 永续合约数据)
    ↓
indicators.py (MACD/KDJ/RSI/CCI) + ma_indicators.py (均线)
    ↓
signal_core.py → compute_signals_six()  ← 回测和实盘的共享核心，防止信号漂移
    ├── 背离: indicators.py → analyze_signals_enhanced()
    ├── 均线: ma_indicators.py → compute_ma_signals()
    ├── 蜡烛图: candlestick_patterns.py → compute_candlestick_scores()
    ├── 布林带: bollinger_strategy.py → compute_bollinger_scores()
    ├── 量价: volume_price_strategy.py → compute_volume_price_scores()
    └── KDJ: kdj_strategy.py → compute_kdj_scores()
    ↓
signal_core.py → calc_fusion_score_six()  (六书融合评分)
    ↓
multi_tf_consensus.py → fuse_tf_scores()  (多周期共识，可选)
    ↓
ml_live_integration.py → MLSignalEnhancer  (ML 增强，可选)
```

**`signal_core.py` 是最关键文件** — 回测 (`optimize_six_book.py`) 和实盘 (`live_signal_generator.py`) 共用同一份信号计算代码。修改此文件时务必同时验证回测和实盘行为。

### 信号计算的三种模式

- **标准**: `compute_signals_six()` — 实盘使用，可读性优先
- **向量化**: `compute_signals_six_fast()` → `signal_vectorized.py` — 回测用，10x 加速
- **多进程**: `compute_signals_six_multiprocess()` — 大规模回测用，fork + COW

### ML 预测子系统 (8 模型, H800 训练)

```
ml_features.py (94 维特征: 73基础 + 21跨资产 BTC/SOL/BNB)
    ↓
┌─ LGB 方向预测 (lgb_direction_model.txt, Optuna, AUC 0.55)
├─ LSTM+Attention (lstm_1h.pt + ONNX, BF16, AUC 0.54)
├─ 跨资产 LGB (lgb_cross_asset_1h.txt, 94维, AUC 0.55)
├─ TFT (tft_1h.pt + ONNX, 148K参数)
├─ MTF 融合 MLP (mtf_fusion_mlp.pt + ONNX, 多周期分数融合, AUC 0.56)
├─ Regime 分类 (vol_regime + trend, vol AUC 0.59)
├─ 分位数回归 (h5+h12, q05~q95, Kelly 仓位 + 动态止损)
└─ PPO 仓位 (ppo_position_agent.zip, 实验性)
    ↓
ml_live_integration.py → MLSignalEnhancer (五层: 方向→融合→Regime→分位数→执行)
    实盘集成: LGB+LSTM 加权 → bull_prob → Kelly 仓位 + 动态止损
    当前状态: **shadow 模式** (只记录不修改信号)
```

### 回测引擎

`strategy_futures.py` (`FuturesEngine`) 模拟币安 USDT 永续合约：
- 保证金计算、强平检查、资金费率 (8h)、滑点、手续费
- 部分止盈状态机: TP1 @ +20% 平 30% → TP2 @ +40% 平 30% → 追踪止损
- 保护机制: 日亏限额 (-3%)、全局回撤暂停 (-15%, 可恢复)、连亏冷却

### 实盘信号流

```
live_runner.py (CLI 入口) → live_signal_generator.py (信号生成)
    → live_trading_engine.py (交易编排)
        → order_manager.py (下单) + risk_manager.py (风控) + notifier.py (通知)
```

### Web 应用路由

`app.py` 通过 `web_routes/pages.py` 和 `web_routes/result_apis.py` 模块化注册路由。结果 API 自动映射: `/api/{strategy_name}` → `{STRATEGY_NAME}_result.json`。

### 配置层级

```
config.py (全局指标参数)
  → live_config.py (实盘策略配置, StrategyConfig, 含版本管理 v1-v5)
    → optimize_six_book_result.json (回测优化最佳参数)
      → 命令行覆盖 (--capital, --leverage 等)
```

当前生产版本: `live_config.py` 中 `_ACTIVE_VERSION = "v5"`

## GPU 训练架构 (H800)

### 数据流: 本机 → H800

H800 位于办公内网，需要跳板机访问，**无法直接连接 Binance API**。数据通过离线打包传输。

```
本机 (可访问 Binance)                          H800 GPU (内网)
┌──────────────────────────┐                  ┌──────────────────────────┐
│ fetch_5year_data.py       │                  │ verify_data.py            │
│   → data/klines/         │   tar.gz + SCP    │   → 数据完整性检查        │
│   → data/mark_klines/    │  ──────────────→  │                          │
│   → data/funding_rates/  │   (~62MB)         │ train_gpu.py              │
│   → data/open_interest/  │                   │   --mode lgb/lstm/optuna  │
│                          │                   │   --mode tft/cross_asset  │
│ pack_for_h800.sh         │                   │   --mode mtf_fusion/ppo   │
│   → macd_train_data.tar.gz│                  │   --mode onnx/retrain     │
│                          │   模型回传          │                          │
│ data/ml_models/ ←        │  ←──────────────  │ data/ml_models/           │
│                          │   (~几MB)          │ data/gpu_results/         │
└──────────────────────────┘                  └──────────────────────────┘
```

### 训练数据清单

| 数据类型 | 目录 | 内容 | 覆盖范围 |
|---------|------|------|---------|
| K线 (OHLCV) | `data/klines/{SYMBOL}/{interval}.parquet` | open/high/low/close/volume/quote_volume/taker_buy_* | 5年, 4交易对×4周期 |
| Mark Price | `data/mark_klines/ETHUSDT/{interval}.parquet` | mark_open/high/low/close | 5年, 多周期 |
| Funding Rate | `data/funding_rates/{SYMBOL}_funding.parquet` | funding_rate, mark_price, interval_hours | 5年, 4交易对 |
| Open Interest | `data/open_interest/{SYMBOL}/{period}.parquet` | open_interest, open_interest_value | ~30天 (API 限制) |

交易对: ETHUSDT (主力), BTCUSDT, SOLUSDT, BNBUSDT
周期: 15m, 1h, 4h, 24h

### GPU 训练模式

`train_gpu.py` 提供 11 种训练模式，全部离线运行不依赖 Binance API：

```bash
# 基础模型
python3 train_gpu.py --mode lgb --tf 1h,4h       # LightGBM Walk-Forward
python3 train_gpu.py --mode lstm --tf 1h          # LSTM+Attention (BF16)
python3 train_gpu.py --mode optuna --trials 500   # Optuna 超参搜索
python3 train_gpu.py --mode backtest --trials 1000 # 回测参数优化

# 高级模型 (v2)
python3 train_gpu.py --mode tft --tf 1h           # TFT (Temporal Fusion Transformer)
python3 train_gpu.py --mode cross_asset --tf 1h   # 跨资产 LGB (BTC/SOL/BNB)
python3 train_gpu.py --mode mtf_fusion            # 多周期融合 MLP
python3 train_gpu.py --mode ppo                   # PPO 仓位优化 (强化学习)
python3 train_gpu.py --mode onnx                  # ONNX 导出 (加速推理)
python3 train_gpu.py --mode incr_wf               # 增量 Walk-Forward
python3 train_gpu.py --mode retrain               # 定时重训

# 批量
python3 train_gpu.py --mode all      # 基础全流程
python3 train_gpu.py --mode all_v2   # 基础 + TFT + 跨资产
python3 train_gpu.py --mode all_v3   # v2 + MTF融合 + PPO
```

数据加载管线 (纯本地):
```
load_klines_local()  → 本地 Parquet, 不触发 API
    ↓
indicators.py + ma_indicators.py  → 技术指标
    ↓
ml_features.py → compute_ml_features()  → 70+ 维特征
    ↓
标签生成: 多尺度利润化标签 (3h/5h/12h/24h)
    ↓
训练: LightGBM GPU / LSTM+Attention / Optuna TPE
    ↓
输出: data/ml_models/*.pt, data/gpu_results/*.json
```

### H800 操作流程

```bash
# 1. 本机: 拉取数据 + 打包
python3 fetch_5year_data.py           # 拉取全量数据 (~15分钟)
./pack_for_h800.sh                    # 打包 (~62MB)

# 2. 传输到 H800 (通过跳板机)
scp -J user@jumphost macd_train_data.tar.gz user@h800:~/work/

# 3. H800: 解压 + 环境搭建
tar -xzf macd_train_data.tar.gz
./setup_h800.sh                       # 自动: GPU检测 + conda + 依赖安装

# 4. H800: 验证 + 训练
python3 verify_data.py                # 数据完整性检查
python3 train_gpu.py --mode lgb       # 开始训练

# 5. 回传模型
tar -czf macd_models.tar.gz data/ml_models/ data/gpu_results/
scp -J jumphost macd_models.tar.gz user@dev:~/macd-analysis/
```

### GPU 相关文件

| 文件 | 作用 |
|------|------|
| `fetch_5year_data.py` | 批量下载 5 年训练数据 (K线+Mark+Funding+OI) |
| `pack_for_h800.sh` | 打包数据+代码，检查完整性，生成传输包 |
| `setup_h800.sh` | H800 一键环境搭建 (GPU 检测, conda, 依赖安装) |
| `requirements-gpu.txt` | GPU 训练依赖 (PyTorch CUDA, Optuna, TensorBoard) |
| `verify_data.py` | 训练数据完整性验证 (必需/可选数据检查) |
| `train_gpu.py` | **GPU 离线训练入口** (11 种模式, 不依赖 Binance API) |

## 关键文件

| 文件 | 作用 |
|------|------|
| `app.py` | Flask 应用主文件，所有路由和 API |
| `signal_core.py` | **统一信号计算引擎** (回测/实盘共用) |
| `optimize_six_book.py` | 六书融合回测优化器 + 多周期联合决策引擎 |
| `live_runner.py` | 实盘交易入口 + 多周期信号检测 + 智能共识算法 |
| `live_signal_generator.py` | 实盘信号生成器 |
| `live_config.py` | 实盘配置 (StrategyConfig)，含 v1-v5 版本管理 |
| `strategy_futures.py` | 合约交易引擎 (FuturesEngine) |
| `binance_fetcher.py` | 币安 K线 + 永续合约数据获取 (含 Parquet 缓存, OI 分段拉取) |
| `multi_tf_consensus.py` | 多周期加权共识算法 |
| `signal_vectorized.py` | P0 向量化信号计算 (性能优化) |
| `ml_features.py` | ML 特征工程 (70+ 维: 动量/趋势/振荡/波动/量价/微结构/时间) |
| `ml_predictor.py` | LightGBM 方向预测 + Walk-Forward + 多尺度集成 + Stacking |
| `ml_regime.py` | 市场 Regime 预测 (波动率聚类 + 趋势质量) |
| `ml_quantile.py` | 分位数回归 (收益分布预测) |
| `ml_live_integration.py` | ML 实盘集成 (五层: 方向→融合→Regime→分位数→执行, shadow模式) |
| `ml_strategy.py` | ML 独立策略回测 / 融合模式 |
| `walk_forward_pipeline.py` | Walk-Forward 月度滚动验证管道 |
| `train_gpu.py` | **H800 GPU 离线训练入口** (11种模式: LGB/LSTM/TFT/跨资产/MTF/PPO/ONNX等) |
| `gunicorn_config.py` | Gunicorn 生产配置 |
| `deploy.sh` | 自动化部署脚本 |

## 部署 & 服务

```
IP:          47.237.191.17
SSH端口:     22222
用户:        root
项目路径:    /opt/macd-analysis
虚拟环境:    /opt/macd-analysis/venv
服务名:      macd-analysis (systemd)
应用端口:    5100 (gunicorn → nginx 反代)
域名:        invest.executor.life
```

**修改后端代码后，必须重启相关服务才能生效。不要只编辑文件就报告完成。**

### 部署命令

```bash
# 推荐: 自动化部署
./deploy.sh update    # 日常更新 (git pull → pip install → restart → 健康检查)
./deploy.sh restart   # 仅重启
./deploy.sh status    # 查看状态
./deploy.sh logs      # 查看日志
./deploy.sh           # 首次完整部署 (含 venv/systemd/nginx)
./deploy.sh ssl       # 配置 SSL 证书

# 手动一行部署
ssh -p 22222 root@47.237.191.17 "cd /opt/macd-analysis && git pull origin main && systemctl restart macd-analysis"

# VNC 备选 (SSH 不通时): 阿里云控制台 → ECS → 远程连接 → VNC
```

### 超时配置 (三层必须一致)

| 层级 | 配置位置 | 当前值 |
|------|---------|--------|
| Flask subprocess | `app.py` → `/api/live/test_signal_multi` | 300s |
| Gunicorn worker | `gunicorn_config.py` → `timeout` | 360s |
| Nginx proxy | `deploy.sh` → `proxy_read_timeout` | 360s |

## 远程服务器访问

- **不要尝试交互式密码 SSH 连接** — 在此环境中无法工作。使用密钥认证或让用户提供替代访问方式。
- SSH 已知问题: "Connection timed out during banner exchange"
  - `deploy.sh` 已添加 `-o IdentityAgent=none -o ProxyCommand=none`
  - 如超时，尝试关闭 VPN 或使用 VNC 方式部署

## 调试 & Bug 修复

- **修复 bug 后必须验证**: 检查日志或测试端点，确认修复已实际生效。不要编辑后就假定成功。如果 commit 已包含该变更，先确认再报告。
- **系统性排查根因**: 尤其是定时任务、配置问题时，先检查特定选择器、配置文件和持久化机制，不要先用通用方案试错。
- **信号一致性**: 修改策略逻辑后，用 `pytest test_core.py` 验证回测和实盘信号是否一致。
- **GPU 训练调试**: `train_gpu.py` 完全不依赖 Binance API。如数据缺失先在本机运行 `fetch_5year_data.py` 补拉后重新打包传输。

## 多周期智能共识算法

`multi_tf_consensus.py` 中的 `fuse_tf_scores()`:

- **加权得分**: 大周期权重高 (24h=28, 4h=15, 1h=8, 15m=3)
- **共振链检测**: 相邻TF连续同向 ≥3级且含4h+ → 强信号
- **大周期定调**: ≥4h 的周期作为趋势基调
- **决策矩阵**: 大小同向→入场, 反向→不做, 分歧→观望

## 数据存储

```
data/
├── klines/{SYMBOL}/{interval}.parquet    # K线缓存 (5年, 缓存优先, API回退)
├── mark_klines/{SYMBOL}/{interval}.parquet # Mark Price (永续标记价)
├── funding_rates/{SYMBOL}_funding.parquet  # 资金费率 (5年)
├── open_interest/{SYMBOL}/{period}.parquet # 持仓量 (~30天, API限制)
├── ml_models/                             # ML 模型文件 (8模型: .txt/.pt/.onnx/.meta.json/.zip)
├── gpu_results/                           # GPU 训练结果 (.json/.parquet)
├── backtests/*.db                         # SQLite (runs/daily_records/trades)
└── live/
    ├── trades_YYYYMMDD.jsonl              # 实盘日志 (JSONL, 崩溃安全)
    └── engine_state.json                  # 引擎状态
```

- **回测结果**: `optimize_six_book_result.json` (最佳参数 + 交易记录)
- **训练产出**: `data/ml_models/` (LightGBM .txt + PyTorch .pt + 元数据 .meta.json)

## 注意事项

- 登录凭据在 `app.py` 的 `USERS` 字典中
- 前端状态持久化使用 `localStorage` (TF选择、检测结果、Tab状态)
- 服务器性能较低，信号检测需要耐心等待
- Regime 感知: `optimize_six_book._compute_regime_controls()` 和 `live_signal_generator._infer_regime_label()` 根据市场状态动态调整参数
- Binance OI API 的 `startTime` 上限约 30 天，`binance_fetcher.py` 已实现分段拉取
- H800 训练完全离线，数据更新需在本机重新拉取后打包传输
