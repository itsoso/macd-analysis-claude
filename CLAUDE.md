# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**MACD Analysis 六书融合策略平台** — 基于 ETH/USDT 的量化交易分析系统。

融合 7 本技术分析经典著作的策略：背离分析、均线、蜡烛图、布林带、量价分析、KDJ、海龟交易法则。提供回测优化、实盘信号检测、多周期联合决策等功能。

## 技术栈

- **后端**: Python 3 + Flask
- **前端**: Jinja2 模板 + 原生 JS（无框架）
- **数据**: Binance API (ETH/USDT K线)，本地 Parquet 缓存 (`data/klines/`)
- **部署**: Gunicorn + Nginx + systemd
- **服务器**: 阿里云 47.237.191.17:22222
- **域名**: https://invest.executor.life
- **GitHub**: git@github.com:itsoso/macd-analysis-claude.git

当修改跨 Python 后端和前端模板时，确保两侧都已更新。服务器上 GitHub 不可用时使用 SCP 部署。

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
```

**`signal_core.py` 是最关键文件** — 回测 (`optimize_six_book.py`) 和实盘 (`live_signal_generator.py`) 共用同一份信号计算代码。修改此文件时务必同时验证回测和实盘行为。

### 信号计算的三种模式

- **标准**: `compute_signals_six()` — 实盘使用，可读性优先
- **向量化**: `compute_signals_six_fast()` → `signal_vectorized.py` — 回测用，10x 加速
- **多进程**: `compute_signals_six_multiprocess()` — 大规模回测用，fork + COW

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
| `binance_fetcher.py` | 币安 K线 + 永续合约数据获取 (含 Parquet 缓存) |
| `multi_tf_consensus.py` | 多周期加权共识算法 |
| `signal_vectorized.py` | P0 向量化信号计算 (性能优化) |
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

## 多周期智能共识算法

`multi_tf_consensus.py` 中的 `fuse_tf_scores()`:

- **加权得分**: 大周期权重高 (24h=28, 4h=15, 1h=8, 15m=3)
- **共振链检测**: 相邻TF连续同向 ≥3级且含4h+ → 强信号
- **大周期定调**: ≥4h 的周期作为趋势基调
- **决策矩阵**: 大小同向→入场, 反向→不做, 分歧→观望

## 数据存储

- **回测结果**: `optimize_six_book_result.json` (最佳参数 + 交易记录)
- **多运行回测**: `data/backtests/*.db` (SQLite, 含 runs/daily_records/trades 表)
- **实盘日志**: `data/live/trades_YYYYMMDD.jsonl` (JSONL 追加写入, 崩溃安全)
- **引擎状态**: `data/live/engine_state.json`
- **K线缓存**: `data/klines/ETHUSDT/{interval}.parquet` (缓存优先, API 回退)

## 注意事项

- 登录凭据在 `app.py` 的 `USERS` 字典中
- 前端状态持久化使用 `localStorage` (TF选择、检测结果、Tab状态)
- 服务器性能较低，信号检测需要耐心等待
- Regime 感知: `optimize_six_book._compute_regime_controls()` 和 `live_signal_generator._infer_regime_label()` 根据市场状态动态调整参数
