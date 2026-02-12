# CLAUDE.md — 项目上下文与部署指南

## 项目概述

**MACD Analysis 六书融合策略平台** — 基于 ETH/USDT 的量化交易分析系统。

融合 7 本技术分析经典著作的策略：背离分析、均线、蜡烛图、布林带、量价分析、KDJ、海龟交易法则。提供回测优化、实盘信号检测、多周期联合决策等功能。

## 技术栈

- **后端**: Python 3 + Flask
- **前端**: Jinja2 模板 + 原生 JS（无框架）
- **数据**: Binance API (ETH/USDT K线)
- **部署**: Gunicorn + Nginx + systemd
- **服务器**: 阿里云 47.236.39.215:22222
- **域名**: https://invest.executor.life

## 关键文件

| 文件 | 作用 |
|------|------|
| `app.py` | Flask 应用主文件，所有路由和 API |
| `optimize_six_book.py` | 六书融合回测优化器 + 多周期联合决策引擎 |
| `live_runner.py` | 实盘交易入口 + 多周期信号检测 + 智能共识算法 |
| `live_signal_generator.py` | 实盘信号生成器 |
| `live_config.py` | 实盘配置 (StrategyConfig) |
| `strategy_futures.py` | 合约交易引擎 (FuturesEngine) |
| `binance_fetcher.py` | 币安 K线数据获取 |
| `indicators.py` | 技术指标计算 |
| `ma_indicators.py` | 均线策略 |
| `candlestick_patterns.py` | 蜡烛图策略 |
| `bollinger_strategy.py` | 布林带策略 |
| `volume_price_strategy.py` | 量价策略 |
| `kdj_strategy.py` | KDJ 策略 |
| `turtle_strategy.py` | 海龟交易策略 |
| `gunicorn_config.py` | Gunicorn 生产配置 |
| `deploy.sh` | 部署脚本 |

## 服务器信息

```
IP:        47.236.39.215
SSH端口:   22222
用户:      root
项目路径:  /opt/macd-analysis
虚拟环境:  /opt/macd-analysis/venv
服务名:    macd-analysis (systemd)
应用端口:  5100 (gunicorn → nginx 反代)
域名:      invest.executor.life
```

## 部署方法

### 方式一: 使用 deploy.sh (推荐)

```bash
# 首次完整部署
./deploy.sh

# 日常更新代码并重启
./deploy.sh update

# 仅重启服务
./deploy.sh restart

# 查看服务状态
./deploy.sh status

# 查看运行日志
./deploy.sh logs

# 配置 SSL 证书
./deploy.sh ssl
```

### 方式二: 手动 SSH 部署

```bash
# 1. 本地: 提交并推送代码
git add -A && git commit -m "your message" && git push origin main

# 2. SSH 登录服务器
ssh -p 22222 root@47.236.39.215

# 3. 服务器上: 拉取代码并重启
cd /opt/macd-analysis
git pull origin main
systemctl restart macd-analysis

# 4. 验证服务状态
systemctl status macd-analysis
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5100/
```

### 方式三: 一行命令远程部署

```bash
ssh -p 22222 root@47.236.39.215 "cd /opt/macd-analysis && git pull origin main && systemctl restart macd-analysis"
```

## 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python app.py
# 访问 http://localhost:5000

# 运行回测优化 (耗时约6分钟)
python optimize_six_book.py

# 测试实盘信号 (单周期)
python live_runner.py --test-signal --tf 1h

# 测试多周期信号
python live_runner.py --test-signal-multi --tfs 15m,1h,4h,8h,24h
```

## 超时配置

多周期信号检测耗时较长，三层超时需一致：

| 层级 | 配置位置 | 当前值 |
|------|---------|--------|
| Flask subprocess | `app.py` → `/api/live/test_signal_multi` | 300s |
| Gunicorn worker | `gunicorn_config.py` → `timeout` | 360s |
| Nginx proxy | `deploy.sh` → `proxy_read_timeout` | 360s |

## 多周期智能共识算法

`live_runner.py` 中的 `compute_weighted_consensus()`:

- **加权得分**: 大周期权重高 (24h=28, 4h=15, 1h=8, 15m=3)
- **共振链检测**: 相邻TF连续同向 ≥3级且含4h+ → 强信号
- **大周期定调**: ≥4h 的周期作为趋势基调
- **决策矩阵**: 大小同向→入场, 反向→不做, 分歧→观望

## 注意事项

- 登录凭据在 `app.py` 的 `USERS` 字典中
- 回测结果保存在 `optimize_six_book_result.json`
- 实盘日志在服务器 `/opt/macd-analysis/logs/`
- 前端状态持久化使用 `localStorage` (TF选择、检测结果、Tab状态)
- SSH 可能因网络波动超时，若连接失败可通过阿里云控制台 VNC 登录
