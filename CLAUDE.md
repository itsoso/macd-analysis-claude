# CLAUDE.md — 项目上下文与部署指南

## 项目概述

**MACD Analysis 六书融合策略平台** — 基于 ETH/USDT 的量化交易分析系统。

融合 7 本技术分析经典著作的策略：背离分析、均线、蜡烛图、布林带、量价分析、KDJ、海龟交易法则。提供回测优化、实盘信号检测、多周期联合决策等功能。

## 技术栈

- **后端**: Python 3 + Flask
- **前端**: Jinja2 模板 + 原生 JS（无框架）
- **数据**: Binance API (ETH/USDT K线)
- **部署**: Gunicorn + Nginx + systemd
- **服务器**: 阿里云 47.237.191.17:22222
- **域名**: https://invest.executor.life
- **GitHub**: git@github.com:itsoso/macd-analysis-claude.git

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
| `deploy.sh` | 自动化部署脚本 |

## 服务器信息

```
IP:          47.237.191.17
SSH端口:     22222
用户:        root
项目路径:    /opt/macd-analysis
虚拟环境:    /opt/macd-analysis/venv
服务名:      macd-analysis (systemd)
应用端口:    5100 (gunicorn → nginx 反代)
域名:        invest.executor.life
GitHub仓库:  git@github.com:itsoso/macd-analysis-claude.git
```

---

## 部署方法

### 方式一: deploy.sh 自动化部署（推荐，需要 SSH 能通）

```bash
# 首次完整部署（含 venv / systemd / nginx 配置）
./deploy.sh

# 日常更新代码并重启（最常用）
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

> deploy.sh 内部流程: 检查SSH连接 → git pull → pip install → systemctl restart → 健康检查

### 方式二: 手动 SSH 部署

```bash
# 1. 本地: 提交并推送代码到 GitHub
git add -A && git commit -m "your message" && git push origin main

# 2. SSH 登录服务器
ssh -p 22222 root@47.237.191.17

# 3. 服务器上: 拉取最新代码并重启
cd /opt/macd-analysis
git pull origin main
pip install -r requirements.txt   # 如果依赖有变化
systemctl restart macd-analysis

# 4. 验证服务状态
systemctl status macd-analysis
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5100/
```

### 方式三: 一行命令远程部署

```bash
ssh -p 22222 root@47.237.191.17 "cd /opt/macd-analysis && git pull origin main && systemctl restart macd-analysis"
```

### 方式四: 阿里云控制台 VNC 部署（SSH 不通时的备选方案）

当 SSH 因为网络/代理/防火墙问题连不上时，使用此方式：

1. 登录阿里云控制台: https://ecs.console.aliyun.com
2. 找到对应 ECS 实例 → 点击 **远程连接** → 选择 **VNC 连接**
3. 在 VNC 终端中执行：

```bash
cd /opt/macd-analysis && git pull origin main && systemctl restart macd-analysis
```

4. 验证: `systemctl status macd-analysis`

### SSH 连接故障排查

当前已知 SSH 可能出现 **"Connection timed out during banner exchange"** 的问题。

**原因分析**:
- TCP 连接（nc -z）成功，但 SSH 协议握手超时
- 可能是 sshd 的 `UseDNS yes` 配置导致反向 DNS 查询耗时
- 可能是本地网络代理/VPN 干扰 SSH 协议
- 本地 `~/.ssh/config` 中的 `IdentityAgent` (SecretAgent) 可能拦截连接

**已配置的绕过措施**:
- `deploy.sh` 中 SSH 已添加 `-o IdentityAgent=none -o ProxyCommand=none`
- `~/.ssh/config` 中已添加 `47.237.191.17` 的专用 Host 配置

**如果仍然超时**:
1. 确认不是 VPN/代理问题: 尝试关闭 VPN 后重试
2. 服务器端修复: 通过 VNC 登录后编辑 `/etc/ssh/sshd_config`，设置 `UseDNS no`，然后 `systemctl restart sshd`
3. 使用方式四（VNC）进行部署

---

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

多周期信号检测耗时较长，三层超时需保持一致：

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
- 服务器性能较低，信号检测需要耐心等待
