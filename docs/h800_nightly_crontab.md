# H800 Nightly 自动训练与汇总

## 1) 一次性检查

```bash
cd /opt/macd-analysis
bash scripts/cron_h800_nightly.sh
```

输出产物：
- 训练日志：`logs/retrain/nightly_*.log`
- 汇总 JSON：`logs/retrain/h800_training_summary_latest.json`
- 汇总 Markdown：`logs/retrain/h800_training_summary_latest.md`

## 2) crontab 配置

```bash
crontab -e
```

示例（每天 02:20 执行）：

```cron
20 2 * * * cd /opt/macd-analysis && /opt/macd-analysis/scripts/cron_h800_nightly.sh >> /opt/macd-analysis/logs/retrain/cron_h800_nightly.log 2>&1
```

## 3) 常用环境变量（可选）

```cron
# 仅重跑 stacking + report（更快）
H800_STAGE=stacking
# 周期
H800_TIMEFRAMES=1h,4h,24h
# 默认别名周期
ML_STACKING_TIMEFRAME=1h
# 样本门槛
MIN_STACKING_SAMPLES=20000
# cron 默认跳过安装与数据校验；可改为 0
H800_NO_INSTALL=1
H800_NO_VERIFY_DATA=1
```

如果要在 crontab 中设置环境变量，写在任务前：

```cron
H800_STAGE=all
H800_TIMEFRAMES=1h,4h,24h
ML_STACKING_TIMEFRAME=1h
20 2 * * * cd /opt/macd-analysis && /opt/macd-analysis/scripts/cron_h800_nightly.sh >> /opt/macd-analysis/logs/retrain/cron_h800_nightly.log 2>&1
```

## 4) 快速排查

```bash
tail -n 200 logs/retrain/cron_h800_nightly.log
tail -n 200 logs/retrain/nightly_*.log
cat logs/retrain/h800_training_summary_latest.json | jq .
```

重点关注：
- `promotion.production_ready_tfs`
- `promotion.blocked_tfs`
- `stacking[*].status/reasons`
