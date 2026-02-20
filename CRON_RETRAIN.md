# 每日自动重训 Cron 配置

> 说明：本文件是 `train_gpu.py --mode retrain` 的轻量增量重训方案。  
> 如果你在 H800 上跑完整训练计划（base/stacking/onnx/report），请优先使用：
> `docs/h800_nightly_crontab.md` + `scripts/cron_h800_nightly.sh`。

## 安装步骤

### 1. 编辑 crontab
```bash
crontab -e
```

### 2. 添加定时任务
```cron
# 每天凌晨 4 点执行模型重训
0 4 * * * /workspace/macd-analysis-claude/cron_retrain.sh >> /workspace/macd-analysis-claude/logs/retrain/cron.log 2>&1
```

### 3. 验证 crontab
```bash
crontab -l
```

## 手动测试

```bash
# 测试脚本
./cron_retrain.sh

# 查看日志
tail -f logs/retrain/retrain_*.log
```

## 重训逻辑

1. **数据准备**: 加载最新 K 线数据
2. **增量训练**: 从旧模型 warm-start，训练 100 轮
3. **验证**: 在最近数据上评估 AUC
4. **自动替换**: 如果新 AUC >= 旧 AUC * 0.98，替换模型
5. **日志记录**: 保存到 `data/gpu_results/retrain_log.jsonl`

## 重训日志

查看历史重训记录:
```bash
cat data/gpu_results/retrain_log.jsonl | jq .
```

示例输出:
```json
{
  "timestamp": "2026-02-20T08:30:47.123456",
  "tf": "1h",
  "old_auc": 0.5566,
  "new_auc": 0.5593,
  "n_train": 37175,
  "n_test": 3063,
  "action": "replaced"
}
```

## 监控建议

1. **每日检查**: 查看 cron 日志确认执行成功
2. **AUC 趋势**: 监控 AUC 是否持续下降（市场 regime 变化）
3. **告警设置**: AUC < 0.52 时发送通知

## 故障排查

### Cron 未执行
```bash
# 检查 cron 服务状态
systemctl status cron

# 查看 cron 日志
grep CRON /var/log/syslog
```

### 训练失败
```bash
# 查看详细日志
tail -100 logs/retrain/retrain_*.log

# 手动执行测试
python3 train_gpu.py --mode retrain --tf 1h
```

## 高级配置

### 多周期重训
修改 `cron_retrain.sh`:
```bash
$PYTHON train_gpu.py --mode retrain --tf 1h,4h
```

### 部署到生产
取消注释 `cron_retrain.sh` 中的部署命令:
```bash
./deploy.sh update
```

### 通知集成
配置 Telegram/Email 通知（见脚本注释）
