# H800 快速执行卡片

## 一键执行命令

```bash
# === H800 上执行 (按顺序) ===

# 1. Multi-Horizon LSTM (20-30 分钟)
python3 train_gpu.py --mode lstm --tf 1h

# 2. ONNX 导出 (5 分钟)
python3 train_gpu.py --mode onnx

# 3. Stacking (30-40 分钟)
python3 train_gpu.py --mode stacking --tf 1h

# 4. 打包回传
tar -czf macd_models_$(date +%Y%m%d_%H%M%S).tar.gz \
    data/ml_models/lstm_1h.* \
    data/ml_models/tft_1h.* \
    data/ml_models/stacking_* \
    data/gpu_results/train_*.log
```

## 快速验证

```bash
# LSTM 验证
python3 -c "import json; m=json.load(open('data/ml_models/lstm_1h_meta.json')); print(f'Best head: {m[\"best_head\"]}, Val AUC: {m[\"val_auc_5h\"]}/{m[\"val_auc_12h\"]}/{m[\"val_auc_24h\"]}')"

# ONNX 验证
python3 -c "import onnxruntime as ort, numpy as np; sess=ort.InferenceSession('data/ml_models/lstm_1h.onnx'); out=sess.run(None, {'input': np.random.randn(1,48,80).astype(np.float32)})[0]; print(f'ONNX range: [{out.min():.4f}, {out.max():.4f}]')"

# Stacking 验证
python3 -c "import json; m=json.load(open('data/ml_models/stacking_meta_1h.json')); print(f'OOF samples: {m[\"n_oof_samples\"]}, Meta AUC: {m[\"oof_meta_auc\"]}')"
```

## 成功标准

| 任务 | 指标 | 目标 |
|------|------|------|
| LSTM | Val AUC (best head) | ≥ 0.57 |
| LSTM | Test-Val 差距 | < 0.02 |
| ONNX | 输出范围 | [0.0, 1.0] |
| Stacking | Meta OOF AUC | ≥ 0.58 |
| Stacking | n_oof_samples | ≥ 10000 |

## 本机部署 (一键)

```bash
# 解压 + 健康检查 + 部署
tar -xzf macd_models_*.tar.gz && \
python check_ml_health.py --skip-live-check && \
./deploy.sh update
```

## Shadow 日志分析 (1-2 小时后)

```bash
# 拉取日志
scp -P 22222 root@47.237.191.17:/opt/macd-analysis/data/live/trades_$(date +%Y%m%d).jsonl data/live/

# 分析
python analyze_shadow_logs.py --days 2 --min-bars 20
```

## 开启 Stacking (确认稳定后)

```bash
ssh -p 22222 root@47.237.191.17 "echo 'ML_ENABLE_STACKING=1' >> /opt/macd-analysis/.env && systemctl restart macd-analysis"
```

---

**预计总耗时**: 2-3 小时 (H800) + 1-2 小时 (部署验证)
