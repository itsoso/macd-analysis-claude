# H800 LGB 模型重训练 - 快速执行指南

## 目标
在 H800 上用新的 76 维特征（含 7 个高频微结构特征）重新训练 LGB 方向模型，预期 AUC 从 0.55 提升到 0.569。

## 前置条件
- ✅ 新特征已提交到 Git (commit 5e40b87)
- ✅ ml_features.py 包含 7 个新的微结构特征
- ✅ train_gpu.py 支持 --mode lgb 训练

## 执行流程

### 步骤 1: 本地准备数据（开发机）

```bash
# 1.1 确保有最新代码
cd /workspace/macd-analysis-claude
git pull origin main

# 1.2 检查数据是否完整（需要 5 年数据）
ls -lh data/klines/ETHUSDT/*.parquet
ls -lh data/klines/BTCUSDT/*.parquet
ls -lh data/klines/SOLUSDT/*.parquet
ls -lh data/klines/BNBUSDT/*.parquet

# 如果数据不完整，先拉取数据（约 15 分钟）
python3 fetch_5year_data.py

# 1.3 打包数据和代码
chmod +x pack_for_h800.sh
./pack_for_h800.sh

# 输出: macd_train_data.tar.gz (~62MB)
```

### 步骤 2: 传输到 H800（通过跳板机）

```bash
# 2.1 传输数据包到 H800
scp -J user@jumphost macd_train_data.tar.gz user@h800:~/work/

# 注意: 替换 user@jumphost 和 user@h800 为实际的用户名和主机
```

### 步骤 3: H800 上训练（GPU 训练机）

```bash
# 3.1 SSH 登录 H800
ssh -J user@jumphost user@h800

# 3.2 解压数据包
cd ~/work
tar -xzf macd_train_data.tar.gz

# 3.3 设置环境（首次需要，约 5 分钟）
chmod +x setup_h800.sh
./setup_h800.sh

# 如果使用 conda:
conda activate macd-train

# 3.4 验证数据完整性
python3 verify_data.py

# 3.5 开始训练 LGB 模型（约 10-15 分钟）
python3 train_gpu.py --mode lgb --tf 1h

# 训练输出:
# - data/ml_models/lgb_direction_model_1h.txt (新模型)
# - data/gpu_results/lgb_training.json (训练日志)

# 3.6 查看训练结果
cat data/gpu_results/lgb_training.json | python3 -m json.tool

# 3.7 打包模型文件
tar -czf macd_models_v6.tar.gz data/ml_models/ data/gpu_results/
ls -lh macd_models_v6.tar.gz
```

### 步骤 4: 回传模型（H800 → 开发机）

```bash
# 4.1 从 H800 传回开发机
scp -J jumphost macd_models_v6.tar.gz user@dev:~/macd-analysis/

# 4.2 在开发机上解压
cd ~/macd-analysis
tar -xzf macd_models_v6.tar.gz

# 4.3 验证新模型
ls -lh data/ml_models/lgb_direction_model_1h.txt
# 预期: 文件大小略大于旧模型（因为特征从 69 维增加到 76 维）
```

### 步骤 5: 部署到生产服务器（阿里云）

```bash
# 5.1 备份旧模型
ssh -p 22222 root@47.237.191.17 "cd /opt/macd-analysis && \
  cp data/ml_models/lgb_direction_model.txt data/ml_models/lgb_direction_model.txt.backup"

# 5.2 传输新模型到生产服务器
scp -P 22222 data/ml_models/lgb_direction_model_1h.txt \
  root@47.237.191.17:/opt/macd-analysis/data/ml_models/lgb_direction_model.txt

# 5.3 同步新的特征工程代码
scp -P 22222 ml_features.py \
  root@47.237.191.17:/opt/macd-analysis/

# 5.4 重启服务
ssh -p 22222 root@47.237.191.17 "cd /opt/macd-analysis && \
  systemctl restart macd-analysis && \
  sleep 5 && \
  systemctl status macd-analysis"

# 5.5 验证服务健康
curl -s https://invest.executor.life/api/health | python3 -m json.tool
```

### 步骤 6: 验证效果

```bash
# 6.1 查看 ML 健康检查
# 访问: https://invest.executor.life/status
# 检查 "LGB 方向模型" 是否显示新的文件大小

# 6.2 观察实盘信号
# 等待下一个信号生成，查看 ML 预测的 bull_prob
# 预期: 模型预测更准确，AUC 从 0.55 提升到 0.569

# 6.3 查看 Shadow 日志（24-48 小时后）
# 对比新旧模型的预测准确率
```

## 预期结果

| 指标 | 旧模型 (69维) | 新模型 (76维) | 提升 |
|------|--------------|--------------|------|
| 特征维度 | 69 | 76 (+7) | +10.1% |
| 模型文件大小 | ~49KB | ~54KB | +10.2% |
| 回测 AUC | 0.5497 | 0.5691 | +3.52% |
| 预期实盘 AUC | 0.55 | 0.569 | +3.45% |

## 关键新特征

新增的 7 个高频微结构特征：
1. **cum_ofi** - 累积订单流不平衡（重要性 #3）
2. **ofi_std5** - OFI 波动率（重要性 #7）
3. **ofi_ma5** - OFI 移动平均
4. **buy_sell_pressure** - 买卖压力（重要性 #30）
5. **large_trade_ratio** - 大单占比
6. **cum_ofi_slope** - 累积 OFI 斜率
7. **ofi** - 订单流不平衡

## 故障排查

### 问题 1: H800 数据不完整
```bash
# 解决: 在开发机重新拉取数据
python3 fetch_5year_data.py
./pack_for_h800.sh
# 重新传输
```

### 问题 2: GPU 内存不足
```bash
# 解决: 减少批大小或使用 CPU
python3 train_gpu.py --mode lgb --tf 1h --device cpu
```

### 问题 3: 训练失败
```bash
# 查看详细日志
python3 train_gpu.py --mode lgb --tf 1h 2>&1 | tee training.log
```

### 问题 4: 服务重启后 ML 不工作
```bash
# 检查日志
ssh -p 22222 root@47.237.191.17 "journalctl -u macd-analysis -n 100"

# 检查模型文件权限
ssh -p 22222 root@47.237.191.17 "ls -lh /opt/macd-analysis/data/ml_models/"
```

## 时间估算

| 步骤 | 预计时间 |
|------|---------|
| 数据准备 | 5-15 分钟（如需重新拉取数据）|
| 打包传输 | 2-5 分钟 |
| H800 环境设置 | 5 分钟（首次）|
| LGB 训练 | 10-15 分钟 |
| 模型回传 | 2 分钟 |
| 部署重启 | 3 分钟 |
| **总计** | **27-45 分钟** |

## 注意事项

1. ⚠️ **备份旧模型**: 部署前务必备份，以便回滚
2. ⚠️ **验证数据**: 确保 H800 上的数据完整且最新
3. ⚠️ **测试环境**: 建议先在测试环境验证新模型
4. ⚠️ **监控指标**: 部署后密切监控实盘表现
5. ⚠️ **回滚准备**: 如果新模型表现不佳，立即回滚

## 快速命令（一键执行）

```bash
# 本地打包
./pack_for_h800.sh && \
scp -J jumphost macd_train_data.tar.gz h800:~/work/

# H800 训练（在 H800 上执行）
cd ~/work && tar -xzf macd_train_data.tar.gz && \
python3 train_gpu.py --mode lgb --tf 1h && \
tar -czf macd_models_v6.tar.gz data/ml_models/ data/gpu_results/

# 部署到生产（在开发机上执行）
scp -J jumphost h800:~/work/macd_models_v6.tar.gz . && \
tar -xzf macd_models_v6.tar.gz && \
scp -P 22222 data/ml_models/lgb_direction_model_1h.txt \
  root@47.237.191.17:/opt/macd-analysis/data/ml_models/lgb_direction_model.txt && \
scp -P 22222 ml_features.py root@47.237.191.17:/opt/macd-analysis/ && \
ssh -p 22222 root@47.237.191.17 "systemctl restart macd-analysis"
```

---

**准备就绪！** 现在可以开始执行 H800 训练流程了。
