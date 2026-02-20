# ML GPU 推理 API 方案设计（方案二：独立 GPU 机 + API）

## 1. 目标

- **GPU 机**：单独部署推理服务，加载 LSTM/TFT/Stacking 等模型并在 GPU 上执行方向预测，通过 HTTP API 返回 `bull_prob`。
- **ECS**：不换规格，在 `ml_live_integration.py` 中优先调用该 API 获取方向预测；失败或未配置时回退到本地 CPU 推理。Regime、分位数仍在 ECS 本地执行。

## 2. 架构

```
                    ┌─────────────────────────────────────────┐
                    │  ECS (47.237.191.17)                     │
                    │  live_signal_generator → enhance_signal  │
                    │  ml_live_integration.py                  │
                    │  - 若 ml_gpu_inference_url 已配置:       │
                    │     POST features → GPU API → bull_prob  │
                    │  - 否则或请求失败: 本地 CPU 方向预测      │
                    │  - Regime / 分位数 / 方向应用: 始终本地   │
                    └──────────────────┬──────────────────────┘
                                       │ HTTP POST /predict
                                       │ (内网或公网 + 鉴权)
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  GPU 机器 (独立 ECS / 自有 GPU 服务器)     │
                    │  ml_inference_server.py (Flask)          │
                    │  - 加载 MLSignalEnhancer(device='cuda')  │
                    │  - 仅执行方向预测 → bull_prob + 子项     │
                    │  - 返回 JSON                              │
                    └─────────────────────────────────────────┘
```

## 3. API 约定

### 3.1 请求

- **方法**: `POST`
- **路径**: `/predict`（或 `/ml/predict`）
- **Content-Type**: `application/json`
- **Body**:

```json
{
  "sell_score": 45.0,
  "buy_score": 38.0,
  "features": {
    "columns": ["ret_1", "ret_2", ...],
    "index": [0, 1, ...],
    "data": [[0.01, 0.02, ...], ...]
  }
}
```

- **features**: 由 ECS 端 `_compute_direction_features(df)` 得到，再取 **最后 96 行**（满足 LSTM 48、TFT 96 的序列长度），用 `df.to_json(orient='split')` 序列化后放入。若不足 96 行则发送全部。
- **sell_score / buy_score**: 当前六书融合的 SS/BS，可选供服务端日志或后续扩展；方向预测本身只需 features。

### 3.2 响应

- **成功** (200):

```json
{
  "success": true,
  "bull_prob": 0.58,
  "stacking_mode": false,
  "lgb_bull_prob": 0.55,
  "lstm_bull_prob": 0.62,
  "tft_bull_prob": 0.54,
  "ca_bull_prob": 0.57
}
```

- 若使用 Stacking：`stacking_mode: true`，可带 `stacking_bull_prob`、`stacking_lgb_prob` 等。
- **失败** (200 但业务错误 或 5xx): `success: false`, `error: "..."`。

### 3.3 超时与重试

- ECS 端：请求超时 **5 秒**，失败（超时/5xx/解析错误）时 **回退本地 CPU**，不重试或最多重试 1 次（可选）。

## 4. 配置

| 位置 | 项 | 说明 |
|------|----|------|
| **live_config.py** | `ml_gpu_inference_url: str = ""` | 为空则不用远程推理；非空如 `http://10.x.x.x:5001/predict` |
| **环境变量** | `ML_GPU_INFERENCE_URL` | 可覆盖 config，便于部署 |
| **GPU 机** | 端口、绑定地址 | 默认 `0.0.0.0:5001`，内网部署可仅监听内网 IP |

## 5. 安全（建议）

- 内网：GPU 与 ECS 同 VPC，API 仅监听内网 IP，安全组只放行 ECS。
- 公网/跨网：API Key（Header `X-API-Key`）或 VPC 对等/专线；HTTPS。

## 6. 兼容与回退

- **无 URL**：与现网一致，全部本地 CPU。
- **有 URL 且请求成功**：方向预测使用远程 `bull_prob`，其余（Regime、分位数、方向应用公式）不变。
- **有 URL 但请求失败**：自动回退到本地方向预测，不抛错，仅打日志。

## 7. 测试验证

- **单元**：Mock `requests.post` 返回固定 `bull_prob`，断言 `enhance_signal` 的 `ml_info["bull_prob"]` 与方向应用一致。
- **集成**：本地启动 `ml_inference_server.py`（CPU 或 GPU），设置 `ML_GPU_INFERENCE_URL=http://127.0.0.1:5001/predict`，跑一次 `live_runner.py --test-signal` 或单独脚本调用 `enhance_signal`，检查返回与日志。
