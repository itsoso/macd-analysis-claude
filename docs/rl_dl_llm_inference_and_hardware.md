# 强化学习 / 深度学习 / LLM 改造与推理硬件方案

当前状态：**已有训练 GPU，暂无推理 GPU**。本文说明：  
1）若采用 RL/DL/LLM 需要做哪些改造；  
2）不单独购买推理 GPU 时的可行方案与配置建议。

---

## 一、若使用强化学习 / 深度学习 / LLM 需要做哪些改造

### 1. 强化学习 (RL，如 PPO 仓位)

| 维度 | 当前状态 | 需要改造 |
|------|----------|----------|
| **训练** | `train_gpu.py` 已有 `train_ppo_position()`，用 Stable-Baselines3 PPO，在训练 GPU 上跑 | 无需改；训练仍在训练机上即可 |
| **推理** | 实盘未接入 PPO policy | **需新增推理路径**：加载 `.zip` policy，在每根 K 线/每次信号时调用 `policy.predict(obs)` 得到动作（仓位比例） |
| **与现有管线** | 六书融合 + ML 方向预测 → 规则执行 | 将 PPO 输出作为「仓位缩放」或「是否开仓」的额外输入，在 `live_trading_engine` 或仓位计算处接入 |
| **依赖** | `stable_baselines3`, `gymnasium` | 推理仅需 `sb3` 加载 policy，**可 CPU 推理**，对推理 GPU 无硬需求 |
| **模型格式** | SB3 默认 `.zip`（内含 policy + 归一化等） | 保持即可；若需跨语言可导出 ONNX（SB3 支持有限，可选） |

**改造要点**：

- 在 `ml_live_integration` 或单独模块中增加「PPO 策略加载 + 单步推理」：输入 = 当前观测向量（与 `TradingEnv._obs()` 一致），输出 = 动作标量。
- 在 `live_signal_generator` 或 `live_trading_engine` 中：把 PPO 输出的仓位比例与现有开平仓逻辑结合（例如：仅当 PPO 仓位 > 某阈值且六书+ML 同向时才下单）。
- **推理设备**：PPO policy 推理很轻，**CPU 即可**；若希望统一放到 GPU 推理机上，可把「PPO 推理」也封装进 `ml_inference_server` 的同一服务，与方向预测一起返回。

---

### 2. 深度学习 (DL，如 LSTM / TFT / 更大序列模型)

| 维度 | 当前状态 | 需要改造 |
|------|----------|----------|
| **训练** | LSTM、TFT、Stacking 已在 `train_gpu` 上 GPU 训练 | 更大模型（Transformer-XL、Informer 等）也仍在训练 GPU 上训练即可 |
| **推理** | `ml_live_integration.py` 中 **全部 CPU**：`map_location='cpu'`、`device='cpu'` | 若上推理 GPU：见下文「推理设备统一」；若无推理 GPU：保持 CPU 或 ONNX CPU |
| **模型格式** | PyTorch `.pt` / `.pth` | 可选：导出 ONNX 后用 `onnxruntime-gpu` 或 `onnxruntime`（CPU）做推理，便于部署与跨环境 |
| **序列长度** | LSTM 48、TFT 96，实盘传 `features.tail(96)` | 若新模型更长（如 192），需在 ECS 与推理 API 间约定更长序列；`ml_inference_server` 已支持变长（不足 96 也可发） |
| **新模型结构** | 当前写死 LSTMAttention、EfficientTFT | 新增模型需在 `ml_live_integration` 中增加加载分支 + 前向调用，并在 `train_gpu` 中保存与 `ml_live_integration` 兼容的 state_dict / config |

**改造要点**：

- **推理设备统一**（若将来有推理 GPU）：  
  - 在 `ml_live_integration.py` 中：`INFERENCE_DEVICE = os.environ.get('ML_INFERENCE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')`  
  - 所有 `map_location='cpu'` → `map_location=INFERENCE_DEVICE`；所有 `device = 'cpu'` → `device = INFERENCE_DEVICE`；输入张量 `.to(INFERENCE_DEVICE)` 再前向。
- **无推理 GPU 时**：  
  - 保持当前 CPU 推理；或  
  - 使用 `train_gpu.export_onnx_models()` 导出 ONNX，用 `onnxruntime`（CPU）推理，往往比 PyTorch CPU 略快、部署简单。
- **更大模型**：若参数量或序列长度明显增加，单次推理延迟会上升，届时再考虑「推理 GPU」或「远程 GPU 推理 API」（方案二）更合适。

---

### 3. 大语言模型 (LLM)

| 维度 | 当前状态 | 需要改造 |
|------|----------|----------|
| **使用方式** | 项目内未使用 LLM | **全新模块**：需决定用途（例如：新闻/报告解读 → 情绪或事件标签；或策略说明生成） |
| **训练/微调** | 无 | 若微调：需在训练 GPU 上跑 SFT/LoRA 等；显存需求高（7B 级 LoRA 约 12GB+） |
| **推理** | 无 | **推理负载高**：7B 级 FP16 约 14GB 显存，量化后约 4–8GB；需专用推理进程或独立服务 |
| **与现有管线** | - | 若仅「文本→情绪/事件」：LLM 输出作为特征输入现有方向预测或 Regime；需设计 prompt、解析输出、对齐时间戳 |
| **依赖** | 无 | 需引入 `transformers`、`vllm`/`llama.cpp`/`sglang` 等推理后端；可选 OpenAI/阿里等 API 替代自建 |

**改造要点**：

- **用途与接口**：先定 LLM 产出物（如情绪分数、事件标签、是否重大新闻），再设计「输入文本 → prompt → 解析 → 数值/类别」的 pipeline，并接入 `ml_live_integration` 或特征工程（如新特征列）。
- **推理部署**：  
  - **自建**：需要 **推理 GPU**（至少单卡 8–24GB）；可与「训练机」共用一张卡做推理（见下文「用训练 GPU 做推理」），或单独买推理机。  
  - **API**：调用云端 LLM API，无需自购推理 GPU，但存在延迟、成本与依赖。
- **安全与延迟**：实盘若依赖 LLM 输出，需设超时与默认值（如 API 超时则回退到无 LLM 特征），避免阻塞交易。

---

## 二、当前硬件：有训练 GPU、无推理 GPU — 可选方案

### 方案 A：不买推理 GPU，全部 CPU 推理（现状增强）

| 项 | 说明 |
|----|------|
| **适用** | 模型规模不大（当前 LSTM/TFT 约 15 万参数）、请求频率低（如每分钟 1 次） |
| **做法** | ECS 上保持 `ml_live_integration` 全 CPU；可选：用 ONNX 导出 + `onnxruntime`（CPU）略提速 |
| **RL** | PPO 推理放 ECS CPU 即可 |
| **LLM** | 不自建；若用 LLM 则走云端 API |
| **成本** | 无额外硬件；延迟相对高（单次数百 ms 级可接受即可） |

---

### 方案 B：用「训练 GPU 机器」同时做推理（推荐，零新购）

你已有训练 GPU，可让**同一台机器既训练又提供推理 API**，无需再买推理专用机。

| 项 | 说明 |
|----|------|
| **架构** | 即现有 **方案二**：训练机部署 `ml_inference_server.py`，ECS 通过 `ml_gpu_inference_url` 调用该机 `/predict`；失败则回退 ECS 本地 CPU |
| **训练** | 训练任务仍在同一 GPU 上跑；建议训练时尽量不并发大量推理请求（或错峰） |
| **推理** | 在训练机上设 `ML_INFERENCE_DEVICE=cuda`，`ml_inference_server` 加载 `MLSignalEnhancer` 时用 CUDA；同一张卡训练空闲时做推理 |
| **网络** | 训练机与 ECS 同 VPC 最佳；若跨公网需鉴权 + HTTPS，并注意 5s 超时与稳定性 |
| **RL** | 若希望 PPO 也走 GPU（通常不必）：可把 PPO 推理一并放进该推理 API；否则 PPO 仍放 ECS CPU |
| **LLM** | 若将来在训练机上跑 LLM 推理（占显存大），需与 LSTM/TFT 共享显存或分时；7B 级建议单独一张卡或单独实例 |

**实施步骤概要**：

1. 在**训练 GPU 机器**上：安装与训练一致的 PyTorch（含 CUDA）、依赖，拉取本仓库。
2. 将训练产出的模型文件（`data/ml_models/` 下 LGB、LSTM、TFT、Stacking 等）同步到该机（或 NFS/对象存储）。
3. 启动推理服务：  
   `ML_INFERENCE_DEVICE=cuda python ml_inference_server.py --host 0.0.0.0 --port 5001`
4. 在 **ECS** 上配置：  
   `ml_gpu_inference_url=http://<训练机内网IP>:5001`（或环境变量 `ML_GPU_INFERENCE_URL`）。
5. 修改 `ml_live_integration.py`：在**推理服务所在进程**内使用 GPU（见 `docs/gpu_inference_options.md` 方案一代码要点）；当前 `ml_inference_server` 会传 `ML_INFERENCE_DEVICE=cuda`，但实际加载在 `ml_live_integration` 里仍写死 `cpu`，需在 **ml_live_integration** 中改为按环境变量或参数选择 `cuda`/`cpu`，推理机才会真正用 GPU。

这样**不需要购买新推理 GPU**，只需在现有训练机上长期跑一个推理进程，并让 ECS 指向它。

**代码已就绪**：`ml_live_integration.py` 已支持按环境变量 `ML_INFERENCE_DEVICE` 或自动检测 CUDA 选择推理设备（LSTM/TFT/Stacking 的加载与前向均使用 `self._inference_device`）。在训练机上启动推理服务时设置 `ML_INFERENCE_DEVICE=cuda` 即可用 GPU。

---

### 方案 C：购买专用推理 GPU 实例（将来若需要）

当出现以下情况时，再考虑单独购买推理 GPU：

- 模型变大（如大 Transformer、LLM），单次推理延迟或显存超过训练机单卡承受能力；
- 需要推理与训练完全隔离（训练时不影响实盘延迟）；
- 多台 ECS 或多策略共用同一推理服务，需要更高 QPS。

可选：

- **方案一（同机）**：ECS 变更为 GPU 实例（如阿里云 gn6i），在 ECS 上直接跑 Flask + `MLSignalEnhancer` 用 CUDA；见 `docs/gpu_inference_options.md`。
- **方案二（独立推理机）**：新购一台 GPU 实例专做推理，部署 `ml_inference_server`，ECS 继续通过 URL 调用；与方案 B 相同逻辑，只是机器专用于推理、不与训练共享。

---

## 三、推理设备在代码中的统一改法（方案 B 必做）

当前 `ml_live_integration.py` 中推理写死 CPU，若在训练机上用 GPU 做推理，需要：

1. **统一设备变量**（文件顶部或 `MLSignalEnhancer.__init__` 内）：
   ```python
   import torch
   _INFERENCE_DEVICE = os.environ.get('ML_INFERENCE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **所有加载模型**：  
   `map_location='cpu'` → `map_location=_INFERENCE_DEVICE`

3. **所有模型与张量**：  
   `device = 'cpu'` → `device = _INFERENCE_DEVICE`；前向前 `tensor.to(_INFERENCE_DEVICE)`。

4. **可选**：若用 ONNX，在推理机上安装 `onnxruntime-gpu`，创建 Session 时 `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`。

这样同一份代码在 ECS（无 GPU）上不设 `ML_INFERENCE_DEVICE` 或设为 `cpu` 则本地 CPU；在训练机上设 `ML_INFERENCE_DEVICE=cuda` 则用 GPU。

---

## 四、总结表

| 技术 | 训练 | 推理改造 | 是否必须买推理 GPU |
|------|------|----------|--------------------|
| **RL (PPO)** | 现有训练 GPU | 新增 policy 加载 + 单步推理并接入仓位逻辑 | **否**，CPU 即可 |
| **DL (LSTM/TFT/现有)** | 现有训练 GPU | 设备改为环境变量控制；可选 ONNX | **否**，可用方案 B（训练机当推理机） |
| **DL (更大模型)** | 现有训练 GPU | 新结构接入 + 序列长度/API 约定 | 视延迟/显存；可先方案 B |
| **LLM** | 训练 GPU 可微调 | 全新 pipeline + 推理服务 | **自建必须** GPU（可与训练机共用一卡）；或用云端 API 免自购 |

**建议**：  
- 短期：采用 **方案 B**（训练机跑 `ml_inference_server` + CUDA），在 `ml_live_integration` 中按上节做设备统一，**不买推理 GPU** 即可用上 GPU 推理。  
- 若后续上 LLM 或更大 DL 且单卡不够，再考虑单独购买推理 GPU 实例或使用 LLM 云端 API。
