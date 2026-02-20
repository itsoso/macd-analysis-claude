# GPU 推理方案：ECS 配置 GPU 或独立 GPU 机 + API

当前实盘推理在 `ml_live_integration.py` 中全部使用 **CPU**（`map_location='cpu'`、`device='cpu'`）。若希望推理利用 GPU，有两种可行方案。

---

## 方案一：当前 ECS 升级/变更为 GPU 实例（同机推理）

### 1.1 阿里云 GPU 实例选型（推理场景）

| 规格族 | GPU | 显存 | 适用场景 | 说明 |
|--------|-----|------|----------|------|
| **gn6i** | NVIDIA T4 × 1 | 16GB | **小规模推理、性价比高** | 官方推荐推理，按量/包月均可 |
| gn7i | NVIDIA A10 × 1 | 24GB | 并发推理、图形+推理 | 性能更强，价格更高 |
| vgn6i-vws | T4 虚拟化分片 | 4/8/16GB | 轻量推理、成本敏感 | vGPU 分片，显存可选 |
| sgn8ia | Lovelace vGPU | 2~48GB | 轻量推理、多实例共享 | 虚拟化型，按显存选规格 |

**推荐**：  
- 预算有限、模型不大（当前 LSTM/TFT 约 148K 参数）：选 **gn6i** 最小规格，如 `ecs.gn6i-c4g1.xlarge`（4 vCPU, 15GB 内存, 1×T4 16GB）。  
- 需要更高并发或后续模型变大：选 **gn7i** 单卡，如 `ecs.gn7i-c8g1.2xlarge`（8 vCPU, 30GB 内存, 1×A10 24GB）。

**注意**：GPU 实例地域/库存可能受限，请在 [实例可购买地域](https://ecs-buy.aliyun.com/instanceTypes/#/instanceTypeByRegion) 确认后再选型。

### 1.2 操作步骤概要

1. **备份与规划**  
   - 对当前 ECS 做镜像或快照备份。  
   - 记录当前系统盘/数据盘、安全组、弹性 IP、域名解析等。

2. **购买/变配为 GPU 实例**  
   - **方式 A**：新购一台 GPU 实例（gn6i/gn7i），同一 VPC 内；把业务迁过去（代码、数据、Nginx、systemd 等），原 ECS 可下线或改作它用。  
   - **方式 B**：若当前实例支持「变配」到同地域的 GPU 规格，可在控制台 ECS → 实例 → 变配 中尝试（不支持则需新购+迁移）。

3. **安装 GPU 驱动与 CUDA**  
   - 阿里云部分公共镜像已带 GPU 驱动；若使用自定义镜像，需在 GPU 实例上安装 NVIDIA 驱动 + CUDA（与当前 PyTorch 版本匹配，如 CUDA 11.8/12.x）。  
   - 文档：[在 GPU 实例上安装 NVIDIA 驱动](https://help.aliyun.com/zh/ecs/user-guide/install-nvidia-drivers-on-a-gpu-instance)。

4. **代码侧：推理使用 GPU**  
   - 在 `ml_live_integration.py` 中，将推理设备从固定 `'cpu'` 改为「有 GPU 则用 GPU」：
     - 在文件前部或 `MLSignalEnhancer.__init__` 中统一定义推理设备，例如：
       - `import torch` 后：`INFERENCE_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`
       - 所有 `map_location='cpu'` 改为 `map_location=INFERENCE_DEVICE`
       - 所有 `device = 'cpu'`（LSTM/TFT/Stacking 等）改为 `device = INFERENCE_DEVICE`
       - 张量送入模型前 `.to(INFERENCE_DEVICE)`（若当前已在 CPU 上创建，需保证输入也在同一 device）。
   - 可选：通过环境变量控制，例如 `ML_INFERENCE_DEVICE=cuda` 或 `cpu`，便于同一份代码在无 GPU 环境强制 CPU。

5. **依赖**  
   - 若使用 PyTorch：安装带 CUDA 的版本，如 `pip install torch --index-url https://download.pytorch.org/whl/cu118`（按 CUDA 版本选）。  
   - 若后续改用 ONNX 推理：安装 `onnxruntime-gpu`，并在创建 InferenceSession 时使用 `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`。

6. **部署与验证**  
   - 部署完成后重启 `macd-analysis` 与 `macd-engine`，查看日志确认无 CUDA 报错；可写一小段脚本对 `MLSignalEnhancer` 做单次 `enhance_signal` 调用并打印耗时，对比同机 CPU 推理。

**优点**：无网络延迟，实现简单，运维单机即可。  
**缺点**：GPU 实例价格高于同档 CPU；变配可能不兼容需迁移。

---

## 方案二：独立 GPU 机器开放推理 API，ECS 通过 HTTP 调用

### 2.1 架构

```
当前 ECS (47.237.191.17)                 GPU 机器 (内网或公网)
┌─────────────────────────────┐         ┌─────────────────────────────┐
│ Flask / live_runner         │  HTTP   │ 推理 API 服务 (Flask/FastAPI) │
│ ml_live_integration.py      │ ──────→ │ 加载 LSTM/TFT/Stacking 等    │
│ 改为: 若配置了 GPU_API_URL   │  请求   │ 在 GPU 上推理 → 返回 bull_prob │
│ 则请求远程推理，否则本地 CPU │ ←────── │ 或返回完整 ML 结果            │
└─────────────────────────────┘         └─────────────────────────────┘
```

### 2.2 GPU 端：推理 API 服务

- **部署**：在带 GPU 的机器（可为阿里云 GPU 实例、或你已有的 H800/其他 GPU 服务器）上单独跑一个进程，只负责「输入特征/DataFrame 序列化 → 模型推理 → 返回结果」。
- **接口约定**（示例）：
  - **POST** `/predict` 或 `/ml/enhance`
  - 请求体：JSON，包含当前 K 线/特征或已计算好的特征（如 `features` 列表、或 `df` 的 base64/JSON 序列化）。
  - 响应：JSON，如 `{"bull_prob": 0.58, "regime_vol": 0.1, "kelly": 0.5, ...}`，与当前 `ml_info` 对齐以便 ECS 端直接使用。
- **实现方式**：
  - 用当前项目里的 `ml_live_integration` 在 GPU 机上跑一份「仅推理」的服务：读入请求 → 调用 `MLSignalEnhancer.enhance_signal(...)`（该进程内 `device='cuda'`）→ 将返回的 `enhanced_ss/bs`、`ml_info` 等序列化返回。
  - 或单独写一个轻量 Flask/FastAPI：只加载 PyTorch/ONNX 模型，接收特征后做前向计算并返回 `bull_prob` 等，减少对全量六书管线的依赖。
- **安全**：  
  - 内网调用：GPU 机与 ECS 同 VPC，API 仅监听内网 IP，用安全组限制只允许 ECS 访问。  
  - 跨网/公网：加 API Key、或 VPC 对等/专线，并走 HTTPS。

### 2.3 ECS 端：调用远程推理

- 在 `ml_live_integration.py` 中增加「远程推理模式」：
  - 若配置了 `GPU_INFERENCE_API_URL`（或从 `live_config` 读），则：
    - 在 `enhance_signal` 内，将本机算好的特征（或当前 bar 的输入）序列化，`requests.post(GPU_INFERENCE_API_URL, json=payload, timeout=5)` 发送；
    - 解析响应 JSON，得到 `bull_prob`、`enhanced_ss/bs`、`kelly` 等，拼成与当前一致的 `ml_info` 和返回值，后续逻辑不变。
  - 若未配置或请求失败（超时/5xx），回退到本地 CPU 推理（当前逻辑）。
- **超时与重试**：建议超时 3–5 秒、重试 1 次，避免阻塞实盘信号循环。

### 2.4 优点与注意点

- **优点**：  
  - 当前 ECS 无需变更规格，成本可控。  
  - GPU 机可专用推理、与 H800 训练环境分离；多台 ECS 可共用一个推理 API。  
- **注意**：  
  - 多一跳网络延迟（内网通常 1–5ms，可接受）。  
  - 需保证 GPU 机高可用（重启、故障时 ECS 能回退 CPU 或另一台推理服务）。  
  - 接口版本要与 ECS 端请求/响应格式兼容，模型文件需在 GPU 机上有且与 ECS 使用的版本一致（或通过 CI/部署脚本同步）。

---

## 方案对比摘要

| 维度 | 方案一：ECS 直接上 GPU | 方案二：独立 GPU 机 + API |
|------|------------------------|---------------------------|
| 实现难度 | 低（变配/迁移 + 改 device） | 中（需部署 API + ECS 客户端） |
| 延迟 | 最低（本机） | 略增（网络 RTT） |
| 成本 | GPU 实例包月/按量 | GPU 机 + 当前 ECS 保留 |
| 扩展性 | 单机算力上限 | 可多实例负载均衡、多 ECS 共用 |
| 运维 | 单机运维 | 需维护 ECS + 推理服务两台 |

**建议**：  
- 若希望改动最小、且可接受一定 GPU 实例费用：优先 **方案一**，选 gn6i 小规格，并在 `ml_live_integration.py` 中统一改为「有 CUDA 则用 GPU」。  
- 若当前 ECS 不能变配、或希望 GPU 与 Web/交易解耦、或有多台 ECS 共用推理：采用 **方案二**，先在一台 GPU 机上实现 `/predict`，再在 ECS 上增加远程调用与回退逻辑。

---

## 代码改动要点（方案一）

在 `ml_live_integration.py` 中：

1. **统一推理设备**（文件顶部或类内常量）：
   ```python
   import torch
   _INFERENCE_DEVICE = os.environ.get('ML_INFERENCE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **所有加载模型处**：`map_location='cpu'` → `map_location=_INFERENCE_DEVICE`。

3. **所有创建/移动模型与张量处**：`device = 'cpu'` → `device = _INFERENCE_DEVICE`；输入张量 `.to(_INFERENCE_DEVICE)` 再前向。

4. **可选**：若使用 ONNX，在 GPU 机上用 `onnxruntime-gpu`，`InferenceSession(..., providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])`。

完成上述修改后，在同一台 GPU ECS 上跑现有服务即可「推理利用 GPU」；无需改接口或调用方式。
