"""
H800 GPU 离线训练入口。
完全不依赖 Binance API，从本地 Parquet 加载数据。

用法:
  python3 train_gpu.py --mode lgb          # LightGBM/XGBoost GPU 训练
  python3 train_gpu.py --mode lstm         # LSTM + Attention 深度学习
  python3 train_gpu.py --mode optuna       # Optuna 超参优化
  python3 train_gpu.py --mode all          # 全部流程
  python3 train_gpu.py --mode lgb --tf 1h  # 指定单周期
  python3 train_gpu.py --mode backtest     # GPU 加速回测优化
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'gpu_results')
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'ml_models')

TIMEFRAMES = ['15m', '1h', '4h', '24h']
SYMBOL = 'ETHUSDT'


# ================================================================
# 数据加载 (纯本地 Parquet，不访问网络)
# ================================================================

def load_klines_local(symbol: str, interval: str) -> pd.DataFrame:
    """从本地 Parquet 加载K线，不触发任何 API 调用"""
    path = os.path.join(BASE_DIR, 'data', 'klines', symbol, f'{interval}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"K线数据不存在: {path}")
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    log.info(f"加载 {symbol}/{interval}: {len(df):,} 条 "
             f"({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def load_funding_local(symbol: str) -> Optional[pd.DataFrame]:
    path = os.path.join(BASE_DIR, 'data', 'funding_rates', f'{symbol}_funding.parquet')
    if not os.path.exists(path):
        log.warning(f"Funding 数据不存在: {path}")
        return None
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated(keep='last')].sort_index()


def load_oi_local(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    period_map = {'15m': '15m', '1h': '1h', '4h': '4h', '24h': '1d'}
    period = period_map.get(interval, '1h')
    path = os.path.join(BASE_DIR, 'data', 'open_interest', symbol, f'{period}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated(keep='first')].sort_index()


def load_mark_local(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    path = os.path.join(BASE_DIR, 'data', 'mark_klines', symbol, f'{interval}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated(keep='first')].sort_index()


def prepare_features(symbol: str, interval: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载数据 → 计算指标 → 生成 ML 特征 → 生成标签
    返回 (features_df, labels_df)
    """
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages
    from ml_features import compute_ml_features

    df = load_klines_local(symbol, interval)
    df = add_all_indicators(df)
    df = add_moving_averages(df)

    # 合并 funding / OI / mark 数据
    funding = load_funding_local(symbol)
    if funding is not None and 'funding_rate' in funding.columns:
        df = df.join(funding[['funding_rate']], how='left')
        df['funding_rate'] = df['funding_rate'].ffill()

    oi = load_oi_local(symbol, interval)
    if oi is not None and 'open_interest_value' in oi.columns:
        df = df.join(oi[['open_interest_value']], how='left')
        df['open_interest_value'] = df['open_interest_value'].ffill()

    mark = load_mark_local(symbol, interval)
    if mark is not None and 'mark_close' in mark.columns:
        for col in ['mark_open', 'mark_high', 'mark_low', 'mark_close']:
            if col in mark.columns:
                df = df.join(mark[[col]], how='left')
                df[col] = df[col].ffill()

    features = compute_ml_features(df)

    # 标签: 多尺度利润化标签
    cost_pct = 0.0015
    labels_dict = {}
    for h in [3, 5, 12, 24]:
        fwd_ret = df['close'].pct_change(h).shift(-h)
        labels_dict[f'profitable_long_{h}'] = (fwd_ret > cost_pct * 2).astype(float)
        labels_dict[f'fwd_ret_{h}'] = fwd_ret

    labels_df = pd.DataFrame(labels_dict, index=df.index)

    # 稀疏列 (OI/funding) 用 0 填充，不因其丢弃整行数据
    sparse_cols = [c for c in features.columns
                   if features[c].isna().sum() > len(features) * 0.5]
    if sparse_cols:
        log.info(f"稀疏列 ({len(sparse_cols)}) 填充 0: {sparse_cols}")
        features[sparse_cols] = features[sparse_cols].fillna(0)

    valid = features.notna().all(axis=1) & labels_df.notna().all(axis=1)
    features = features[valid]
    labels_df = labels_df.loc[features.index]

    log.info(f"特征: {features.shape[1]} 维, 有效样本: {len(features):,}")
    return features, labels_df


# ================================================================
# 模式 1: LightGBM/XGBoost GPU 训练
# ================================================================

def detect_gpu():
    """检测是否有 GPU 可用"""
    gpu_info = {'torch_cuda': False, 'lgb_gpu': False}
    try:
        import torch
        gpu_info['torch_cuda'] = torch.cuda.is_available()
        if gpu_info['torch_cuda']:
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['gpu_mem_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        # 实际测试 CUDA 是否可用
        try:
            import numpy as _np
            _X = _np.array([[0,1],[1,0],[0,1],[1,0]], dtype=_np.float64)
            _y = _np.array([0,1,0,1], dtype=_np.float64)
            _test_data = lgb.Dataset(_X, label=_y)
            _test_params = {'device': 'cuda', 'num_leaves': 4, 'verbose': -1}
            lgb.train(_test_params, _test_data, num_boost_round=1)
            gpu_info['lgb_gpu'] = True
        except Exception:
            gpu_info['lgb_gpu'] = False
    except ImportError:
        pass
    return gpu_info


def train_lgb_gpu(timeframes: List[str] = None):
    """LightGBM GPU Walk-Forward 训练"""
    from ml_predictor import WalkForwardEngine, MLConfig
    from ml_features import select_features

    timeframes = timeframes or TIMEFRAMES
    gpu_info = detect_gpu()
    log.info(f"GPU 状态: {gpu_info}")

    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"训练 LightGBM — {SYMBOL}/{tf}")
        log.info(f"{'='*60}")

        t0 = time.time()
        features, labels_df = prepare_features(SYMBOL, tf)

        config = MLConfig()
        config.use_multi_horizon = True
        config.expanding_window = True
        config.train_window = min(2400, len(features) // 2)
        config.use_feature_selection = True

        # GPU 参数: CUDA 版 LightGBM 在大数据量上不稳定 (segfault),
        # LightGBM CPU 对此数据量已足够快, GPU 加速留给 LSTM/TFT
        if gpu_info.get('lgb_gpu'):
            log.info("LightGBM 使用 CPU (CUDA 版在大数据量上不稳定)")

        config.lgb_num_boost_round = 500
        config.lgb_early_stopping_rounds = 50

        primary_label = labels_df['profitable_long_5']
        engine = WalkForwardEngine(config)
        result_df = engine.run(features, primary_label, labels_df=labels_df, verbose=True)

        summary = engine.summary()
        elapsed = time.time() - t0
        summary['elapsed_sec'] = round(elapsed, 1)
        summary['timeframe'] = tf
        summary['samples'] = len(features)
        summary['features'] = features.shape[1]

        all_results[tf] = summary
        log.info(f"\n{tf} 完成: {summary}")
        log.info(f"耗时: {elapsed:.1f}s")

        # 保存预测结果
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_df.to_parquet(os.path.join(RESULTS_DIR, f'lgb_wf_{tf}.parquet'))

    # 汇总
    save_results('lgb_walkforward', all_results)
    return all_results


# ================================================================
# 模式 2: LSTM + Attention 深度学习
# ================================================================

def train_lstm(timeframes: List[str] = None, multi_horizon: bool = True):
    """PyTorch LSTM + Attention 训练"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        log.error("PyTorch 未安装，跳过 LSTM 训练")
        return {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"PyTorch 设备: {device}")

    timeframes = timeframes or TIMEFRAMES
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"训练 LSTM — {SYMBOL}/{tf}")
        log.info(f"配置: multi_horizon={bool(multi_horizon)}")
        log.info(f"{'='*60}")

        t0 = time.time()
        features, labels_df = prepare_features(SYMBOL, tf)

        result = _train_lstm_single(features, labels_df, tf, device, multi_horizon=multi_horizon)
        result['elapsed_sec'] = round(time.time() - t0, 1)
        all_results[tf] = result
        log.info(f"{tf} 完成: {result}")

    save_results('lstm_training', all_results)
    return all_results


def _train_lstm_single(features, labels_df, tf, device, multi_horizon: bool = True):
    """单周期 LSTM 训练 (Walk-Forward)"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    SEQ_LEN = 48
    HIDDEN_DIM = 192  # 强化: 128 → 192
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 256
    EPOCHS = 50  # 已经是 50
    LR = 1e-3

    class LSTMAttention(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
            self.attn_fc = nn.Linear(hidden_dim * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # Attention over time steps
            attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
            context = (attn_weights * lstm_out).sum(dim=1)
            return self.classifier(context).squeeze(-1)  # 输出 logits

    class LSTMMultiHorizon(nn.Module):
        """Multi-Horizon LSTM: 3 classification heads for 5h/12h/24h predictions"""
        def __init__(self, input_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
            self.attn_fc = nn.Linear(hidden_dim * 2, 1)

            # 3 separate classification heads
            self.head_5h = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
            self.head_12h = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
            self.head_24h = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x, return_all=False):
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
            context = (attn_weights * lstm_out).sum(dim=1)

            out_5h = self.head_5h(context).squeeze(-1)
            out_12h = self.head_12h(context).squeeze(-1)
            out_24h = self.head_24h(context).squeeze(-1)

            if return_all:
                return out_5h, out_12h, out_24h
            # Default: return best head (determined during training)
            return out_5h

    # 准备序列数据
    feat_values = features.values.astype(np.float32)

    # Multi-horizon labels
    use_multi_horizon = bool(multi_horizon)
    if use_multi_horizon:
        label_5h = labels_df['profitable_long_5'].values.astype(np.float32)
        label_12h = labels_df['profitable_long_12'].values.astype(np.float32)
        label_24h = labels_df['profitable_long_24'].values.astype(np.float32)
        log.info(f"[Multi-Horizon] 使用 3 个预测头: 5h/12h/24h")
    else:
        label_values = labels_df['profitable_long_5'].values.astype(np.float32)

    # 标准化 (逐特征)
    feat_mean = np.nanmean(feat_values[:len(feat_values)//2], axis=0)
    feat_std = np.nanstd(feat_values[:len(feat_values)//2], axis=0) + 1e-8
    feat_values = (feat_values - feat_mean) / feat_std
    feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=3.0, neginf=-3.0)
    feat_values = np.clip(feat_values, -5, 5)

    input_dim = feat_values.shape[1]

    # Walk-Forward 分割: 60% train, 10% val, 30% test
    n = len(feat_values) - SEQ_LEN
    train_end = int(n * 0.6)
    val_end = int(n * 0.7)

    def make_sequences(start, end):
        X, y_5h, y_12h, y_24h = [], [], [], []
        for i in range(start, min(end, n)):
            seq = feat_values[i:i + SEQ_LEN]
            if use_multi_horizon:
                lbl_5 = label_5h[i + SEQ_LEN]
                lbl_12 = label_12h[i + SEQ_LEN]
                lbl_24 = label_24h[i + SEQ_LEN]
                if not (np.isnan(lbl_5) or np.isnan(lbl_12) or np.isnan(lbl_24)):
                    X.append(seq)
                    y_5h.append(lbl_5)
                    y_12h.append(lbl_12)
                    y_24h.append(lbl_24)
            else:
                lbl = label_values[i + SEQ_LEN]
                if not np.isnan(lbl):
                    X.append(seq)
                    y_5h.append(lbl)
        if not X:
            return None, None, None, None
        X_tensor = torch.tensor(np.array(X))
        if use_multi_horizon:
            return X_tensor, torch.tensor(np.array(y_5h)), torch.tensor(np.array(y_12h)), torch.tensor(np.array(y_24h))
        else:
            return X_tensor, torch.tensor(np.array(y_5h)), None, None

    X_train, y_train_5h, y_train_12h, y_train_24h = make_sequences(0, train_end)
    X_val, y_val_5h, y_val_12h, y_val_24h = make_sequences(train_end, val_end)
    X_test, y_test_5h, y_test_12h, y_test_24h = make_sequences(val_end, n)

    if X_train is None or len(X_train) < 100:
        return {'error': '训练数据不足', 'train_samples': 0}

    log.info(f"序列数据: train={len(X_train)}, val={len(X_val) if X_val is not None else 0}, "
             f"test={len(X_test) if X_test is not None else 0}")

    # 模型
    if use_multi_horizon:
        model = LSTMMultiHorizon(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    else:
        model = LSTMAttention(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and not use_bf16 else None
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if use_multi_horizon:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train_5h, y_train_12h, y_train_24h),
            batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'),
        )
    else:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train_5h),
            batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'),
        )

    best_val_auc = 0
    best_head_name = '5h'  # Track which head performs best
    patience = 10
    no_improve = 0
    val_auc_5h = val_auc_12h = val_auc_24h = 0.5

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            if use_multi_horizon:
                xb, yb_5h, yb_12h, yb_24h = batch
                xb = xb.to(device)
                yb_5h, yb_12h, yb_24h = yb_5h.to(device), yb_12h.to(device), yb_24h.to(device)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    if use_multi_horizon:
                        pred_5h, pred_12h, pred_24h = model(xb, return_all=True)
                        loss = (criterion(pred_5h, yb_5h) +
                                criterion(pred_12h, yb_12h) +
                                criterion(pred_24h, yb_24h)) / 3.0
                    else:
                        pred = model(xb)
                        loss = criterion(pred, yb)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                if use_multi_horizon:
                    pred_5h, pred_12h, pred_24h = model(xb, return_all=True)
                    loss = (criterion(pred_5h, yb_5h) +
                            criterion(pred_12h, yb_12h) +
                            criterion(pred_24h, yb_24h)) / 3.0
                else:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # 验证
        if X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                if use_multi_horizon:
                    val_pred_5h, val_pred_12h, val_pred_24h = model(X_val.to(device), return_all=True)
                    val_pred_5h = torch.sigmoid(val_pred_5h).cpu().numpy()
                    val_pred_12h = torch.sigmoid(val_pred_12h).cpu().numpy()
                    val_pred_24h = torch.sigmoid(val_pred_24h).cpu().numpy()
                    val_true_5h = y_val_5h.numpy()
                    val_true_12h = y_val_12h.numpy()
                    val_true_24h = y_val_24h.numpy()
                else:
                    val_pred = torch.sigmoid(model(X_val.to(device))).cpu().numpy()
                    val_true = y_val_5h.numpy()
            try:
                from sklearn.metrics import roc_auc_score
                if use_multi_horizon:
                    val_auc_5h = roc_auc_score(val_true_5h, val_pred_5h)
                    val_auc_12h = roc_auc_score(val_true_12h, val_pred_12h)
                    val_auc_24h = roc_auc_score(val_true_24h, val_pred_24h)
                    val_auc = max(val_auc_5h, val_auc_12h, val_auc_24h)  # Use best head
                    if val_auc == val_auc_5h:
                        best_head_name = '5h'
                    elif val_auc == val_auc_12h:
                        best_head_name = '12h'
                    else:
                        best_head_name = '24h'
                else:
                    val_auc = roc_auc_score(val_true, val_pred)
            except Exception:
                val_auc = 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
                # 保存最佳模型
                os.makedirs(MODEL_DIR, exist_ok=True)
                model_path = os.path.join(MODEL_DIR, f'lstm_{tf}.pt')
                # 删除旧文件（如果存在且无法写入）
                if os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                    except PermissionError:
                        log.warning(f"无法删除旧模型文件 {model_path}，尝试覆盖")
                torch.save(model.state_dict(), model_path)
                # 保存元数据（推理端契约）
                import json
                meta_path = os.path.join(MODEL_DIR, f'lstm_{tf}_meta.json')
                meta_payload = {
                    'multi_horizon': bool(use_multi_horizon),
                    'best_head': best_head_name if use_multi_horizon else '5h',
                    'seq_len': SEQ_LEN,
                    'input_dim': input_dim,
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'feature_names': list(features.columns),
                    'feat_mean': feat_mean.tolist(),
                    'feat_std': feat_std.tolist(),
                }
                if use_multi_horizon:
                    meta_payload.update({
                        'val_auc_5h': round(val_auc_5h, 4),
                        'val_auc_12h': round(val_auc_12h, 4),
                        'val_auc_24h': round(val_auc_24h, 4),
                    })
                else:
                    meta_payload['val_auc_5h'] = round(val_auc, 4)
                with open(meta_path, 'w') as f:
                    json.dump(meta_payload, f, indent=2)
            else:
                no_improve += 1

            if epoch % 5 == 0 or no_improve == 0:
                if use_multi_horizon:
                    log.info(f"  Epoch {epoch:3d}: loss={epoch_loss/len(train_loader):.4f} "
                             f"val_AUC: 5h={val_auc_5h:.4f} 12h={val_auc_12h:.4f} 24h={val_auc_24h:.4f} "
                             f"best={best_val_auc:.4f} ({best_head_name})")
                else:
                    log.info(f"  Epoch {epoch:3d}: loss={epoch_loss/len(train_loader):.4f} "
                             f"val_AUC={val_auc:.4f} best={best_val_auc:.4f}")

            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    # 测试集评估
    test_auc = 0.5
    test_auc_5h = test_auc_12h = test_auc_24h = 0.5
    if X_test is not None and len(X_test) > 0:
        best_path = os.path.join(MODEL_DIR, f'lstm_{tf}.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            if use_multi_horizon:
                test_pred_5h, test_pred_12h, test_pred_24h = model(X_test.to(device), return_all=True)
                test_pred_5h = torch.sigmoid(test_pred_5h).cpu().numpy()
                test_pred_12h = torch.sigmoid(test_pred_12h).cpu().numpy()
                test_pred_24h = torch.sigmoid(test_pred_24h).cpu().numpy()
                test_true_5h = y_test_5h.numpy()
                test_true_12h = y_test_12h.numpy()
                test_true_24h = y_test_24h.numpy()
            else:
                test_pred = torch.sigmoid(model(X_test.to(device))).cpu().numpy()
                test_true = y_test_5h.numpy()
        try:
            from sklearn.metrics import roc_auc_score
            if use_multi_horizon:
                test_auc_5h = roc_auc_score(test_true_5h, test_pred_5h)
                test_auc_12h = roc_auc_score(test_true_12h, test_pred_12h)
                test_auc_24h = roc_auc_score(test_true_24h, test_pred_24h)
                test_auc = max(test_auc_5h, test_auc_12h, test_auc_24h)
            else:
                test_auc = roc_auc_score(test_true, test_pred)
        except Exception:
            pass

    result = {
        'best_val_auc': round(best_val_auc, 4),
        'test_auc': round(test_auc, 4),
        'train_samples': len(X_train),
        'val_samples': len(X_val) if X_val is not None else 0,
        'test_samples': len(X_test) if X_test is not None else 0,
        'epochs_trained': epoch + 1,
        'input_dim': input_dim,
        'multi_horizon': bool(use_multi_horizon),
        'bf16': use_bf16,
    }

    if use_multi_horizon:
        result.update({
            'best_head': best_head_name,
            'val_auc_5h': round(val_auc_5h, 4),
            'val_auc_12h': round(val_auc_12h, 4),
            'val_auc_24h': round(val_auc_24h, 4),
            'test_auc_5h': round(test_auc_5h, 4),
            'test_auc_12h': round(test_auc_12h, 4),
            'test_auc_24h': round(test_auc_24h, 4),
        })
        log.info(f"\n[Multi-Horizon] 最佳预测头: {best_head_name}")
        log.info(f"  验证 AUC: 5h={val_auc_5h:.4f}, 12h={val_auc_12h:.4f}, 24h={val_auc_24h:.4f}")
        log.info(f"  测试 AUC: 5h={test_auc_5h:.4f}, 12h={test_auc_12h:.4f}, 24h={test_auc_24h:.4f}")

    return result


# ================================================================
# 模式 3: Optuna 超参优化
# ================================================================

def train_optuna(timeframes: List[str] = None, n_trials: int = 200):
    """Optuna 贝叶斯超参优化 (GPU 加速)"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("Optuna 未安装: pip install optuna")
        return {}

    timeframes = timeframes or ['1h']
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"Optuna 超参优化 — {SYMBOL}/{tf} ({n_trials} trials)")
        log.info(f"{'='*60}")

        t0 = time.time()
        features, labels_df = prepare_features(SYMBOL, tf)
        label = labels_df['profitable_long_5']

        # 分割: 70% 搜索, 30% 最终验证
        split = int(len(features) * 0.7)
        search_feat = features.iloc[:split]
        search_label = label.iloc[:split]
        holdout_feat = features.iloc[split:]
        holdout_label = label.iloc[split:]

        gpu_info = detect_gpu()

        def objective(trial):
            import lightgbm as lgb

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 5.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 10.0, log=True),
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42,
            }

            if gpu_info.get('lgb_gpu'):
                params['device'] = 'cuda'

            # Walk-Forward 内部验证
            n = len(search_feat)
            train_end = int(n * 0.7)
            purge = 24
            val_start = train_end + purge
            X_tr = search_feat.iloc[:train_end]
            y_tr = search_label.iloc[:train_end]
            X_va = search_feat.iloc[val_start:]
            y_va = search_label.iloc[val_start:]

            valid_tr = y_tr.notna()
            valid_va = y_va.notna()
            X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
            X_va, y_va = X_va[valid_va], y_va[valid_va]

            if len(X_tr) < 500 or len(X_va) < 50:
                return 0.5

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=trial.suggest_int('num_boost_round', 100, 800),
                valid_sets=[dval],
                valid_names=['val'],
                callbacks=[
                    lgb.early_stopping(30),
                    lgb.log_evaluation(0),
                ],
            )

            pred = model.predict(X_va)
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_va, pred)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        log.info(f"\n最优参数 (val_AUC={best.value:.4f}):")
        for k, v in best.params.items():
            log.info(f"  {k}: {v}")

        # Holdout 验证
        holdout_auc = _evaluate_on_holdout(
            search_feat, search_label, holdout_feat, holdout_label,
            best.params, gpu_info
        )

        result = {
            'best_val_auc': round(best.value, 4),
            'holdout_auc': round(holdout_auc, 4),
            'best_params': best.params,
            'n_trials': n_trials,
            'elapsed_sec': round(time.time() - t0, 1),
        }
        all_results[tf] = result

    save_results('optuna_search', all_results)
    return all_results


def _evaluate_on_holdout(X_train, y_train, X_test, y_test, params, gpu_info):
    """用最优参数在 holdout 集上评估"""
    import lightgbm as lgb

    lgb_params = {k: v for k, v in params.items()
                  if k not in ('num_boost_round',)}
    lgb_params.update({
        'objective': 'binary', 'metric': 'auc',
        'verbose': -1, 'n_jobs': -1, 'seed': 42,
    })
    if gpu_info.get('lgb_gpu'):
        lgb_params['device'] = 'cuda'

    valid_tr = y_train.notna()
    valid_te = y_test.notna()

    dtrain = lgb.Dataset(X_train[valid_tr], label=y_train[valid_tr])
    num_rounds = params.get('num_boost_round', 300)

    model = lgb.train(lgb_params, dtrain, num_boost_round=num_rounds)
    pred = model.predict(X_test[valid_te])

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_test[valid_te], pred)


# ================================================================
# 模式 4: 回测参数 GPU 加速优化
# ================================================================

def _vectorized_backtest(close, ss, bs, params):
    """
    向量化回测: 用 NumPy 数组替代逐 bar 循环。
    返回 Sharpe ratio。

    逻辑:
      - bs >= long_threshold → 开多
      - ss >= short_threshold → 开空
      - SL/TP 基于 entry price
      - 同一时间只持一个仓位
    """
    n = len(close)
    long_thresh = params['long_threshold']
    short_thresh = params['short_threshold']
    long_sl = params['long_sl']
    long_tp = params['long_tp']
    short_sl = params['short_sl']
    short_tp = params['short_tp']
    leverage = params['leverage']
    fee_rate = 0.0008  # 手续费

    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    equity = 10000.0
    equity_curve = np.empty(n)
    equity_curve[0] = equity

    for i in range(1, n):
        p = close[i]

        if position == 1:
            pnl_pct = (p / entry_price - 1) * leverage
            if pnl_pct <= long_sl or pnl_pct >= long_tp:
                equity *= (1 + pnl_pct - fee_rate)
                position = 0
        elif position == -1:
            pnl_pct = (1 - p / entry_price) * leverage
            if pnl_pct <= short_sl or pnl_pct >= short_tp:
                equity *= (1 + pnl_pct - fee_rate)
                position = 0

        if position == 0:
            if bs[i] >= long_thresh:
                position = 1
                entry_price = p
                equity *= (1 - fee_rate)
            elif ss[i] >= short_thresh:
                position = -1
                entry_price = p
                equity *= (1 - fee_rate)

        equity_curve[i] = equity

    # 平掉剩余仓位
    if position == 1:
        pnl_pct = (close[-1] / entry_price - 1) * leverage
        equity *= (1 + pnl_pct - fee_rate)
    elif position == -1:
        pnl_pct = (1 - close[-1] / entry_price) * leverage
        equity *= (1 + pnl_pct - fee_rate)

    # Sharpe
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10 or returns.std() < 1e-10:
        return -10.0
    sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24)  # 年化
    return float(sharpe) if np.isfinite(sharpe) else -10.0


def train_backtest_optuna(timeframes: List[str] = None, n_trials: int = 500):
    """用 Optuna 优化回测参数 (向量化 + GPU/CPU 加速)"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("Optuna 未安装")
        return {}

    from signal_core import compute_signals_six, calc_fusion_score_six_batch
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages

    timeframes = timeframes or ['1h', '4h']
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"回测参数优化 (向量化) — {SYMBOL}/{tf} ({n_trials} trials)")
        log.info(f"{'='*60}")

        t0 = time.time()

        df = load_klines_local(SYMBOL, tf)
        df = add_all_indicators(df)
        df = add_moving_averages(df)

        # 加载所有 TF 数据
        data_all = {tf: df}
        signals = compute_signals_six(df, tf, data_all)

        sig_time = time.time() - t0
        log.info(f"信号计算: {sig_time:.1f}s")

        # 批量计算 ss/bs
        default_config = {
            'fusion_mode': 'c6_veto_4',
            'weights': {'div': 1.0, 'ma': 1.0, 'cs': 1.0, 'bb': 1.0, 'vp': 1.0, 'kdj': 1.0},
        }
        score_dict, ordered_ts = calc_fusion_score_six_batch(
            signals, df, default_config, warmup=60)

        # 最近 120 天
        trade_days = 120
        cutoff = df.index[-1] - pd.Timedelta(days=trade_days)
        timestamps = [t for t in ordered_ts if t >= cutoff and t in score_dict]

        close_arr = np.array([df.loc[t, 'close'] for t in timestamps], dtype=np.float64)
        ss_arr = np.array([score_dict[t][0] for t in timestamps], dtype=np.float64)
        bs_arr = np.array([score_dict[t][1] for t in timestamps], dtype=np.float64)
        log.info(f"回测数据: {len(timestamps)} 条 ({timestamps[0]} ~ {timestamps[-1]})")

        t_optuna = time.time()

        def objective(trial):
            params = {
                'long_threshold': trial.suggest_int('long_threshold', 25, 55),
                'short_threshold': trial.suggest_int('short_threshold', 15, 45),
                'long_sl': trial.suggest_float('long_sl', -0.20, -0.04),
                'long_tp': trial.suggest_float('long_tp', 0.10, 0.60),
                'short_sl': trial.suggest_float('short_sl', -0.35, -0.10),
                'short_tp': trial.suggest_float('short_tp', 0.20, 1.00),
                'leverage': trial.suggest_int('leverage', 2, 8),
            }
            return _vectorized_backtest(close_arr, ss_arr, bs_arr, params)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        optuna_time = time.time() - t_optuna
        total_time = time.time() - t0

        best = study.best_trial
        all_results[tf] = {
            'best_sharpe': round(best.value, 4),
            'best_params': best.params,
            'n_trials': n_trials,
            'signal_time_sec': round(sig_time, 1),
            'optuna_time_sec': round(optuna_time, 1),
            'total_time_sec': round(total_time, 1),
            'trials_per_sec': round(n_trials / optuna_time, 1),
        }
        log.info(f"最优 Sharpe: {best.value:.4f}")
        log.info(f"最优参数: {json.dumps(best.params, indent=2)}")
        log.info(f"速度: {n_trials/optuna_time:.0f} trials/sec "
                 f"(Optuna {optuna_time:.1f}s, 信号 {sig_time:.1f}s)")

    save_results('backtest_optuna', all_results)
    return all_results


# ================================================================
# 工具函数
# ================================================================

def save_results(name: str, results: Dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'{name}_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"结果已保存: {path}")


def print_gpu_status():
    """打印 GPU 状态信息"""
    gpu = detect_gpu()
    log.info("=" * 60)
    log.info("GPU 状态")
    log.info("=" * 60)
    if gpu.get('torch_cuda'):
        log.info(f"  GPU:  {gpu.get('gpu_name', 'unknown')}")
        log.info(f"  显存: {gpu.get('gpu_mem_gb', 0):.0f} GB")
        try:
            import torch
            log.info(f"  BF16: {torch.cuda.is_bf16_supported()}")
        except Exception:
            pass
    else:
        log.info("  无 GPU，使用 CPU 模式")
    log.info(f"  LightGBM GPU: {gpu.get('lgb_gpu', False)}")


# ================================================================
# 模式 5: TFT (Temporal Fusion Transformer) 训练
# ================================================================

def train_tft(timeframes: List[str] = None):
    """训练 TFT 模型 — 利用 H800 GPU 的大显存"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    timeframes = timeframes or ['1h']
    all_results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"训练 TFT — {SYMBOL}/{tf}")
        log.info(f"{'='*60}")

        t0 = time.time()
        features, labels_df = prepare_features(SYMBOL, tf)

        # 添加跨资产特征
        features = _add_cross_asset_features(features, tf)
        log.info(f"含跨资产特征: {features.shape[1]} 维")

        result = _train_tft_single(features, labels_df, tf, device, use_bf16)
        result['elapsed_sec'] = round(time.time() - t0, 1)
        all_results[tf] = result
        log.info(f"{tf} TFT 完成: {result}")

    save_results('tft_training', all_results)
    return all_results


def _add_cross_asset_features(features: pd.DataFrame, interval: str) -> pd.DataFrame:
    """添加跨资产特征 (BTC/SOL/BNB 对 ETH 的影响)"""
    cross_symbols = ['BTCUSDT', 'SOLUSDT', 'BNBUSDT']
    added = 0

    for sym in cross_symbols:
        try:
            df_cross = load_klines_local(sym, interval)
            if df_cross is None or len(df_cross) < 100:
                continue

            prefix = sym[:3].lower()  # btc, sol, bnb
            close_cross = df_cross['close'].reindex(features.index, method='ffill')

            # 1. 收益率
            for p in [1, 5, 21]:
                features[f'{prefix}_ret_{p}'] = close_cross.pct_change(p)
                added += 1

            # 2. 与 ETH 的滚动相关性
            if 'ret_1' in features.columns:
                eth_ret = features['ret_1']
                cross_ret = close_cross.pct_change(1)
                for w in [20, 60]:
                    features[f'{prefix}_eth_corr_{w}'] = eth_ret.rolling(w).corr(cross_ret)
                    added += 1

            # 3. 相对强度 (ETH vs cross)
            eth_close = features.index.map(
                lambda dt: features['ret_5'].get(dt, 0) if 'ret_5' in features.columns else 0
            )
            features[f'{prefix}_rel_strength'] = features.get('ret_5', 0) - close_cross.pct_change(5)
            added += 1

            # 4. 波动率比
            cross_vol = close_cross.pct_change(1).rolling(20).std()
            if 'hvol_20' in features.columns:
                features[f'{prefix}_vol_ratio'] = features['hvol_20'] / (cross_vol + 1e-10)
                added += 1

        except Exception as e:
            log.warning(f"跨资产特征 {sym} 失败: {e}")

    # 填充 NaN
    new_cols = [c for c in features.columns if any(c.startswith(p) for p in ['btc_', 'sol_', 'bnb_'])]
    features[new_cols] = features[new_cols].fillna(0)
    log.info(f"跨资产特征: +{added} 维 ({', '.join(cross_symbols)})")
    return features


def _train_tft_single(features, labels_df, tf, device, use_bf16):
    """单周期 TFT 训练"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    SEQ_LEN = 96
    D_MODEL = 64       # 较小的模型，防止过拟合
    N_HEADS = 4
    D_FF = 128
    N_LAYERS = 2
    BATCH_SIZE = 256
    EPOCHS = 60
    LR = 5e-4
    PATIENCE = 15

    input_dim = features.shape[1]
    n = len(features)

    # 标准化
    feat_values = features.values.astype(np.float32)
    train_end = int(n * 0.6)
    feat_mean = np.nanmean(feat_values[:train_end], axis=0)
    feat_std = np.nanstd(feat_values[:train_end], axis=0) + 1e-8
    feat_values = (feat_values - feat_mean) / feat_std
    feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=3.0, neginf=-3.0)

    # 二分类标签 (profitable_long_5)
    label_values = labels_df['profitable_long_5'].values.astype(np.float32)

    # 构建序列
    def make_sequences(start, end):
        X, y = [], []
        for i in range(start + SEQ_LEN, end):
            X.append(feat_values[i - SEQ_LEN:i])
            y.append(label_values[i])
        if not X:
            return None, None
        return np.array(X), np.array(y)

    val_end = int(n * 0.8)
    X_train, y_train = make_sequences(0, train_end)
    X_val, y_val = make_sequences(train_end, val_end)
    X_test, y_test = make_sequences(val_end, n)

    if X_train is None or len(X_train) < 100:
        return {'error': 'insufficient_data'}

    log.info(f"序列数据: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 简化版 TFT (不用 VSN 避免 73 个 GRN 太慢，改用高效版)
    class EfficientTFT(nn.Module):
        """高效 TFT: 用单个投影替代逐特征 VSN"""
        def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, dropout=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # LSTM encoder
            self.lstm = nn.LSTM(d_model, d_model, n_layers,
                                batch_first=True, dropout=dropout if n_layers > 1 else 0)
            self.lstm_norm = nn.LayerNorm(d_model)

            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation='gelu', batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.transformer_norm = nn.LayerNorm(d_model)

            # Temporal attention pooling
            self.attn_pool = nn.Linear(d_model, 1)

            # Output
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, 1),  # binary logit
            )

        def forward(self, x):
            # x: (B, T, F)
            h = self.input_proj(x)              # (B, T, d_model)
            lstm_out, _ = self.lstm(h)          # (B, T, d_model)
            h = self.lstm_norm(lstm_out + h)    # residual

            h = self.transformer(h)             # (B, T, d_model)
            h = self.transformer_norm(h)

            # Attention pooling over time
            attn_w = torch.softmax(self.attn_pool(h), dim=1)  # (B, T, 1)
            context = (attn_w * h).sum(dim=1)   # (B, d_model)

            return self.classifier(context).squeeze(-1)  # (B,) logits

    model = EfficientTFT(input_dim, D_MODEL, N_HEADS, D_FF, N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"TFT 参数量: {n_params:,} ({n_params/1e6:.1f}M)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device) if X_test is not None else None

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_auc = 0.5
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        with torch.amp.autocast('cuda', enabled=use_bf16):
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(X_val_t)).cpu().numpy()
            val_true = y_val
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_true, val_pred)
        except Exception:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'tft_{tf}.pt'))
        else:
            no_improve += 1

        if epoch % 5 == 0 or no_improve == 0:
            log.info(f"  Epoch {epoch:3d}: loss={epoch_loss/len(train_loader):.4f} "
                     f"val_AUC={val_auc:.4f} best={best_val_auc:.4f}")

        if no_improve >= PATIENCE:
            log.info(f"  Early stopping at epoch {epoch}")
            break

    # Test
    test_auc = 0.5
    if X_test_t is not None:
        best_path = os.path.join(MODEL_DIR, f'tft_{tf}.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            test_pred = torch.sigmoid(model(X_test_t)).cpu().numpy()
        try:
            test_auc = roc_auc_score(y_test, test_pred)
        except Exception:
            pass

    # 保存元数据
    meta = {
        'input_dim': input_dim,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'seq_len': SEQ_LEN,
        'n_params': n_params,
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
        'feature_names': list(features.columns),
    }
    with open(os.path.join(MODEL_DIR, f'tft_{tf}.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return {
        'best_val_auc': round(best_val_auc, 4),
        'test_auc': round(test_auc, 4),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test) if X_test is not None else 0,
        'epochs_trained': epoch + 1,
        'input_dim': input_dim,
        'n_params': n_params,
        'bf16': use_bf16,
    }


# ================================================================
# 模式 6: 跨资产增强 LightGBM 训练
# ================================================================

def train_cross_asset(timeframes: List[str] = None):
    """用跨资产特征重新训练 LightGBM"""
    import lightgbm as lgb

    timeframes = timeframes or ['1h']
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"跨资产增强训练 — {SYMBOL}/{tf}")
        log.info(f"{'='*60}")

        t0 = time.time()
        features, labels_df = prepare_features(SYMBOL, tf)
        features = _add_cross_asset_features(features, tf)

        label = labels_df['profitable_long_5']
        n = len(features)

        # 时间分割
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        purge = 24

        X_train = features.iloc[:train_end]
        y_train = label.iloc[:train_end]
        X_val = features.iloc[train_end + purge:val_end]
        y_val = label.iloc[train_end + purge:val_end]
        X_test = features.iloc[val_end + purge:]
        y_test = label.iloc[val_end + purge:]

        # 过滤 NaN
        for df_pair in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
            pass  # already clean from prepare_features

        valid_tr = y_train.notna()
        valid_va = y_val.notna()
        valid_te = y_test.notna()
        X_train, y_train = X_train[valid_tr], y_train[valid_tr]
        X_val, y_val = X_val[valid_va], y_val[valid_va]
        X_test, y_test = X_test[valid_te], y_test[valid_te]

        log.info(f"训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
        log.info(f"特征: {features.shape[1]} (含跨资产)")

        # Optuna 最优参数
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 34,
            'learning_rate': 0.0102,
            'feature_fraction': 0.573,
            'bagging_fraction': 0.513,
            'bagging_freq': 3,
            'min_child_samples': 56,
            'lambda_l1': 0.0114,
            'lambda_l2': 0.2146,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        from sklearn.metrics import roc_auc_score
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_auc = roc_auc_score(y_val, val_pred)
        test_auc = roc_auc_score(y_test, test_pred)

        log.info(f"验证 AUC: {val_auc:.4f}, 测试 AUC: {test_auc:.4f}")

        # 保存
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'lgb_cross_asset_{tf}.txt')
        model.save_model(model_path)

        # 特征重要性 (top 20)
        imp = dict(zip(features.columns, model.feature_importance(importance_type='gain').tolist()))
        top_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
        log.info("Top 20 特征:")
        for name, score in top_feats:
            log.info(f"  {name}: {score:.1f}")

        meta = {
            'feature_names': list(features.columns),
            'feature_importance': imp,
            'val_auc': round(val_auc, 4),
            'test_auc': round(test_auc, 4),
        }
        with open(model_path + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        result = {
            'val_auc': round(val_auc, 4),
            'test_auc': round(test_auc, 4),
            'n_features': features.shape[1],
            'elapsed_sec': round(time.time() - t0, 1),
        }
        all_results[tf] = result

    save_results('cross_asset_training', all_results)
    return all_results


# ================================================================
# 模式 7: 增量 Walk-Forward (warm-start)
# ================================================================

def train_incremental_wf(timeframes: List[str] = None):
    """增量 Walk-Forward: warm-start 复用上一折模型，大幅加速"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    timeframes = timeframes or ['15m']
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"增量 Walk-Forward — {SYMBOL}/{tf}")
        log.info(f"{'='*60}")

        t0 = time.time()
        features, labels_df = prepare_features(SYMBOL, tf)
        label = labels_df['profitable_long_5']

        n = len(features)
        TRAIN_WINDOW = 2400
        VAL_WINDOW = 600
        PURGE = 24
        STEP = 120  # 每 120 bar 前进一步
        RETRAIN_INTERVAL = 30  # 每 30 折完全重训一次

        params = {
            'objective': 'binary', 'metric': 'auc',
            'boosting_type': 'gbdt', 'num_leaves': 34,
            'learning_rate': 0.01, 'feature_fraction': 0.573,
            'bagging_fraction': 0.513, 'bagging_freq': 3,
            'min_child_samples': 56, 'lambda_l1': 0.0114,
            'lambda_l2': 0.2146, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
        }

        start_idx = TRAIN_WINDOW + VAL_WINDOW + PURGE
        cursor = start_idx
        fold = 0
        prev_model = None
        val_aucs = []
        predictions = pd.Series(index=features.index, dtype=float)
        predictions[:] = np.nan

        while cursor < n:
            train_end = cursor - PURGE - VAL_WINDOW
            train_start = max(0, train_end - TRAIN_WINDOW)
            val_start = train_end + PURGE
            val_end = cursor
            test_start = cursor
            test_end = min(n, cursor + STEP)

            X_tr = features.iloc[train_start:train_end]
            y_tr = label.iloc[train_start:train_end]
            X_va = features.iloc[val_start:val_end]
            y_va = label.iloc[val_start:val_end]
            X_te = features.iloc[test_start:test_end]

            valid_tr = y_tr.notna()
            valid_va = y_va.notna()
            X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
            X_va, y_va = X_va[valid_va], y_va[valid_va]

            if len(X_tr) < 500:
                cursor += STEP
                continue

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            # 决定是否完全重训
            if fold % RETRAIN_INTERVAL == 0 or prev_model is None:
                # 完全重训
                model = lgb.train(
                    params, dtrain, num_boost_round=356,
                    valid_sets=[dval], valid_names=['val'],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
            else:
                # 增量训练: 在前一个模型基础上继续 boost 50 轮
                model = lgb.train(
                    params, dtrain, num_boost_round=50,
                    valid_sets=[dval], valid_names=['val'],
                    init_model=prev_model,
                    callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)],
                )

            prev_model = model

            # 验证 AUC
            try:
                va_pred = model.predict(X_va)
                va_auc = roc_auc_score(y_va, va_pred)
            except Exception:
                va_auc = 0.5
            val_aucs.append(va_auc)

            # OOS 预测
            te_pred = model.predict(X_te)
            predictions.iloc[test_start:test_end] = te_pred

            if fold % 50 == 0:
                log.info(f"  Fold {fold}: val_AUC={va_auc:.4f} "
                         f"test={test_end-test_start} "
                         f"[{features.index[test_start].strftime('%m-%d')} ~ "
                         f"{features.index[min(test_end-1, n-1)].strftime('%m-%d')}] "
                         f"{'[RETRAIN]' if fold % RETRAIN_INTERVAL == 0 else '[INCR]'}")

            fold += 1
            cursor += STEP

        elapsed = time.time() - t0
        avg_auc = float(np.mean(val_aucs)) if val_aucs else 0.5

        # 保存预测
        result_df = pd.DataFrame({
            'bull_prob': predictions,
            'ml_long_score': predictions * 100,
            'ml_short_score': (1 - predictions) * 100,
        }, index=features.index)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_df.to_parquet(os.path.join(RESULTS_DIR, f'incr_wf_{tf}.parquet'))

        result = {
            'total_folds': fold,
            'avg_val_auc': round(avg_auc, 4),
            'min_val_auc': round(float(np.min(val_aucs)), 4) if val_aucs else 0,
            'max_val_auc': round(float(np.max(val_aucs)), 4) if val_aucs else 0,
            'elapsed_sec': round(elapsed, 1),
            'timeframe': tf,
            'samples': n,
            'features': features.shape[1],
            'retrain_folds': fold // RETRAIN_INTERVAL + 1,
        }
        all_results[tf] = result
        log.info(f"\n增量 WF {tf}: {fold} folds, avg AUC={avg_auc:.4f}, "
                 f"{elapsed:.0f}s (vs 标准WF ~{elapsed*5:.0f}s 预估)")

    save_results('incremental_wf', all_results)
    return all_results


# ================================================================
# ================================================================
# 模式 7: PPO 仓位优化 (强化学习)
# ================================================================

def train_ppo_position(timeframes: List[str] = None, total_steps: int = 200_000):
    """
    用 PPO 训练仓位优化 agent。

    观测: [bull_prob, regime_conf, q50, q05, q95, position, unrealized_pnl, drawdown]
    动作: [position_scale] (连续 0~1, 表示仓位比例)
    奖励: 风险调整收益 (return / max(vol, 0.01))
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    timeframes = timeframes or ['1h']
    tf = timeframes[0]

    log.info(f"\n{'='*60}")
    log.info(f"PPO 仓位优化 — {SYMBOL}/{tf} ({total_steps} steps)")
    log.info(f"{'='*60}")

    t0 = time.time()

    # 准备数据
    from signal_core import compute_signals_six, calc_fusion_score_six_batch
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages

    df = load_klines_local(SYMBOL, tf)
    df = add_all_indicators(df)
    df = add_moving_averages(df)
    data_all = {tf: df}
    signals = compute_signals_six(df, tf, data_all)

    default_config = {
        'fusion_mode': 'c6_veto_4',
        'weights': {'div': 1.0, 'ma': 1.0, 'cs': 1.0, 'bb': 1.0, 'vp': 1.0, 'kdj': 1.0},
    }
    score_dict, ordered_ts = calc_fusion_score_six_batch(
        signals, df, default_config, warmup=60)

    timestamps = [t for t in ordered_ts if t in score_dict]
    close_arr = np.array([df.loc[t, 'close'] for t in timestamps])
    ss_arr = np.array([score_dict[t][0] for t in timestamps])
    bs_arr = np.array([score_dict[t][1] for t in timestamps])

    # 添加额外特征
    returns_arr = np.diff(close_arr, prepend=close_arr[0]) / np.maximum(close_arr, 1)
    vol_arr = pd.Series(returns_arr).rolling(20).std().fillna(0.01).values

    log.info(f"训练数据: {len(timestamps)} 条")

    class TradingEnv(gym.Env):
        """简化交易环境: agent 决定仓位大小"""

        def __init__(self, close, ss, bs, vol, episode_len=500):
            super().__init__()
            self.close = close
            self.ss = ss
            self.bs = bs
            self.vol = vol
            self.n = len(close)
            self.episode_len = min(episode_len, self.n - 10)

            # 动作: 仓位方向和大小 [-1, 1]
            # -1 = full short, 0 = flat, 1 = full long
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

            # 观测: [ss, bs, net_score, vol, position, unrealized_pnl, drawdown, time_frac]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

            self.reset()

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            max_start = self.n - self.episode_len - 1
            self.start = self.np_random.integers(0, max(1, max_start))
            self.step_idx = 0
            self.position = 0.0
            self.equity = 1.0
            self.peak_equity = 1.0
            self.entry_price = 0.0
            return self._obs(), {}

        def _obs(self):
            i = self.start + self.step_idx
            ss = self.ss[i] / 100.0
            bs = self.bs[i] / 100.0
            net = (bs - ss)
            vol = self.vol[i] * 10
            dd = (self.peak_equity - self.equity) / max(self.peak_equity, 0.01)
            unreal = 0.0
            if self.position != 0 and self.entry_price > 0:
                unreal = (self.close[i] / self.entry_price - 1) * np.sign(self.position) * 5
            time_frac = self.step_idx / self.episode_len
            return np.array([ss, bs, net, vol, self.position, unreal, dd, time_frac],
                            dtype=np.float32)

        def step(self, action):
            target_pos = float(np.clip(action[0], -1, 1))
            i = self.start + self.step_idx
            price = self.close[i]

            # 计算 PnL
            if self.position != 0 and self.entry_price > 0:
                ret = (price / self.entry_price - 1) * self.position * 5  # 5x leverage
                pnl = ret * abs(self.position)
            else:
                pnl = 0

            # 交易成本
            pos_change = abs(target_pos - self.position)
            cost = pos_change * 0.001  # 0.1% 单边手续费

            self.equity += pnl - cost
            self.peak_equity = max(self.peak_equity, self.equity)

            # 更新仓位
            if abs(target_pos) > 0.05:
                self.position = target_pos
                self.entry_price = price
            else:
                self.position = 0.0
                self.entry_price = 0.0

            # 奖励: 风险调整收益
            reward = (pnl - cost) / max(self.vol[i], 0.005)

            # 惩罚过大回撤
            dd = (self.peak_equity - self.equity) / max(self.peak_equity, 0.01)
            if dd > 0.15:
                reward -= 0.5

            # 爆仓检测
            self.step_idx += 1
            terminated = self.equity <= 0.2 or self.step_idx >= self.episode_len
            truncated = False

            return self._obs(), float(reward), terminated, truncated, {}

    # 创建向量化环境
    def make_env():
        return TradingEnv(close_arr, ss_arr, bs_arr, vol_arr, episode_len=500)

    env = DummyVecEnv([make_env for _ in range(4)])

    # 训练 PPO
    log.info("训练 PPO agent...")
    model = PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device='cuda' if np.lib.NumpyVersion(np.__version__) else 'auto',
    )
    model.learn(total_timesteps=total_steps)

    # 评估
    log.info("评估 PPO agent...")
    eval_env = TradingEnv(close_arr, ss_arr, bs_arr, vol_arr,
                          episode_len=min(2000, len(close_arr) - 10))
    eval_env.start = max(0, len(close_arr) - 2010)
    eval_env.step_idx = 0
    eval_env.position = 0.0
    eval_env.equity = 1.0
    eval_env.peak_equity = 1.0
    eval_env.entry_price = 0.0

    obs, _ = eval_env._obs(), {}
    positions = []
    equities = [1.0]

    for _ in range(min(2000, len(close_arr) - eval_env.start - 1)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        positions.append(float(action[0]))
        equities.append(eval_env.equity)
        if done:
            break

    final_eq = equities[-1]
    max_dd = max((max(equities[:i+1]) - equities[i]) / max(equities[:i+1])
                 for i in range(len(equities)))
    avg_pos = np.mean(np.abs(positions))

    log.info(f"  最终权益: {final_eq:.4f} ({(final_eq-1)*100:+.1f}%)")
    log.info(f"  最大回撤: {max_dd:.1%}")
    log.info(f"  平均仓位: {avg_pos:.2f}")
    log.info(f"  持仓步数: {sum(1 for p in positions if abs(p) > 0.05)}/{len(positions)}")

    # 保存
    save_path = os.path.join(MODEL_DIR, 'ppo_position_agent')
    model.save(save_path)
    meta = {
        'model': 'PPO',
        'tf': tf,
        'total_steps': total_steps,
        'final_equity': round(final_eq, 4),
        'max_drawdown': round(max_dd, 4),
        'avg_position': round(avg_pos, 4),
        'trained_at': datetime.now().isoformat(),
    }
    with open(save_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"  PPO 模型保存: {save_path} ({elapsed:.1f}s)")

    return meta


# ================================================================
# 模式 8: ONNX 导出 + 推理加速
# ================================================================

def export_onnx_models():
    """
    将 PyTorch 模型导出为 ONNX 格式，并用 ONNX Runtime GPU 做推理加速。
    对比 PyTorch vs ONNX Runtime 推理延迟。
    """
    import torch
    import torch.nn as nn

    log.info(f"\n{'='*60}")
    log.info(f"ONNX 模型导出 + 推理加速对比")
    log.info(f"{'='*60}")

    results = {}

    # 1. LSTM 方向预测模型
    lstm_path = os.path.join(MODEL_DIR, 'lstm_1h.pt')
    if os.path.exists(lstm_path):
        try:
            ckpt = torch.load(lstm_path, map_location='cpu', weights_only=False)
            meta_path = os.path.join(MODEL_DIR, 'lstm_1h_meta.json')
            lstm_meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as _mf:
                        lstm_meta = json.load(_mf)
                except Exception as meta_err:
                    log.warning(f"  LSTM meta 读取失败，回退权重推断: {meta_err}")
            # 从 state_dict 推断维度
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state = ckpt['state_dict']
                input_dim = int(lstm_meta.get('input_dim', ckpt.get('input_dim', state['lstm.weight_ih_l0'].shape[1])))
                hidden_dim = int(lstm_meta.get('hidden_dim', ckpt.get('hidden_dim', state['lstm.weight_hh_l0'].shape[1])))
            else:
                state = ckpt  # raw state_dict
                input_dim = int(lstm_meta.get('input_dim', state['lstm.weight_ih_l0'].shape[1]))
                hidden_dim = int(lstm_meta.get('hidden_dim', state['lstm.weight_hh_l0'].shape[1]))
            num_layers = int(lstm_meta.get('num_layers', 2))
            dropout = float(lstm_meta.get('dropout', 0.3))
            seq_len = int(lstm_meta.get('seq_len', 48))

            # 检测是否为 Multi-Horizon 模型 (head_5h.* keys vs classifier.* keys)
            is_multi_horizon = any('head_5h' in k for k in state)

            if is_multi_horizon:
                # 读取 meta 文件确定最佳预测头
                best_head = str(lstm_meta.get('best_head', '5h')).lower()
                if best_head not in ('5h', '12h', '24h'):
                    best_head = '5h'
                log.info(f"  检测到 Multi-Horizon LSTM，使用最佳预测头: {best_head}")

                clf_in = hidden_dim * 2
                clf_hid = state.get(f'head_{best_head}.0.weight',
                                    torch.zeros(64, clf_in)).shape[0]

                class LSTMDirection(nn.Module):
                    def __init__(self, in_dim, hid_dim):
                        super().__init__()
                        self.lstm = nn.LSTM(
                            in_dim, hid_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
                        self.attn_fc = nn.Linear(hid_dim * 2, 1)
                        self.classifier = nn.Sequential(
                            nn.Linear(hid_dim * 2, clf_hid),
                            nn.GELU(),
                            nn.Dropout(0.2),
                            nn.Linear(clf_hid, 1),
                        )

                    def forward(self, x):
                        out, _ = self.lstm(x)
                        w = torch.softmax(self.attn_fc(out), dim=1)
                        ctx = (out * w).sum(dim=1)
                        return self.classifier(ctx)

                # 重映射 state dict: head_{best_head}.* → classifier.*
                remapped = {}
                for k, v in state.items():
                    if k.startswith(f'head_{best_head}.'):
                        remapped[k.replace(f'head_{best_head}.', 'classifier.')] = v
                    elif not k.startswith('head_'):
                        remapped[k] = v
                model = LSTMDirection(input_dim, hidden_dim)
                model.load_state_dict(remapped)
            else:
                class LSTMDirection(nn.Module):
                    def __init__(self, in_dim, hid_dim):
                        super().__init__()
                        self.lstm = nn.LSTM(
                            in_dim, hid_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
                        self.attn_fc = nn.Linear(hid_dim * 2, 1)
                        clf_in = hid_dim * 2
                        # Infer classifier hidden from state if available
                        clf_hid = state.get('classifier.0.weight',
                                            torch.zeros(64, clf_in)).shape[0]
                        self.classifier = nn.Sequential(
                            nn.Linear(clf_in, clf_hid),
                            nn.GELU(),
                            nn.Dropout(0.2),
                            nn.Linear(clf_hid, 1),
                        )

                    def forward(self, x):
                        out, _ = self.lstm(x)
                        w = torch.softmax(self.attn_fc(out), dim=1)
                        ctx = (out * w).sum(dim=1)
                        return self.classifier(ctx)

                model = LSTMDirection(input_dim, hidden_dim)
                model.load_state_dict(state)
            model.eval()

            dummy = torch.randn(1, seq_len, input_dim)
            onnx_path = os.path.join(MODEL_DIR, 'lstm_1h.onnx')

            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['logit'],
                dynamic_axes={'input': {0: 'batch'}},
                opset_version=17,
            )

            # 性能对比
            import onnxruntime as ort
            n_runs = 100

            # PyTorch CPU
            t0 = time.time()
            for _ in range(n_runs):
                with torch.no_grad():
                    model(dummy)
            pt_cpu = (time.time() - t0) / n_runs * 1000

            # ONNX CPU
            sess_cpu = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            inp = {'input': dummy.numpy()}
            t0 = time.time()
            for _ in range(n_runs):
                sess_cpu.run(None, inp)
            ort_cpu = (time.time() - t0) / n_runs * 1000

            # ONNX GPU
            ort_gpu = None
            try:
                sess_gpu = ort.InferenceSession(
                    onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                # warmup
                for _ in range(10):
                    sess_gpu.run(None, inp)
                t0 = time.time()
                for _ in range(n_runs):
                    sess_gpu.run(None, inp)
                ort_gpu = (time.time() - t0) / n_runs * 1000
            except Exception as e:
                log.warning(f"ONNX GPU 推理失败: {e}")

            result = {
                'pytorch_cpu_ms': round(pt_cpu, 3),
                'onnx_cpu_ms': round(ort_cpu, 3),
                'speedup_cpu': round(pt_cpu / ort_cpu, 2) if ort_cpu > 0 else 0,
            }
            if ort_gpu is not None:
                result['onnx_gpu_ms'] = round(ort_gpu, 3)
                result['speedup_gpu'] = round(pt_cpu / ort_gpu, 2) if ort_gpu > 0 else 0

            results['lstm_1h'] = result
            log.info(f"  LSTM 1h: PyTorch={pt_cpu:.2f}ms, ONNX CPU={ort_cpu:.2f}ms "
                     f"({pt_cpu/ort_cpu:.1f}x)" +
                     (f", ONNX GPU={ort_gpu:.2f}ms ({pt_cpu/ort_gpu:.1f}x)" if ort_gpu else ""))

            size_pt = os.path.getsize(lstm_path) / 1024
            size_onnx = os.path.getsize(onnx_path) / 1024
            log.info(f"  模型大小: PyTorch={size_pt:.0f}KB, ONNX={size_onnx:.0f}KB")

        except Exception as e:
            log.warning(f"  LSTM ONNX 导出失败: {e}")

    # 2. TFT 模型
    tft_path = os.path.join(MODEL_DIR, 'tft_1h.pt')
    if os.path.exists(tft_path):
        try:
            ckpt = torch.load(tft_path, map_location='cpu', weights_only=False)
            # TFT: 从权重推断维度
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state = ckpt['state_dict']
                input_dim = ckpt.get('input_dim', state['input_proj.0.weight'].shape[1])
                d_model = ckpt.get('d_model', state['input_proj.0.weight'].shape[0])
            else:
                state = ckpt
                input_dim = state['input_proj.0.weight'].shape[1]
                d_model = state['input_proj.0.weight'].shape[0]
            # 读取 meta 文件
            meta_path = tft_path.replace('.pt', '.meta.json')
            tft_meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    tft_meta = json.load(f)
            n_heads = tft_meta.get('n_heads', 4)
            d_ff = tft_meta.get('d_ff', d_model * 2)
            n_layers = tft_meta.get('n_layers', 2)
            seq_len = tft_meta.get('seq_len', 48)

            class EfficientTFT(nn.Module):
                def __init__(self, in_dim, d_m, nh, dff, nl, dropout=0.15):
                    super().__init__()
                    self.input_proj = nn.Sequential(
                        nn.Linear(in_dim, d_m), nn.LayerNorm(d_m),
                        nn.GELU(), nn.Dropout(dropout))
                    self.lstm = nn.LSTM(d_m, d_m, nl, batch_first=True,
                                        dropout=dropout if nl > 1 else 0)
                    self.lstm_norm = nn.LayerNorm(d_m)
                    enc_layer = nn.TransformerEncoderLayer(
                        d_model=d_m, nhead=nh, dim_feedforward=dff,
                        dropout=dropout, activation='gelu', batch_first=True)
                    self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
                    self.transformer_norm = nn.LayerNorm(d_m)
                    self.attn_pool = nn.Linear(d_m, 1)
                    self.classifier = nn.Sequential(
                        nn.Linear(d_m, dff), nn.GELU(),
                        nn.Dropout(dropout), nn.Linear(dff, 1))

                def forward(self, x):
                    x = self.input_proj(x)
                    lstm_out, _ = self.lstm(x)
                    x = self.lstm_norm(x + lstm_out)
                    tf_out = self.transformer(x)
                    x = self.transformer_norm(x + tf_out)
                    w = torch.softmax(self.attn_pool(x), dim=1)
                    pooled = (x * w).sum(dim=1)
                    return self.classifier(pooled)

            model = EfficientTFT(input_dim, d_model, n_heads, d_ff, n_layers)
            model.load_state_dict(state)
            model.eval()

            dummy = torch.randn(1, seq_len, input_dim)
            onnx_path = os.path.join(MODEL_DIR, 'tft_1h.onnx')

            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['logit'],
                dynamic_axes={'input': {0: 'batch'}},
                opset_version=17,
            )

            import onnxruntime as ort
            n_runs = 100

            t0 = time.time()
            for _ in range(n_runs):
                with torch.no_grad():
                    model(dummy)
            pt_cpu = (time.time() - t0) / n_runs * 1000

            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            inp = {'input': dummy.numpy()}
            t0 = time.time()
            for _ in range(n_runs):
                sess.run(None, inp)
            ort_cpu = (time.time() - t0) / n_runs * 1000

            ort_gpu = None
            try:
                sess_gpu = ort.InferenceSession(
                    onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                for _ in range(10):
                    sess_gpu.run(None, inp)
                t0 = time.time()
                for _ in range(n_runs):
                    sess_gpu.run(None, inp)
                ort_gpu = (time.time() - t0) / n_runs * 1000
            except Exception as e:
                log.warning(f"TFT ONNX GPU: {e}")

            result = {
                'pytorch_cpu_ms': round(pt_cpu, 3),
                'onnx_cpu_ms': round(ort_cpu, 3),
                'speedup_cpu': round(pt_cpu / ort_cpu, 2),
            }
            if ort_gpu:
                result['onnx_gpu_ms'] = round(ort_gpu, 3)
                result['speedup_gpu'] = round(pt_cpu / ort_gpu, 2)

            results['tft_1h'] = result
            log.info(f"  TFT 1h: PyTorch={pt_cpu:.2f}ms, ONNX CPU={ort_cpu:.2f}ms "
                     f"({pt_cpu/ort_cpu:.1f}x)" +
                     (f", ONNX GPU={ort_gpu:.2f}ms ({pt_cpu/ort_gpu:.1f}x)" if ort_gpu else ""))

        except Exception as e:
            log.warning(f"  TFT ONNX 导出失败: {e}")

    # 3. MTF Fusion MLP
    mlp_path = os.path.join(MODEL_DIR, 'mtf_fusion_mlp.pt')
    if os.path.exists(mlp_path):
        try:
            ckpt = torch.load(mlp_path, map_location='cpu', weights_only=False)
            input_dim = ckpt['input_dim']
            hidden = ckpt.get('hidden', 64)

            class MTFFusionMLP(nn.Module):
                def __init__(self, in_dim, hid=64):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, hid), nn.LayerNorm(hid),
                        nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(hid, hid // 2), nn.LayerNorm(hid // 2),
                        nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(hid // 2, 1))

                def forward(self, x):
                    return self.net(x).squeeze(-1)

            model = MTFFusionMLP(input_dim, hidden)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()

            dummy = torch.randn(1, input_dim)
            onnx_path = os.path.join(MODEL_DIR, 'mtf_fusion_mlp.onnx')

            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['logit'],
                dynamic_axes={'input': {0: 'batch'}},
                opset_version=17,
            )

            import onnxruntime as ort
            n_runs = 1000

            t0 = time.time()
            for _ in range(n_runs):
                with torch.no_grad():
                    model(dummy)
            pt_cpu = (time.time() - t0) / n_runs * 1000

            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            inp = {'input': dummy.numpy()}
            t0 = time.time()
            for _ in range(n_runs):
                sess.run(None, inp)
            ort_cpu = (time.time() - t0) / n_runs * 1000

            results['mtf_fusion'] = {
                'pytorch_cpu_ms': round(pt_cpu, 3),
                'onnx_cpu_ms': round(ort_cpu, 3),
                'speedup_cpu': round(pt_cpu / ort_cpu, 2),
            }
            log.info(f"  MTF MLP: PyTorch={pt_cpu:.3f}ms, ONNX={ort_cpu:.3f}ms "
                     f"({pt_cpu/ort_cpu:.1f}x)")

        except Exception as e:
            log.warning(f"  MTF MLP ONNX 导出失败: {e}")

    save_results('onnx_export', results)
    return results


# ================================================================
# 模式 9: 在线学习定时重训
# ================================================================

def train_online_retrain(timeframes: List[str] = None):
    """
    在线学习: 检查模型新鲜度 → 增量重训 → 验证 → 自动替换。
    设计为 cron 每日运行: 0 4 * * * python train_gpu.py --mode retrain

    流程:
      1. 加载当前模型的训练元数据 (训练日期、AUC)
      2. 用最新数据增量训练 (warm-start)
      3. 在最近 7 天上验证
      4. 如果新模型 AUC >= 老模型 AUC * 0.98 → 替换
      5. 记录重训日志
    """
    timeframes = timeframes or ['1h']
    tf = timeframes[0]

    log.info(f"\n{'='*60}")
    log.info(f"在线学习定时重训 — {SYMBOL}/{tf}")
    log.info(f"{'='*60}")

    t0 = time.time()
    import lightgbm as lgb_lib

    # 1. 检查当前模型状态
    model_path = os.path.join(MODEL_DIR, 'lgb_direction_model.txt')
    meta_path = model_path + '.meta.json'
    old_meta = {}
    old_model = None

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old_meta = json.load(f)
        log.info(f"  当前模型: AUC={old_meta.get('test_auc', '?')}, "
                 f"训练于={old_meta.get('trained_at', '?')}")
    if os.path.exists(model_path):
        old_model = lgb_lib.Booster(model_file=model_path)

    # 2. 准备数据
    features, labels_df = prepare_features(SYMBOL, tf)
    target = labels_df['profitable_long_5']

    # 如果旧模型存在，使用其特征集（避免维度不匹配）
    if old_meta and 'feature_names' in old_meta:
        old_features = old_meta['feature_names']
        missing = [f for f in old_features if f not in features.columns]
        if missing:
            log.warning(f"  缺失特征 ({len(missing)}): {missing[:5]}...")
        features = features[[f for f in old_features if f in features.columns]]
        log.info(f"  使用旧模型特征集: {len(features.columns)} 维")

    n = len(features)
    n_train = int(n * 0.85)
    n_val = int(n * 0.08)

    X_train = features.iloc[:n_train]
    y_train = target.iloc[:n_train]
    X_val = features.iloc[n_train:n_train + n_val]
    y_val = target.iloc[n_train:n_train + n_val]
    X_test = features.iloc[n_train + n_val:]
    y_test = target.iloc[n_train + n_val:]

    log.info(f"  数据: {n:,} 条, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 3. 增量训练 (warm-start from old model)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 34,
        'learning_rate': 0.005,  # 低 LR 用于微调
        'feature_fraction': 0.573,
        'bagging_fraction': 0.513,
        'bagging_freq': 3,
        'min_child_samples': 56,
        'lambda_l1': 0.0114,
        'lambda_l2': 0.2146,
        'verbose': -1,
    }

    gpu = detect_gpu()
    if gpu.get('lgb_gpu'):
        params['device'] = 'cuda'

    dtrain = lgb_lib.Dataset(X_train, y_train)
    dval = lgb_lib.Dataset(X_val, y_val, reference=dtrain)

    callbacks = [
        lgb_lib.early_stopping(20),
        lgb_lib.log_evaluation(50),
    ]

    new_model = lgb_lib.train(
        params, dtrain,
        num_boost_round=100,  # 增量 100 轮
        init_model=old_model,  # warm-start
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=callbacks,
    )

    # 4. 评估
    from sklearn.metrics import roc_auc_score
    new_pred = new_model.predict(X_test)
    new_auc = roc_auc_score(y_test, new_pred)

    old_auc = old_meta.get('test_auc', 0)
    log.info(f"  新模型 AUC: {new_auc:.4f}, 旧模型 AUC: {old_auc}")

    # 5. 决定是否替换
    retrain_log = {
        'timestamp': datetime.now().isoformat(),
        'tf': tf,
        'old_auc': old_auc,
        'new_auc': round(new_auc, 4),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    threshold = old_auc * 0.98 if old_auc > 0 else 0
    if new_auc >= threshold:
        # 替换模型
        new_model.save_model(model_path)
        new_meta = {
            **old_meta,
            'test_auc': round(new_auc, 4),
            'trained_at': datetime.now().isoformat(),
            'retrain_type': 'incremental',
            'prev_auc': old_auc,
            'num_trees': new_model.num_trees(),
            'feature_names': new_model.feature_name(),
        }
        with open(meta_path, 'w') as f:
            json.dump(new_meta, f, indent=2)
        retrain_log['action'] = 'replaced'
        log.info(f"  模型已替换 ✓ (AUC {old_auc} → {new_auc:.4f})")
    else:
        retrain_log['action'] = 'kept_old'
        log.info(f"  保留旧模型 (新 AUC {new_auc:.4f} < 阈值 {threshold:.4f})")

    # 6. 记录重训日志
    log_path = os.path.join(RESULTS_DIR, 'retrain_log.jsonl')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(retrain_log) + '\n')
    log.info(f"  重训日志: {log_path}")

    elapsed = time.time() - t0
    log.info(f"  耗时: {elapsed:.1f}s")

    return retrain_log


# ================================================================
# 模式 10: 多周期神经融合 (MTF Fusion MLP)
# ================================================================

def train_mtf_fusion(decision_tfs=None):
    """
    训练多周期融合 MLP: 输入各 TF 的 (ss, bs) → 输出方向概率。
    替代 multi_tf_consensus.py 中的硬编码权重。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    decision_tfs = decision_tfs or ['15m', '1h', '4h', '8h', '24h']
    log.info(f"\n{'='*60}")
    log.info(f"多周期神经融合 — TFs: {decision_tfs}")
    log.info(f"{'='*60}")

    t0 = time.time()

    # ── 1. 为每个 TF 计算信号 ──
    from signal_core import compute_signals_six, calc_fusion_score_six_batch
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages

    tf_signals = {}
    common_index = None

    # 加载所有 TF 数据 (compute_signals_six 需要 data_all)
    data_all = {}
    for tf in decision_tfs:
        try:
            df = load_klines_local(SYMBOL, tf)
            df = add_all_indicators(df)
            df = add_moving_averages(df)
            data_all[tf] = df
        except Exception as e:
            log.warning(f"  {tf}: 数据加载失败 - {e}")

    # 计算每个 TF 的信号
    default_config = {
        'fusion_mode': 'c6_veto_4',
        'weights': {'div': 1.0, 'ma': 1.0, 'cs': 1.0, 'bb': 1.0, 'vp': 1.0, 'kdj': 1.0},
    }

    for tf in decision_tfs:
        if tf not in data_all:
            continue
        try:
            df = data_all[tf]
            signals = compute_signals_six(df, tf, data_all)

            # 批量计算 (ss, bs) 序列
            score_dict, ordered_ts = calc_fusion_score_six_batch(
                signals, df, default_config, warmup=60)

            if not score_dict:
                log.warning(f"  {tf}: 无信号分数")
                continue

            # 转为 Series
            timestamps = [t for t in ordered_ts if t in score_dict]
            ss_vals = [score_dict[t][0] for t in timestamps]
            bs_vals = [score_dict[t][1] for t in timestamps]
            ss_series = pd.Series(ss_vals, index=timestamps)
            bs_series = pd.Series(bs_vals, index=timestamps)

            tf_signals[tf] = {
                'ss': ss_series, 'bs': bs_series,
                'close': df['close'],
            }
            log.info(f"  {tf}: {len(ss_series)} 条信号")

            if common_index is None:
                common_index = ss_series.index
            else:
                common_index = common_index.intersection(ss_series.index)
        except Exception as e:
            log.warning(f"  {tf}: 加载失败 - {e}")

    available_tfs = [tf for tf in decision_tfs if tf in tf_signals]
    if len(available_tfs) < 2:
        log.error("可用 TF 不足 2 个，无法训练融合模型")
        return {}

    log.info(f"可用 TF: {available_tfs}, 共同时间范围: {len(common_index)} 条")

    # ── 2. 对齐到共同时间 + 构建特征矩阵 ──
    # 对于不同频率的 TF，用 ffill 对齐到最小粒度
    # 使用 1h 粒度作为基准（平衡样本数和信息量）
    base_tf = '1h'
    if base_tf not in tf_signals:
        base_tf = available_tfs[0]

    base_index = tf_signals[base_tf]['ss'].index
    feature_cols = []
    feature_data = {}

    for tf in available_tfs:
        ss = tf_signals[tf]['ss'].reindex(base_index, method='ffill')
        bs = tf_signals[tf]['bs'].reindex(base_index, method='ffill')
        feature_data[f'{tf}_ss'] = ss
        feature_data[f'{tf}_bs'] = bs
        # 方向净分
        feature_data[f'{tf}_net'] = bs - ss
        # 分数强度
        feature_data[f'{tf}_max'] = np.maximum(ss, bs)
        feature_cols.extend([f'{tf}_ss', f'{tf}_bs', f'{tf}_net', f'{tf}_max'])

    features = pd.DataFrame(feature_data, index=base_index)

    # 添加交叉特征: 大周期和小周期的一致性
    large_tfs = [tf for tf in available_tfs
                 if {'15m': 15, '30m': 30, '1h': 60, '4h': 240, '8h': 480, '24h': 1440}.get(tf, 60) >= 240]
    small_tfs = [tf for tf in available_tfs
                 if {'15m': 15, '30m': 30, '1h': 60, '4h': 240, '8h': 480, '24h': 1440}.get(tf, 60) < 240]

    if large_tfs and small_tfs:
        large_net = features[[f'{tf}_net' for tf in large_tfs]].mean(axis=1)
        small_net = features[[f'{tf}_net' for tf in small_tfs]].mean(axis=1)
        features['large_small_agree'] = (large_net * small_net > 0).astype(float)
        features['large_net_avg'] = large_net
        features['small_net_avg'] = small_net
        feature_cols.extend(['large_small_agree', 'large_net_avg', 'small_net_avg'])

    # 标签: 未来 5 根 K 线收益
    base_close = tf_signals[base_tf]['close'].reindex(base_index)
    fwd_ret = base_close.pct_change(5).shift(-5)
    cost = 0.0015
    label = (fwd_ret > cost * 2).astype(float)  # 1 = profitable long

    # 清理
    valid = features.notna().all(axis=1) & label.notna()
    features = features[valid]
    label = label[valid]

    n = len(features)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train = features.iloc[:n_train].values
    y_train = label.iloc[:n_train].values
    X_val = features.iloc[n_train:n_train + n_val].values
    y_val = label.iloc[n_train:n_train + n_val].values
    X_test = features.iloc[n_train + n_val:].values
    y_test = label.iloc[n_train + n_val:].values

    log.info(f"特征维度: {X_train.shape[1]}, 训练/验证/测试: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    log.info(f"正类比例: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

    # ── 3. 训练 MLP ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]

    class MTFFusionMLP(nn.Module):
        def __init__(self, in_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    # 标准化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std

    train_ds = TensorDataset(
        torch.tensor(X_train_n, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(X_val_n, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = MTFFusionMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Focal loss for imbalanced data
    def focal_loss(pred, target, gamma=2.0, alpha=0.25):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        pt = target * p + (1 - target) * (1 - p)
        focal_weight = alpha * (1 - pt) ** gamma
        return (focal_weight * bce).mean()

    best_val_auc = 0
    patience = 10
    no_improve = 0

    for epoch in range(80):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = focal_loss(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(yb.numpy())

        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds)

        if epoch % 10 == 0 or val_auc > best_val_auc:
            log.info(f"  Epoch {epoch:3d}: loss={total_loss/len(train_loader):.4f}, val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stop at epoch {epoch}, best val AUC={best_val_auc:.4f}")
                break

    # ── 4. 测试 ──
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test_n, dtype=torch.float32).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    test_auc = roc_auc_score(y_test, test_probs)
    log.info(f"\n  Test AUC: {test_auc:.4f} (val: {best_val_auc:.4f})")

    # ── 5. 保存 ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'mtf_fusion_mlp.pt')
    torch.save({
        'state_dict': best_state,
        'input_dim': input_dim,
        'hidden': 64,
        'feature_cols': feature_cols,
        'available_tfs': available_tfs,
        'mean': mean.tolist(),
        'std': std.tolist(),
    }, save_path)

    meta = {
        'model': 'MTFFusionMLP',
        'tfs': available_tfs,
        'input_dim': input_dim,
        'feature_cols': feature_cols,
        'val_auc': round(best_val_auc, 4),
        'test_auc': round(test_auc, 4),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'trained_at': datetime.now().isoformat(),
    }
    meta_path = save_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"  MTF 融合模型保存: {save_path} ({elapsed:.1f}s)")

    return {'val_auc': best_val_auc, 'test_auc': test_auc, 'tfs': available_tfs}


# ================================================================
# 模式 12: Stacking Ensemble Meta-Learner
# ================================================================

def train_stacking_ensemble(timeframes: List[str] = None, min_samples: int = 20000):
    """
    Stacking 集成: 4 个异构基模型的 OOF 预测 → LogisticRegression 元学习器。

    基模型:
      1. LightGBM Direction (73 维, Optuna 参数)
      2. XGBoost (73 维, 保守配置)
      3. LSTM+Attention (73 维, 48-bar 序列)
      4. TFT (94 维含跨资产, 96-bar 序列)

    流程:
      A. 数据准备 + 时间分割
      B. 5-Fold 时序 CV 生成 OOF 预测
      C. 训练 LogisticRegression 元学习器
      D. 全量重训基模型 + 保存
    """
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    timeframes = timeframes or ['1h']
    primary_tf = timeframes[0]
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"Stacking Ensemble — {SYMBOL}/{tf}")
        log.info(f"{'='*60}")

        t0 = time.time()

        # ---- A. 数据准备 ----
        features, labels_df = prepare_features(SYMBOL, tf)
        features_cross = _add_cross_asset_features(features.copy(), tf)

        label = labels_df['profitable_long_5']
        n = len(features)

        # 样本门禁：小样本下 Stacking 容易过拟合，直接跳过避免浪费 H800 资源
        if n < int(min_samples):
            reason = f"insufficient_samples({n}<{int(min_samples)})"
            log.warning(f"跳过 {tf} Stacking: {reason}")
            all_results[tf] = {
                "timeframe": tf,
                "skipped": True,
                "skip_reason": reason,
                "n_samples": n,
                "min_samples_required": int(min_samples),
            }
            continue

        # 时间分割: 70% train, 15% val, 15% test
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        purge = 24

        y_all = label.values.astype(np.float32)

        # 基模型用 73 维特征 (LGB/XGB/LSTM), TFT 用 94 维 (含跨资产)
        feat_73 = features.values.astype(np.float32)
        feat_94 = features_cross.values.astype(np.float32)
        feat_73_cols = list(features.columns)
        feat_94_cols = list(features_cross.columns)

        # 标准化参数改为按 fold 训练窗口计算，避免全局统计量泄漏未来信息。

        log.info(f"数据: {n} 样本, train={train_end}, val={val_end-train_end-purge}, "
                 f"test={n-val_end-purge}")
        log.info(f"特征: 73 维 (LGB/XGB/LSTM) + 94 维 (TFT)")

        # ---- B. 5-Fold OOF 预测 ----
        N_FOLDS = 5
        fold_size = train_end // N_FOLDS

        oof_preds = np.full((train_end, 5), np.nan)  # (n_train, 5 models)
        oof_valid = np.zeros(train_end, dtype=bool)

        log.info(f"\n生成 OOF 预测 ({N_FOLDS} folds, expanding window)...")

        for fold_idx in range(N_FOLDS):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < N_FOLDS - 1 else train_end

            # Expanding window: 用 fold 之前的所有数据训练
            if fold_idx == 0:
                log.info(f"  Fold 0: 跳过 (无训练数据)")
                continue

            tr_end = fold_start
            tr_valid = ~np.isnan(y_all[:tr_end])
            fold_valid = ~np.isnan(y_all[fold_start:fold_end])

            if tr_valid.sum() < 200 or fold_valid.sum() < 50:
                log.info(f"  Fold {fold_idx}: 跳过 (样本不足)")
                continue

            X_tr_73 = feat_73[:tr_end][tr_valid]
            X_tr_94 = feat_94[:tr_end][tr_valid]
            y_tr = y_all[:tr_end][tr_valid]

            X_fold_73 = feat_73[fold_start:fold_end][fold_valid]
            X_fold_94 = feat_94[fold_start:fold_end][fold_valid]

            fold_mean_73 = np.nanmean(X_tr_73, axis=0)
            fold_std_73 = np.nanstd(X_tr_73, axis=0) + 1e-8
            fold_mean_94 = np.nanmean(X_tr_94, axis=0)
            fold_std_94 = np.nanstd(X_tr_94, axis=0) + 1e-8

            log.info(f"  Fold {fold_idx}: train={len(X_tr_73)}, predict={len(X_fold_73)}")

            # --- 基模型 1: LGB ---
            try:
                import lightgbm as lgb
                lgb_params = {
                    'objective': 'binary', 'metric': 'auc',
                    'boosting_type': 'gbdt', 'num_leaves': 34,
                    'learning_rate': 0.0102, 'feature_fraction': 0.573,
                    'bagging_fraction': 0.513, 'bagging_freq': 3,
                    'min_child_samples': 56, 'lambda_l1': 0.0114,
                    'lambda_l2': 0.2146, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
                }
                dtrain = lgb.Dataset(X_tr_73, label=y_tr)
                lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=300)
                lgb_pred = lgb_model.predict(X_fold_73)
                oof_preds[fold_start:fold_end, 0][fold_valid] = lgb_pred
            except Exception as e:
                log.warning(f"  Fold {fold_idx} LGB 失败: {e}")

            # --- 基模型 2: XGBoost ---
            try:
                import xgboost as xgb
                xgb_params = {
                    'objective': 'binary:logistic', 'eval_metric': 'auc',
                    'max_depth': 5, 'learning_rate': 0.01,
                    'subsample': 0.7, 'colsample_bytree': 0.7,
                    'min_child_weight': 10, 'reg_alpha': 0.01,
                    'reg_lambda': 1.0, 'seed': 42, 'verbosity': 0,
                }
                if detect_gpu().get('torch_cuda'):
                    xgb_params['tree_method'] = 'hist'
                    xgb_params['device'] = 'cuda'
                dmat_tr = xgb.DMatrix(X_tr_73, label=y_tr)
                dmat_fold = xgb.DMatrix(X_fold_73)
                xgb_model = xgb.train(xgb_params, dmat_tr, num_boost_round=300)
                xgb_pred = xgb_model.predict(dmat_fold)
                oof_preds[fold_start:fold_end, 1][fold_valid] = xgb_pred
            except Exception as e:
                log.warning(f"  Fold {fold_idx} XGBoost 失败: {e}")

            # --- 基模型 3: LSTM+Attention ---
            try:
                lstm_pred = _stacking_train_lstm_fold(
                    feat_73, y_all, tr_end, fold_start, fold_end,
                    fold_valid, feat_73.shape[1], fold_mean_73, fold_std_73)
                if lstm_pred is not None:
                    oof_preds[fold_start:fold_end, 2][fold_valid] = lstm_pred
            except Exception as e:
                log.warning(f"  Fold {fold_idx} LSTM 失败: {e}")

            # --- 基模型 4: TFT ---
            try:
                tft_pred = _stacking_train_tft_fold(
                    feat_94, y_all, tr_end, fold_start, fold_end,
                    fold_valid, feat_94.shape[1], fold_mean_94, fold_std_94)
                if tft_pred is not None:
                    oof_preds[fold_start:fold_end, 3][fold_valid] = tft_pred
            except Exception as e:
                log.warning(f"  Fold {fold_idx} TFT 失败: {e}")

            # --- 基模型 5: Cross-Asset LGB (94 dims) ---
            try:
                import lightgbm as lgb
                lgb_cross_params = {
                    'objective': 'binary', 'metric': 'auc',
                    'boosting_type': 'gbdt', 'num_leaves': 34,
                    'learning_rate': 0.0102, 'feature_fraction': 0.573,
                    'bagging_fraction': 0.513, 'bagging_freq': 3,
                    'min_child_samples': 56, 'lambda_l1': 0.0114,
                    'lambda_l2': 0.2146, 'verbose': -1, 'n_jobs': -1, 'seed': 43,
                }
                dtrain_cross = lgb.Dataset(X_tr_94, label=y_tr)
                lgb_cross_model = lgb.train(lgb_cross_params, dtrain_cross, num_boost_round=300)
                lgb_cross_pred = lgb_cross_model.predict(X_fold_94)
                oof_preds[fold_start:fold_end, 4][fold_valid] = lgb_cross_pred
            except Exception as e:
                log.warning(f"  Fold {fold_idx} Cross-Asset LGB 失败: {e}")

            oof_valid[fold_start:fold_end] |= fold_valid

        # OOF 汇总
        has_oof = oof_valid & ~np.isnan(y_all[:train_end])
        # 对缺失模型的 OOF 用 0.5 填充 (中性概率)
        oof_filled = oof_preds[:train_end].copy()
        oof_filled[np.isnan(oof_filled)] = 0.5
        has_oof &= np.any(~np.isnan(oof_preds[:train_end]), axis=1)

        oof_X = oof_filled[has_oof]
        oof_y = y_all[:train_end][has_oof]

        model_names = ['LGB', 'XGBoost', 'LSTM', 'TFT', 'CrossAssetLGB']
        log.info(f"\nOOF 汇总: {len(oof_X)} 有效样本")
        for i, name in enumerate(model_names):
            valid_mask = ~np.isnan(oof_preds[:train_end][has_oof, i])
            if valid_mask.sum() > 50:
                auc_i = roc_auc_score(oof_y[valid_mask],
                                      oof_preds[:train_end][has_oof, i][valid_mask])
                log.info(f"  {name} OOF AUC: {auc_i:.4f} ({valid_mask.sum()} 样本)")
            else:
                log.info(f"  {name} OOF: 样本不足")

        # H800-6: OOF 相关性诊断
        log.info(f"\n[H800-6] OOF 诊断:")
        oof_mat = oof_filled[has_oof]
        oof_stds = oof_mat.std(axis=0)
        log.info(f"  OOF 标准差: {dict(zip(model_names, oof_stds.round(4)))}")
        log.info(f"  OOF 互相关矩阵:")
        corr_mat = np.corrcoef(oof_mat.T)
        for i, name in enumerate(model_names):
            corr_str = " ".join([f"{c:6.3f}" for c in corr_mat[i]])
            log.info(f"    {name:15s}: {corr_str}")
        # 检测近常数模型
        low_var_models = [model_names[i] for i, std in enumerate(oof_stds) if std < 0.05]
        if low_var_models:
            log.warning(f"  ⚠️  低方差模型 (std<0.05): {low_var_models} - 考虑移除")

        # 可选附加特征: hvol_20, regime_label
        extra_feat_names = []
        oof_extra = []
        hvol_idx = feat_73_cols.index('hvol_20') if 'hvol_20' in feat_73_cols else -1
        if hvol_idx >= 0:
            oof_extra.append(feat_73[:train_end][has_oof, hvol_idx:hvol_idx+1])
            extra_feat_names.append('hvol_20')

        if oof_extra:
            oof_X = np.hstack([oof_X] + oof_extra)
            log.info(f"  附加特征: {extra_feat_names} → 元学习器输入 {oof_X.shape[1]} 维")

        # ---- C. 训练元学习器 ----
        log.info(f"\n训练 LogisticRegression 元学习器...")

        meta_model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                        max_iter=1000, random_state=42)
        meta_model.fit(oof_X, oof_y)

        # OOF 性能
        oof_meta_pred = meta_model.predict_proba(oof_X)[:, 1]
        oof_meta_auc = roc_auc_score(oof_y, oof_meta_pred)
        log.info(f"  Meta OOF AUC: {oof_meta_auc:.4f}")

        # 诊断: 检查负系数基模型 (L2 正则化可能对高度相关的树模型产生负权重)
        base_coefs = meta_model.coef_[0][:len(model_names)]
        neg_coef_models = [(model_names[i], round(float(base_coefs[i]), 4))
                           for i in range(len(model_names)) if base_coefs[i] < 0]
        if neg_coef_models:
            log.warning(f"  ⚠️  负系数基模型: {neg_coef_models}")
            log.warning(f"     负系数意味着该模型的看多预测反而使元学习器看空。")
            log.warning(f"     可能原因: 两模型高度相关 + L2正则化 → 考虑 C=0.5 或移除该模型")
        for i, (name, _) in enumerate(zip(model_names, base_coefs)):
            log.info(f"  系数 {name}: {base_coefs[i]:.4f}")
        if extra_feat_names:
            extra_coefs = meta_model.coef_[0][len(model_names):]
            for name, coef in zip(extra_feat_names, extra_coefs):
                log.info(f"  系数 {name}: {coef:.4f}")
        log.info(f"  截距: {float(meta_model.intercept_[0]):.4f}")

        # 标签分布诊断 (高偏差警告)
        label_mean = oof_y.mean()
        oof_pred_mean = oof_meta_pred.mean()
        log.info(f"  标签均值: {label_mean:.4f}, 预测均值: {oof_pred_mean:.4f}")
        if abs(oof_pred_mean - label_mean) > 0.05:
            log.warning(f"  ⚠️  元学习器预测存在偏差 ({oof_pred_mean:.3f} vs 标签{label_mean:.3f})，"
                        f"可能导致实盘系统性{'看空' if oof_pred_mean < label_mean else '看多'}")

        # 验证/测试切分（最终评估放到全量基模型保存后，避免估计偏差）
        # 注意: val_auc 使用全量重训基模型 (trained on train+val) 在 val 上评估 → 偏高 (in-sample)
        # test_auc 使用全量重训基模型在 test 上评估 → 真实 holdout
        val_start = train_end + purge
        val_mask = ~np.isnan(y_all[val_start:val_end])
        test_start = val_end + purge
        test_mask = ~np.isnan(y_all[test_start:])
        val_auc = 0.5
        test_auc = 0.5

        # ---- D. 全量重训基模型 + 保存 ----
        log.info(f"\n全量重训基模型 (train+val)...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        full_end = val_end
        full_valid = ~np.isnan(y_all[:full_end])
        X_full_73 = feat_73[:full_end][full_valid]
        X_full_94 = feat_94[:full_end][full_valid]
        y_full = y_all[:full_end][full_valid]
        feat_mean_73 = np.nanmean(X_full_73, axis=0)
        feat_std_73 = np.nanstd(X_full_73, axis=0) + 1e-8
        feat_mean_94 = np.nanmean(X_full_94, axis=0)
        feat_std_94 = np.nanstd(X_full_94, axis=0) + 1e-8

        # 重训 LGB
        try:
            import lightgbm as lgb
            lgb_params = {
                'objective': 'binary', 'metric': 'auc',
                'boosting_type': 'gbdt', 'num_leaves': 34,
                'learning_rate': 0.0102, 'feature_fraction': 0.573,
                'bagging_fraction': 0.513, 'bagging_freq': 3,
                'min_child_samples': 56, 'lambda_l1': 0.0114,
                'lambda_l2': 0.2146, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
            }
            dtrain = lgb.Dataset(X_full_73, label=y_full)
            final_lgb = lgb.train(lgb_params, dtrain, num_boost_round=400)
            final_lgb.save_model(os.path.join(MODEL_DIR, f'stacking_lgb_{tf}.txt'))
            log.info("  LGB 保存完成")
        except Exception as e:
            log.warning(f"  LGB 全量训练失败: {e}")

        # 重训 XGBoost
        try:
            import xgboost as xgb
            xgb_params = {
                'objective': 'binary:logistic', 'eval_metric': 'auc',
                'max_depth': 5, 'learning_rate': 0.01,
                'subsample': 0.7, 'colsample_bytree': 0.7,
                'min_child_weight': 10, 'reg_alpha': 0.01,
                'reg_lambda': 1.0, 'seed': 42, 'verbosity': 0,
            }
            dmat = xgb.DMatrix(X_full_73, label=y_full)
            final_xgb = xgb.train(xgb_params, dmat, num_boost_round=400)
            final_xgb.save_model(os.path.join(MODEL_DIR, f'stacking_xgb_{tf}.json'))
            log.info("  XGBoost 保存完成")
        except Exception as e:
            log.warning(f"  XGBoost 全量训练失败: {e}")

        # 重训 LSTM
        try:
            _stacking_retrain_lstm_full(
                feat_73, y_all, full_end, full_valid, feat_73.shape[1], tf,
                feat_mean_73, feat_std_73
            )
            log.info("  LSTM 保存完成")
        except Exception as e:
            log.warning(f"  LSTM 全量训练失败: {e}")

        # 重训 TFT
        try:
            _stacking_retrain_tft_full(
                feat_94, y_all, full_end, full_valid, feat_94.shape[1], tf,
                feat_mean_94, feat_std_94
            )
            log.info("  TFT 保存完成")
        except Exception as e:
            log.warning(f"  TFT 全量训练失败: {e}")

        # 重训 Cross-Asset LGB
        try:
            import lightgbm as lgb
            lgb_cross_params = {
                'objective': 'binary', 'metric': 'auc',
                'boosting_type': 'gbdt', 'num_leaves': 34,
                'learning_rate': 0.0102, 'feature_fraction': 0.573,
                'bagging_fraction': 0.513, 'bagging_freq': 3,
                'min_child_samples': 56, 'lambda_l1': 0.0114,
                'lambda_l2': 0.2146, 'verbose': -1, 'n_jobs': -1, 'seed': 43,
            }
            dtrain_cross = lgb.Dataset(X_full_94, label=y_full)
            final_lgb_cross = lgb.train(lgb_cross_params, dtrain_cross, num_boost_round=400)
            final_lgb_cross.save_model(os.path.join(MODEL_DIR, f'stacking_lgb_cross_{tf}.txt'))
            log.info("  Cross-Asset LGB 保存完成")
        except Exception as e:
            log.warning(f"  Cross-Asset LGB 全量训练失败: {e}")

        # ---- E. 用最终保存的 5 个基模型评估元学习器 ----
        val_auc = _stacking_evaluate_meta_with_saved_models(
            meta_model=meta_model,
            tf=tf,
            feat_73=feat_73,
            feat_94=feat_94,
            y_all=y_all,
            start=val_start,
            end=val_end,
            valid_mask=val_mask,
            hvol_idx=hvol_idx,
            extra_feat_names=extra_feat_names,
            feat_mean_73=feat_mean_73,
            feat_std_73=feat_std_73,
            feat_mean_94=feat_mean_94,
            feat_std_94=feat_std_94,
        )
        test_auc = _stacking_evaluate_meta_with_saved_models(
            meta_model=meta_model,
            tf=tf,
            feat_73=feat_73,
            feat_94=feat_94,
            y_all=y_all,
            start=test_start,
            end=len(y_all),
            valid_mask=test_mask,
            hvol_idx=hvol_idx,
            extra_feat_names=extra_feat_names,
            feat_mean_73=feat_mean_73,
            feat_std_73=feat_std_73,
            feat_mean_94=feat_mean_94,
            feat_std_94=feat_std_94,
        )
        log.info(f"  验证 AUC(最终基模型): {val_auc:.4f}")
        log.info(f"  测试 AUC(最终基模型): {test_auc:.4f}")

        # 保存元学习器 (周期特定文件名)
        meta_file = f'stacking_meta_{tf}.pkl'
        meta_path = os.path.join(MODEL_DIR, meta_file)
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_model, f)

        # 保存元数据 (含系数，便于 ONNX 导出和 debug)
        meta_coef = meta_model.coef_[0].tolist()
        meta_intercept = float(meta_model.intercept_[0])
        base_model_names = ['lgb', 'xgboost', 'lstm', 'tft', 'cross_asset_lgb'] + extra_feat_names
        stacking_meta = {
            'version': 'stacking_v2',
            'timeframe': tf,
            'trained_at': datetime.now().isoformat(),
            'base_models': ['lgb', 'xgboost', 'lstm', 'tft', 'cross_asset_lgb'],
            'extra_features': extra_feat_names,
            'meta_input_dim': len(base_model_names),
            'meta_coefficients': dict(zip(base_model_names, meta_coef)),
            'meta_intercept': meta_intercept,
            'oof_meta_auc': round(oof_meta_auc, 4),
            'val_auc': round(val_auc, 4),
            'test_auc': round(test_auc, 4),
            'n_oof_samples': int(len(oof_X)),
            'n_folds': N_FOLDS,
            'feature_names_73': feat_73_cols,
            'feature_names_94': feat_94_cols,
            'feat_mean_73': feat_mean_73.tolist(),
            'feat_std_73': feat_std_73.tolist(),
            'feat_mean_94': feat_mean_94.tolist(),
            'feat_std_94': feat_std_94.tolist(),
            'model_files': {
                'lgb': f'stacking_lgb_{tf}.txt',
                'xgboost': f'stacking_xgb_{tf}.json',
                'lstm': f'stacking_lstm_{tf}.pt',
                'tft': f'stacking_tft_{tf}.pt',
                'cross_asset_lgb': f'stacking_lgb_cross_{tf}.txt',
                'meta': meta_file,
            },
            'thresholds': {
                'long_threshold': 0.58,
                'short_threshold': 0.42,
            },
        }
        meta_json_path = os.path.join(MODEL_DIR, f'stacking_meta_{tf}.json')
        with open(meta_json_path, 'w') as f:
            json.dump(stacking_meta, f, indent=2, default=str)

        # 兼容旧推理逻辑：将主周期复制为默认文件名
        if tf == primary_tf:
            shutil.copyfile(meta_path, os.path.join(MODEL_DIR, 'stacking_meta.pkl'))
            shutil.copyfile(meta_json_path, os.path.join(MODEL_DIR, 'stacking_meta.json'))
            log.info(f"  主周期别名已更新: {primary_tf} -> stacking_meta.json/.pkl")

        elapsed = time.time() - t0
        result = {
            'oof_meta_auc': round(oof_meta_auc, 4),
            'val_auc': round(val_auc, 4),
            'test_auc': round(test_auc, 4),
            'n_oof_samples': len(oof_X),
            'elapsed_sec': round(elapsed, 1),
        }
        all_results[tf] = result
        log.info(f"\nStacking 完成: {result}")

    save_results('stacking_ensemble', all_results)
    return all_results


def _stacking_normalize_seq(seq: np.ndarray, feat_mean: np.ndarray, feat_std: np.ndarray) -> np.ndarray:
    """序列标准化并裁剪异常值。"""
    seq_n = (seq - feat_mean) / np.maximum(feat_std, 1e-8)
    seq_n = np.nan_to_num(seq_n, nan=0.0, posinf=3.0, neginf=-3.0)
    return np.clip(seq_n, -5, 5).astype(np.float32)


def _stacking_train_lstm_fold(feat_raw, y_all, tr_end, fold_start, fold_end,
                              fold_valid, input_dim, feat_mean, feat_std,
                              seq_len=48, epochs=20):
    """在单个 fold 上快速训练 LSTM 并返回 fold 预测"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_mean = np.asarray(feat_mean, dtype=np.float32)
    feat_std = np.asarray(feat_std, dtype=np.float32)

    # 构建训练序列（使用该 fold 训练窗口统计量标准化）
    X_seqs, y_seqs = [], []
    for i in range(seq_len, tr_end):
        if not np.isnan(y_all[i]):
            seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
            X_seqs.append(seq)
            y_seqs.append(y_all[i])
    if len(X_seqs) < 100:
        return None

    X_train_t = torch.FloatTensor(np.array(X_seqs)).to(device)
    y_train_t = torch.FloatTensor(np.array(y_seqs)).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=256, shuffle=True)

    class LSTMAttention(nn.Module):
        def __init__(self, in_dim, hidden=128, layers=2, drop=0.3):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, layers,
                                batch_first=True, dropout=drop, bidirectional=True)
            self.attn_fc = nn.Linear(hidden * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            w = torch.softmax(self.attn_fc(out), dim=1)
            ctx = (w * out).sum(dim=1)
            return self.classifier(ctx).squeeze(-1)

    model = LSTMAttention(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # fold 预测
    fold_seqs = []
    fold_indices = []
    for i in range(fold_start, fold_end):
        rel = i - fold_start
        if fold_valid[rel] and i >= seq_len:
            seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
            fold_seqs.append(seq)
            fold_indices.append(rel)

    if not fold_seqs:
        return None

    model.eval()
    with torch.no_grad():
        X_fold_t = torch.FloatTensor(np.array(fold_seqs)).to(device)
        logits = model(X_fold_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    result = np.full(fold_valid.sum(), 0.5)
    # Map fold_indices back to positions in the valid-only array
    valid_positions = np.where(fold_valid)[0]
    pos_map = {v: idx for idx, v in enumerate(valid_positions)}
    for fi, prob in zip(fold_indices, probs):
        if fi in pos_map:
            result[pos_map[fi]] = prob

    return result


def _stacking_train_tft_fold(feat_raw, y_all, tr_end, fold_start, fold_end,
                             fold_valid, input_dim, feat_mean, feat_std,
                             seq_len=96, epochs=15):
    """在单个 fold 上快速训练 TFT 并返回 fold 预测"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_mean = np.asarray(feat_mean, dtype=np.float32)
    feat_std = np.asarray(feat_std, dtype=np.float32)

    # 构建训练序列（使用该 fold 训练窗口统计量标准化）
    X_seqs, y_seqs = [], []
    for i in range(seq_len, tr_end):
        if not np.isnan(y_all[i]):
            seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
            X_seqs.append(seq)
            y_seqs.append(y_all[i])
    if len(X_seqs) < 100:
        return None

    X_train_t = torch.FloatTensor(np.array(X_seqs)).to(device)
    y_train_t = torch.FloatTensor(np.array(y_seqs)).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=256, shuffle=True)

    class EfficientTFT(nn.Module):
        def __init__(self, in_dim, d_model=64, n_heads=4, d_ff=128, n_layers=2, drop=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Dropout(drop))
            self.lstm = nn.LSTM(d_model, d_model, n_layers,
                                batch_first=True, dropout=drop if n_layers > 1 else 0)
            self.lstm_norm = nn.LayerNorm(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=drop, activation='gelu', batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.transformer_norm = nn.LayerNorm(d_model)
            self.attn_pool = nn.Linear(d_model, 1)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(drop),
                nn.Linear(d_ff, 1))

        def forward(self, x):
            h = self.input_proj(x)
            lstm_out, _ = self.lstm(h)
            h = self.lstm_norm(lstm_out + h)
            h = self.transformer(h)
            h = self.transformer_norm(h)
            w = torch.softmax(self.attn_pool(h), dim=1)
            ctx = (w * h).sum(dim=1)
            return self.classifier(ctx).squeeze(-1)

    model = EfficientTFT(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # fold 预测
    fold_seqs = []
    fold_indices = []
    for i in range(fold_start, fold_end):
        rel = i - fold_start
        if fold_valid[rel] and i >= seq_len:
            seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
            fold_seqs.append(seq)
            fold_indices.append(rel)

    if not fold_seqs:
        return None

    model.eval()
    with torch.no_grad():
        X_fold_t = torch.FloatTensor(np.array(fold_seqs)).to(device)
        logits = model(X_fold_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    result = np.full(fold_valid.sum(), 0.5)
    valid_positions = np.where(fold_valid)[0]
    pos_map = {v: idx for idx, v in enumerate(valid_positions)}
    for fi, prob in zip(fold_indices, probs):
        if fi in pos_map:
            result[pos_map[fi]] = prob

    return result


def _stacking_predict_lstm_saved(feat_raw, start, end, valid_mask, model_path,
                                 input_dim, feat_mean, feat_std, seq_len=48):
    """用已保存的 Stacking LSTM 模型做区间预测。"""
    result = np.full(int(valid_mask.sum()), 0.5, dtype=np.float32)
    if result.size == 0 or not os.path.exists(model_path):
        return result
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return result

    class LSTMAttention(nn.Module):
        def __init__(self, in_dim, hidden=128, layers=2, drop=0.3):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, layers,
                                batch_first=True, dropout=drop, bidirectional=True)
            self.attn_fc = nn.Linear(hidden * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            w = torch.softmax(self.attn_fc(out), dim=1)
            ctx = (w * out).sum(dim=1)
            return self.classifier(ctx).squeeze(-1)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMAttention(input_dim).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        feat_mean = np.asarray(feat_mean, dtype=np.float32)
        feat_std = np.asarray(feat_std, dtype=np.float32)
        seqs, rel_idx = [], []
        for i in range(start, end):
            rel = i - start
            if valid_mask[rel] and i >= seq_len:
                seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
                seqs.append(seq)
                rel_idx.append(rel)
        if not seqs:
            return result

        with torch.no_grad():
            logits = model(torch.FloatTensor(np.array(seqs)).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()

        valid_positions = np.where(valid_mask)[0]
        pos_map = {v: idx for idx, v in enumerate(valid_positions)}
        for rel, prob in zip(rel_idx, probs):
            pos = pos_map.get(rel)
            if pos is not None:
                result[pos] = float(np.asarray(prob).reshape(-1)[0])
    except Exception as e:
        log.warning(f"  LSTM 区间评估失败: {e}")
    return result


def _stacking_predict_tft_saved(feat_raw, start, end, valid_mask, model_path,
                                input_dim, feat_mean, feat_std, seq_len=96):
    """用已保存的 Stacking TFT 模型做区间预测。"""
    result = np.full(int(valid_mask.sum()), 0.5, dtype=np.float32)
    if result.size == 0 or not os.path.exists(model_path):
        return result
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return result

    class EfficientTFT(nn.Module):
        def __init__(self, in_dim, d_model=64, n_heads=4, d_ff=128, n_layers=2, drop=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Dropout(drop))
            self.lstm = nn.LSTM(d_model, d_model, n_layers,
                                batch_first=True, dropout=drop if n_layers > 1 else 0)
            self.lstm_norm = nn.LayerNorm(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=drop, activation='gelu', batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.transformer_norm = nn.LayerNorm(d_model)
            self.attn_pool = nn.Linear(d_model, 1)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(drop),
                nn.Linear(d_ff, 1))

        def forward(self, x):
            h = self.input_proj(x)
            lstm_out, _ = self.lstm(h)
            h = self.lstm_norm(lstm_out + h)
            h = self.transformer(h)
            h = self.transformer_norm(h)
            w = torch.softmax(self.attn_pool(h), dim=1)
            ctx = (w * h).sum(dim=1)
            return self.classifier(ctx).squeeze(-1)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EfficientTFT(input_dim).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        feat_mean = np.asarray(feat_mean, dtype=np.float32)
        feat_std = np.asarray(feat_std, dtype=np.float32)
        seqs, rel_idx = [], []
        for i in range(start, end):
            rel = i - start
            if valid_mask[rel] and i >= seq_len:
                seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
                seqs.append(seq)
                rel_idx.append(rel)
        if not seqs:
            return result

        with torch.no_grad():
            logits = model(torch.FloatTensor(np.array(seqs)).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()

        valid_positions = np.where(valid_mask)[0]
        pos_map = {v: idx for idx, v in enumerate(valid_positions)}
        for rel, prob in zip(rel_idx, probs):
            pos = pos_map.get(rel)
            if pos is not None:
                result[pos] = float(np.asarray(prob).reshape(-1)[0])
    except Exception as e:
        log.warning(f"  TFT 区间评估失败: {e}")
    return result


def _stacking_evaluate_meta_with_saved_models(
    meta_model,
    tf: str,
    feat_73: np.ndarray,
    feat_94: np.ndarray,
    y_all: np.ndarray,
    start: int,
    end: int,
    valid_mask: np.ndarray,
    hvol_idx: int,
    extra_feat_names: List[str],
    feat_mean_73: np.ndarray,
    feat_std_73: np.ndarray,
    feat_mean_94: np.ndarray,
    feat_std_94: np.ndarray,
) -> float:
    """使用最终保存的 5 个基模型评估 meta learner。"""
    from sklearn.metrics import roc_auc_score

    if valid_mask.sum() < 20:
        return 0.5

    X_73 = feat_73[start:end][valid_mask]
    X_94 = feat_94[start:end][valid_mask]
    y_true = y_all[start:end][valid_mask]

    # 1) LGB
    lgb_pred = np.full(len(X_73), 0.5, dtype=np.float32)
    try:
        import lightgbm as lgb
        lgb_path = os.path.join(MODEL_DIR, f'stacking_lgb_{tf}.txt')
        if os.path.exists(lgb_path):
            lgb_model = lgb.Booster(model_file=lgb_path)
            lgb_pred = lgb_model.predict(X_73).astype(np.float32)
    except Exception as e:
        log.warning(f"  LGB 评估失败: {e}")

    # 2) XGBoost
    xgb_pred = np.full(len(X_73), 0.5, dtype=np.float32)
    try:
        import xgboost as xgb
        xgb_path = os.path.join(MODEL_DIR, f'stacking_xgb_{tf}.json')
        if os.path.exists(xgb_path):
            xgb_model = xgb.Booster()
            xgb_model.load_model(xgb_path)
            xgb_pred = xgb_model.predict(xgb.DMatrix(X_73)).astype(np.float32)
    except Exception as e:
        log.warning(f"  XGBoost 评估失败: {e}")

    # 3) LSTM
    lstm_path = os.path.join(MODEL_DIR, f'stacking_lstm_{tf}.pt')
    lstm_pred = _stacking_predict_lstm_saved(
        feat_raw=feat_73,
        start=start,
        end=end,
        valid_mask=valid_mask,
        model_path=lstm_path,
        input_dim=feat_73.shape[1],
        feat_mean=feat_mean_73,
        feat_std=feat_std_73,
        seq_len=48,
    )

    # 4) TFT
    tft_path = os.path.join(MODEL_DIR, f'stacking_tft_{tf}.pt')
    tft_pred = _stacking_predict_tft_saved(
        feat_raw=feat_94,
        start=start,
        end=end,
        valid_mask=valid_mask,
        model_path=tft_path,
        input_dim=feat_94.shape[1],
        feat_mean=feat_mean_94,
        feat_std=feat_std_94,
        seq_len=96,
    )

    # 5) Cross-Asset LGB
    lgb_cross_pred = np.full(len(X_73), 0.5, dtype=np.float32)
    try:
        import lightgbm as lgb
        cross_path = os.path.join(MODEL_DIR, f'stacking_lgb_cross_{tf}.txt')
        if os.path.exists(cross_path):
            cross_model = lgb.Booster(model_file=cross_path)
            lgb_cross_pred = cross_model.predict(X_94).astype(np.float32)
    except Exception as e:
        log.warning(f"  CrossAsset LGB 评估失败: {e}")

    meta_X = np.column_stack([lgb_pred, xgb_pred, lstm_pred, tft_pred, lgb_cross_pred])

    # 附加特征
    if hvol_idx >= 0 and 'hvol_20' in extra_feat_names:
        hvol = feat_73[start:end][valid_mask, hvol_idx:hvol_idx + 1]
        hvol = np.nan_to_num(hvol, nan=0.0, posinf=0.0, neginf=0.0)
        meta_X = np.hstack([meta_X, hvol])

    meta_pred = meta_model.predict_proba(meta_X)[:, 1]
    try:
        return float(roc_auc_score(y_true, meta_pred))
    except Exception:
        return 0.5


def _stacking_retrain_lstm_full(feat_raw, y_all, full_end, full_valid,
                                input_dim, tf, feat_mean, feat_std,
                                seq_len=48, epochs=30):
    """全量重训 LSTM 用于 stacking 推理"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_mean = np.asarray(feat_mean, dtype=np.float32)
    feat_std = np.asarray(feat_std, dtype=np.float32)

    X_seqs, y_seqs = [], []
    for i in range(seq_len, full_end):
        if full_valid[i] and not np.isnan(y_all[i]):
            seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
            X_seqs.append(seq)
            y_seqs.append(y_all[i])

    if len(X_seqs) < 100:
        return

    X_t = torch.FloatTensor(np.array(X_seqs)).to(device)
    y_t = torch.FloatTensor(np.array(y_seqs)).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)

    class LSTMAttention(nn.Module):
        def __init__(self, in_dim, hidden=128, layers=2, drop=0.3):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, layers,
                                batch_first=True, dropout=drop, bidirectional=True)
            self.attn_fc = nn.Linear(hidden * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            w = torch.softmax(self.attn_fc(out), dim=1)
            ctx = (w * out).sum(dim=1)
            return self.classifier(ctx).squeeze(-1)

    model = LSTMAttention(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 标签平滑 (0→0.05, 1→0.95): 防止全量重训后 logit 过于极端，导致生产推理出现极端概率
    # OOF 训练时 (20 epochs) 不会过拟合，但全量重训 (30 epochs) 容易收敛到极端 logit
    label_smooth_eps = 0.05

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            smooth_yb = yb * (1 - label_smooth_eps) + 0.5 * label_smooth_eps
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), smooth_yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    save_path = os.path.join(MODEL_DIR, f'stacking_lstm_{tf}.pt')
    torch.save(model.state_dict(), save_path)


def _stacking_retrain_tft_full(feat_raw, y_all, full_end, full_valid,
                               input_dim, tf, feat_mean, feat_std,
                               seq_len=96, epochs=25):
    """全量重训 TFT 用于 stacking 推理"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_mean = np.asarray(feat_mean, dtype=np.float32)
    feat_std = np.asarray(feat_std, dtype=np.float32)

    X_seqs, y_seqs = [], []
    for i in range(seq_len, full_end):
        if full_valid[i] and not np.isnan(y_all[i]):
            seq = _stacking_normalize_seq(feat_raw[i - seq_len:i], feat_mean, feat_std)
            X_seqs.append(seq)
            y_seqs.append(y_all[i])

    if len(X_seqs) < 100:
        return

    X_t = torch.FloatTensor(np.array(X_seqs)).to(device)
    y_t = torch.FloatTensor(np.array(y_seqs)).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)

    class EfficientTFT(nn.Module):
        def __init__(self, in_dim, d_model=64, n_heads=4, d_ff=128, n_layers=2, drop=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Dropout(drop))
            self.lstm = nn.LSTM(d_model, d_model, n_layers,
                                batch_first=True, dropout=drop if n_layers > 1 else 0)
            self.lstm_norm = nn.LayerNorm(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=drop, activation='gelu', batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.transformer_norm = nn.LayerNorm(d_model)
            self.attn_pool = nn.Linear(d_model, 1)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(drop),
                nn.Linear(d_ff, 1))

        def forward(self, x):
            h = self.input_proj(x)
            lstm_out, _ = self.lstm(h)
            h = self.lstm_norm(lstm_out + h)
            h = self.transformer(h)
            h = self.transformer_norm(h)
            w = torch.softmax(self.attn_pool(h), dim=1)
            ctx = (w * h).sum(dim=1)
            return self.classifier(ctx).squeeze(-1)

    model = EfficientTFT(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # 标签平滑: 防止全量重训 (25 epochs) 后 TFT logit 过于极端
    label_smooth_eps = 0.05

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            smooth_yb = yb * (1 - label_smooth_eps) + 0.5 * label_smooth_eps
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), smooth_yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    save_path = os.path.join(MODEL_DIR, f'stacking_tft_{tf}.pt')
    torch.save(model.state_dict(), save_path)


# ================================================================
# 模式 13: TabNet 训练 (表格数据专用深度学习)
# ================================================================

def train_tabnet(timeframes: List[str] = None, min_samples: int = 10000):
    """
    训练 TabNet 模型 — 表格数据专用深度学习

    特点:
    - 可解释的注意力机制
    - 稀疏特征选择
    - 性能优于传统 GBDT (在某些任务上)
    """
    timeframes = timeframes or ['1h']
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"训练 TabNet — {SYMBOL}/{tf}")
        log.info(f"{'='*60}")

        t0 = time.time()

        # 准备数据
        features, labels_df = prepare_features(SYMBOL, tf)

        # 添加跨资产特征
        features = _add_cross_asset_features(features, tf)
        n_samples = len(features)
        log.info(f"含跨资产特征: {features.shape[1]} 维, 样本: {n_samples}")

        # 样本门禁：TabNet 在小样本上不稳定，达到门槛再训练
        if n_samples < int(min_samples):
            reason = f"insufficient_samples({n_samples}<{int(min_samples)})"
            log.warning(f"跳过 {tf} TabNet: {reason}")
            all_results[tf] = {
                "timeframe": tf,
                "skipped": True,
                "skip_reason": reason,
                "n_samples": n_samples,
                "min_samples_required": int(min_samples),
            }
            continue

        try:
            from ml_tabnet import train_tabnet_walk_forward
        except ImportError:
            log.error("ml_tabnet 模块未找到，请确保 ml_tabnet.py 存在")
            all_results[tf] = {
                "timeframe": tf,
                "skipped": True,
                "skip_reason": "ml_tabnet_missing",
                "n_samples": n_samples,
            }
            continue

        # 使用利润化标签
        target_col = 'profitable_long_5'
        if target_col not in labels_df.columns:
            log.warning(f"{target_col} 不存在，使用 fwd_dir_5")
            target_col = 'fwd_dir_5'

        # Walk-Forward 训练
        log.info("开始 Walk-Forward 训练...")
        model, wf_results = train_tabnet_walk_forward(
            df=pd.DataFrame(index=features.index),
            features=features,
            labels=labels_df,
            target_col=target_col,
            train_window=2000,
            test_window=500,
            step=250,
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-3,
            mask_type='entmax',
        )

        # 保存模型
        model_path = os.path.join(MODEL_DIR, f'tabnet_{tf}.zip')
        model.save(model_path)

        # 获取特征重要性
        feat_imp = model.get_feature_importance()
        log.info(f"\nTop 10 重要特征:")
        for idx, row in feat_imp.head(10).iterrows():
            log.info(f"  {row['feature']}: {row['importance']:.4f}")

        result = {
            'timeframe': tf,
            'n_features': features.shape[1],
            'n_samples': len(features),
            'mean_auc': wf_results['mean_auc'],
            'std_auc': wf_results['std_auc'],
            'n_folds': wf_results['n_folds'],
            'model_path': model_path,
            'elapsed_sec': round(time.time() - t0, 1),
            'top_features': feat_imp.head(20).to_dict('records'),
        }

        all_results[tf] = result
        log.info(f"\n{tf} TabNet 完成:")
        log.info(f"  AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
        log.info(f"  耗时: {result['elapsed_sec']}s")

    save_results('tabnet_training', all_results)
    return all_results


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='H800 GPU 离线训练')
    parser.add_argument('--mode', type=str, default='lgb',
                        choices=['lgb', 'lstm', 'optuna', 'backtest', 'tft',
                                 'cross_asset', 'incr_wf', 'mtf_fusion',
                                 'ppo', 'onnx', 'retrain', 'stacking', 'tabnet',
                                 'all', 'all_v2', 'all_v3', 'all_v4'],
                        help='训练模式')
    parser.add_argument('--tf', type=str, default=None,
                        help='指定周期 (逗号分隔, 如 1h,4h)')
    parser.add_argument('--trials', type=int, default=200,
                        help='Optuna 试验次数')
    parser.add_argument('--symbol', type=str, default='ETHUSDT',
                        help='交易对')
    parser.add_argument('--multi-horizon', type=int, default=1, choices=[0, 1],
                        help='LSTM 是否启用 Multi-Horizon (1=启用, 0=单头回退)')
    parser.add_argument('--min-stacking-samples', type=int,
                        default=int(os.environ.get('ML_STACKING_MIN_OOF_SAMPLES', '20000')),
                        help='Stacking 最小样本门槛 (默认 20000)')
    parser.add_argument('--min-tabnet-samples', type=int,
                        default=int(os.environ.get('ML_TABNET_MIN_SAMPLES', '10000')),
                        help='TabNet 最小样本门槛 (默认 10000)')
    args = parser.parse_args()

    global SYMBOL
    SYMBOL = args.symbol

    tfs = args.tf.split(',') if args.tf else None

    print_gpu_status()

    t_total = time.time()
    results = {}

    if args.mode in ('lgb', 'all'):
        results['lgb'] = train_lgb_gpu(tfs)

    if args.mode in ('lstm', 'all'):
        results['lstm'] = train_lstm(tfs, multi_horizon=bool(args.multi_horizon))

    if args.mode in ('optuna', 'all'):
        results['optuna'] = train_optuna(tfs, n_trials=args.trials)

    if args.mode == 'backtest':
        results['backtest'] = train_backtest_optuna(tfs, n_trials=args.trials)

    if args.mode in ('tft', 'all_v2'):
        results['tft'] = train_tft(tfs)

    if args.mode in ('cross_asset', 'all_v2'):
        results['cross_asset'] = train_cross_asset(tfs)

    if args.mode in ('incr_wf', 'all_v2'):
        results['incr_wf'] = train_incremental_wf(tfs)

    if args.mode in ('mtf_fusion', 'all_v2', 'all_v3'):
        results['mtf_fusion'] = train_mtf_fusion(tfs)

    if args.mode in ('ppo', 'all_v3'):
        results['ppo'] = train_ppo_position(tfs)

    if args.mode in ('onnx', 'all_v3'):
        results['onnx'] = export_onnx_models()

    if args.mode in ('retrain',):
        results['retrain'] = train_online_retrain(tfs)

    if args.mode in ('stacking', 'all_v4'):
        results['stacking'] = train_stacking_ensemble(tfs, min_samples=args.min_stacking_samples)

    if args.mode == 'tabnet':
        results['tabnet'] = train_tabnet(tfs, min_samples=args.min_tabnet_samples)

    total_elapsed = time.time() - t_total
    log.info(f"\n{'='*60}")
    log.info(f"全部训练完成！总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}分钟)")
    log.info(f"{'='*60}")

    # 打包模型用于部署回传
    if os.path.exists(MODEL_DIR):
        model_files = [f for f in os.listdir(MODEL_DIR)
                       if f.endswith(('.pt', '.txt', '.json'))]
        if model_files:
            log.info(f"\n训练产出 ({len(model_files)} 个文件):")
            for f in sorted(model_files):
                size = os.path.getsize(os.path.join(MODEL_DIR, f)) // 1024
                log.info(f"  {f} ({size}KB)")
            log.info("\n回传模型到生产机:")
            log.info("  tar -czf macd_models.tar.gz data/ml_models/ data/gpu_results/")
            log.info("  scp -J jumphost macd_models.tar.gz prod:/opt/macd-analysis/")


if __name__ == '__main__':
    main()
