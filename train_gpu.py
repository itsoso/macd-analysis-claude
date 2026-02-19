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

def train_lstm(timeframes: List[str] = None):
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
        log.info(f"{'='*60}")

        t0 = time.time()
        features, labels_df = prepare_features(SYMBOL, tf)

        result = _train_lstm_single(features, labels_df, tf, device)
        result['elapsed_sec'] = round(time.time() - t0, 1)
        all_results[tf] = result
        log.info(f"{tf} 完成: {result}")

    save_results('lstm_training', all_results)
    return all_results


def _train_lstm_single(features, labels_df, tf, device):
    """单周期 LSTM 训练 (Walk-Forward)"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    SEQ_LEN = 48
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 256
    EPOCHS = 50
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

    # 准备序列数据
    feat_values = features.values.astype(np.float32)
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
        X, y = [], []
        for i in range(start, min(end, n)):
            seq = feat_values[i:i + SEQ_LEN]
            lbl = label_values[i + SEQ_LEN]
            if not np.isnan(lbl):
                X.append(seq)
                y.append(lbl)
        if not X:
            return None, None
        return torch.tensor(np.array(X)), torch.tensor(np.array(y))

    X_train, y_train = make_sequences(0, train_end)
    X_val, y_val = make_sequences(train_end, val_end)
    X_test, y_test = make_sequences(val_end, n)

    if X_train is None or len(X_train) < 100:
        return {'error': '训练数据不足', 'train_samples': 0}

    log.info(f"序列数据: train={len(X_train)}, val={len(X_val) if X_val is not None else 0}, "
             f"test={len(X_test) if X_test is not None else 0}")

    # 模型
    model = LSTMAttention(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and not use_bf16 else None
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'),
    )

    best_val_auc = 0
    patience = 10
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
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
                val_pred = torch.sigmoid(model(X_val.to(device))).cpu().numpy()
                val_true = y_val.numpy()
            try:
                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(val_true, val_pred)
            except Exception:
                val_auc = 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
                # 保存最佳模型
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(MODEL_DIR, f'lstm_{tf}.pt'))
            else:
                no_improve += 1

            if epoch % 5 == 0 or no_improve == 0:
                log.info(f"  Epoch {epoch:3d}: loss={epoch_loss/len(train_loader):.4f} "
                         f"val_AUC={val_auc:.4f} best={best_val_auc:.4f}")

            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    # 测试集评估
    test_auc = 0.5
    if X_test is not None and len(X_test) > 0:
        best_path = os.path.join(MODEL_DIR, f'lstm_{tf}.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            test_pred = torch.sigmoid(model(X_test.to(device))).cpu().numpy()
            test_true = y_test.numpy()
        try:
            from sklearn.metrics import roc_auc_score
            test_auc = roc_auc_score(test_true, test_pred)
        except Exception:
            pass

    return {
        'best_val_auc': round(best_val_auc, 4),
        'test_auc': round(test_auc, 4),
        'train_samples': len(X_train),
        'val_samples': len(X_val) if X_val is not None else 0,
        'test_samples': len(X_test) if X_test is not None else 0,
        'epochs_trained': epoch + 1,
        'input_dim': input_dim,
        'bf16': use_bf16,
    }


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

def train_backtest_optuna(timeframes: List[str] = None, n_trials: int = 500):
    """用 Optuna 优化回测参数 (替代网格搜索)"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("Optuna 未安装")
        return {}

    from signal_core import compute_signals_six
    from strategy_futures import FuturesEngine

    timeframes = timeframes or ['1h', '4h']
    all_results = {}

    for tf in timeframes:
        log.info(f"\n{'='*60}")
        log.info(f"回测参数优化 — {SYMBOL}/{tf} ({n_trials} trials)")
        log.info(f"{'='*60}")

        t0 = time.time()

        # 加载数据 + 计算信号 (只算一次)
        from indicators import add_all_indicators
        from ma_indicators import add_moving_averages

        df = load_klines_local(SYMBOL, tf)
        df = add_all_indicators(df)
        df = add_moving_averages(df)
        signals = compute_signals_six(df)

        # 最近 120 天用于回测
        trade_days = 120
        cutoff = df.index[-1] - pd.Timedelta(days=trade_days)
        df_trade = df[df.index >= cutoff]
        sig_trade = {k: v[v.index >= cutoff] if hasattr(v, 'index') else v
                     for k, v in signals.items()}

        def objective(trial):
            params = {
                'buy_threshold': trial.suggest_int('buy_threshold', 15, 40),
                'long_threshold': trial.suggest_int('long_threshold', 25, 50),
                'sell_threshold': trial.suggest_int('sell_threshold', 10, 30),
                'short_threshold': trial.suggest_int('short_threshold', 15, 40),
                'long_sl': trial.suggest_float('long_sl', -0.20, -0.04),
                'long_tp': trial.suggest_float('long_tp', 0.10, 0.60),
                'short_sl': trial.suggest_float('short_sl', -0.35, -0.10),
                'short_tp': trial.suggest_float('short_tp', 0.20, 1.00),
                'leverage': trial.suggest_int('leverage', 2, 8),
                'margin_use': trial.suggest_float('margin_use', 0.3, 0.8),
            }

            try:
                engine = FuturesEngine(
                    initial_capital=10000,
                    leverage=params['leverage'],
                    margin_usage=params['margin_use'],
                )

                for i in range(len(df_trade)):
                    bar = df_trade.iloc[i]
                    # 简化的信号处理: 使用融合分数
                    score = sig_trade.get('fusion_score_series')
                    if score is not None and hasattr(score, 'iloc') and i < len(score):
                        s = score.iloc[i] if i < len(score) else 0
                        if s >= params['long_threshold']:
                            engine.open_long(bar['close'], bar.name)
                        elif s <= -params['short_threshold']:
                            engine.open_short(bar['close'], bar.name)

                    engine.update_bar(bar)

                result = engine.get_summary()
                sharpe = result.get('sharpe_ratio', -10)
                return sharpe if np.isfinite(sharpe) else -10

            except Exception:
                return -10

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        all_results[tf] = {
            'best_sharpe': round(best.value, 4),
            'best_params': best.params,
            'n_trials': n_trials,
            'elapsed_sec': round(time.time() - t0, 1),
        }
        log.info(f"最优 Sharpe: {best.value:.4f}")
        log.info(f"最优参数: {json.dumps(best.params, indent=2)}")

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
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='H800 GPU 离线训练')
    parser.add_argument('--mode', type=str, default='lgb',
                        choices=['lgb', 'lstm', 'optuna', 'backtest', 'all'],
                        help='训练模式')
    parser.add_argument('--tf', type=str, default=None,
                        help='指定周期 (逗号分隔, 如 1h,4h)')
    parser.add_argument('--trials', type=int, default=200,
                        help='Optuna 试验次数')
    parser.add_argument('--symbol', type=str, default='ETHUSDT',
                        help='交易对')
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
        results['lstm'] = train_lstm(tfs)

    if args.mode in ('optuna', 'all'):
        results['optuna'] = train_optuna(tfs, n_trials=args.trials)

    if args.mode == 'backtest':
        results['backtest'] = train_backtest_optuna(tfs, n_trials=args.trials)

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
