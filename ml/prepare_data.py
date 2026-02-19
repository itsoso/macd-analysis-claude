"""
数据准备与特征工程

从 binance_fetcher 获取多周期 K线数据，
调用 signal_core 生成全量六书特征，
构造分类/回归标签，切分数据集。
"""

import os
import sys
import numpy as np
import pandas as pd

# 项目根目录加入 path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from signal_core import compute_signals_six, calc_fusion_score_six_batch
from ml.config import (
    SYMBOL, TIMEFRAMES, PRIMARY_TF, FETCH_DAYS,
    LABEL_HORIZON, LABEL_THRESHOLD,
    TRAIN_RATIO, VAL_RATIO, LOOKBACK,
    DATA_DIR,
)


# ── 1. 数据获取 ──────────────────────────────────────────

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


def fetch_all_tf_data(days=FETCH_DAYS) -> dict:
    """
    获取所有周期的 K线数据并添加指标。
    优先从 ml/data/raw/ 加载预下载的 Parquet 文件 (适用于无外网的训练机),
    否则尝试 binance API。
    """
    data_all = {}
    for tf in TIMEFRAMES:
        # 优先: 预下载的含指标文件
        ind_path = os.path.join(RAW_DIR, f"{SYMBOL}_{tf}_indicators.parquet")
        raw_path = os.path.join(RAW_DIR, f"{SYMBOL}_{tf}_raw.parquet")

        if os.path.exists(ind_path):
            print(f"[prepare] 从本地加载 {ind_path}...")
            df = pd.read_parquet(ind_path)
        elif os.path.exists(raw_path):
            print(f"[prepare] 从本地加载 {raw_path} + 计算指标...")
            df = pd.read_parquet(raw_path)
            df = add_all_indicators(df)
        else:
            print(f"[prepare] 从 API 获取 {SYMBOL} {tf} K线 (days={days})...")
            df = fetch_binance_klines(symbol=SYMBOL, interval=tf, days=days)
            if df is None or len(df) == 0:
                raise RuntimeError(
                    f"无法获取 {tf} 数据。请先在外网机器运行:\n"
                    f"  python ml/download_data.py\n"
                    f"然后将 ml/data/raw/ 复制到训练机。"
                )
            df = add_all_indicators(df)

        data_all[tf] = df
        print(f"  -> {len(df)} bars, range: {df.index[0]} ~ {df.index[-1]}")
    return data_all


# ── 2. 特征提取 ──────────────────────────────────────────

def extract_features(data_all: dict, tf: str) -> pd.DataFrame:
    """
    对指定周期提取全量特征:
      - OHLCV 原始列
      - 技术指标 (MACD, KDJ, RSI, CCI, MA)
      - 六书信号分数 (通过 calc_fusion_score_six_batch + return_features)
      - 滚动窗口统计
      - 时间编码
    """
    df = data_all[tf].copy()
    print(f"[features] 提取 {tf} 特征, {len(df)} bars...")

    # 2a. 六书信号
    signals = compute_signals_six(df, tf, data_all, max_bars=0, fast=False)
    config = {"fusion_mode": "c6_veto_4"}
    score_dict, ordered_ts, feature_dict = calc_fusion_score_six_batch(
        signals, df, config, warmup=60, return_features=True
    )

    # 构造六书特征 DataFrame
    feat_rows = []
    for ts in ordered_ts:
        row = feature_dict[ts]
        row["fusion_sell"], row["fusion_buy"] = score_dict[ts]
        feat_rows.append(row)
    six_book_df = pd.DataFrame(feat_rows, index=ordered_ts)

    # 2b. 从 signals 中提取额外 Series 特征
    extra = pd.DataFrame(index=df.index)
    # MA 排列
    ma_arr = signals["ma"].get("arrangement", pd.Series("mixed", index=df.index))
    extra["ma_bullish"] = (ma_arr == "bullish").astype(np.float32)
    extra["ma_bearish"] = (ma_arr == "bearish").astype(np.float32)
    # 原始 sub-scores
    for key in ["cs_sell", "cs_buy", "bb_sell", "bb_buy",
                "vp_sell", "vp_buy", "kdj_sell", "kdj_buy"]:
        s = signals.get(key)
        if s is not None and hasattr(s, "values"):
            extra[f"raw_{key}"] = s.values.astype(np.float32)

    # 2c. OHLCV + 技术指标
    base_cols = [c for c in df.columns if c in [
        "open", "high", "low", "close", "volume", "quote_volume",
        "DIF", "DEA", "MACD_BAR", "DIF_FAST", "DEA_FAST", "MACD_BAR_FAST",
        "K", "D", "J", "CCI", "RSI6", "RSI12",
        "MA5", "MA10", "MA30", "MA60", "MA120", "MA240",
    ]]
    base = df[base_cols].copy()

    # 2d. 价格衍生特征
    close = df["close"]
    base["returns_1"] = close.pct_change(1)
    base["returns_5"] = close.pct_change(5)
    base["returns_20"] = close.pct_change(20)
    base["volatility_20"] = close.pct_change().rolling(20).std()
    base["volume_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()
    base["high_low_range"] = (df["high"] - df["low"]) / close
    base["close_ma5_ratio"] = close / df["MA5"] - 1 if "MA5" in df.columns else 0
    base["close_ma30_ratio"] = close / df["MA30"] - 1 if "MA30" in df.columns else 0
    base["close_ma120_ratio"] = close / df["MA120"] - 1 if "MA120" in df.columns else 0

    # 2e. 滚动窗口统计 (看回 N 根)
    for w in [5, 10, 20]:
        base[f"close_zscore_{w}"] = (close - close.rolling(w).mean()) / close.rolling(w).std()
        base[f"volume_zscore_{w}"] = (
            (df["volume"] - df["volume"].rolling(w).mean()) / df["volume"].rolling(w).std()
        )
        base[f"returns_mean_{w}"] = close.pct_change().rolling(w).mean()
        base[f"returns_std_{w}"] = close.pct_change().rolling(w).std()

    # 2f. 时间编码 (周期性)
    idx = df.index
    base["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24).astype(np.float32)
    base["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24).astype(np.float32)
    base["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    base["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    base["month_sin"] = np.sin(2 * np.pi * idx.month / 12).astype(np.float32)
    base["month_cos"] = np.cos(2 * np.pi * idx.month / 12).astype(np.float32)

    # 2g. 合并所有特征
    result = base.join(extra, how="left").join(six_book_df, how="left")
    result = result.astype(np.float32)

    print(f"  -> 特征维度: {result.shape[1]} 列")
    return result


# ── 3. Regime 检测 ────────────────────────────────────────

def compute_regime(df: pd.DataFrame) -> pd.Series:
    """
    简单 regime 分类:
      0 = 趋势 (trend)
      1 = 震荡 (range)
      2 = 高波动 (volatile)
    """
    close = df["close"]
    ret = close.pct_change()
    vol_20 = ret.rolling(20).std()
    vol_median = vol_20.rolling(100, min_periods=20).median()

    # MA 趋势强度
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma_ratio = (ma20 / ma60 - 1).abs()

    regime = pd.Series(1, index=df.index, dtype=np.int64)  # 默认震荡
    regime[ma_ratio > 0.03] = 0  # 趋势
    regime[vol_20 > vol_median * 1.5] = 2  # 高波动

    return regime


# ── 4. 标签构造 ──────────────────────────────────────────

def make_labels(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    构造标签:
      - cls_label: 0=short, 1=hold, 2=long
      - reg_label: 未来 N 根 K线收益率 (连续值)
    """
    horizon = LABEL_HORIZON.get(tf, 6)
    future_ret = df["close"].pct_change(horizon).shift(-horizon)

    cls_label = pd.Series(1, index=df.index, dtype=np.int64)  # hold
    cls_label[future_ret > LABEL_THRESHOLD] = 2   # long
    cls_label[future_ret < -LABEL_THRESHOLD] = 0  # short

    labels = pd.DataFrame({
        "cls_label": cls_label,
        "reg_label": future_ret.astype(np.float32),
    }, index=df.index)

    # 统计
    counts = cls_label.value_counts().sort_index()
    total = len(cls_label.dropna())
    print(f"[labels] {tf} horizon={horizon}, threshold={LABEL_THRESHOLD}")
    print(f"  short={counts.get(0,0)} ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  hold ={counts.get(1,0)} ({counts.get(1,0)/total*100:.1f}%)")
    print(f"  long ={counts.get(2,0)} ({counts.get(2,0)/total*100:.1f}%)")

    return labels


# ── 5. 跨周期特征对齐 ────────────────────────────────────

def align_cross_tf_features(features_all: dict, target_tf: str) -> pd.DataFrame:
    """
    将其他周期的融合分数对齐到目标周期。
    大周期 forward-fill 到小周期时间轴。
    """
    target_idx = features_all[target_tf].index
    cross_feats = pd.DataFrame(index=target_idx)

    for tf, feat_df in features_all.items():
        if tf == target_tf:
            continue
        # 只取融合分数等关键特征
        cols_to_align = [c for c in feat_df.columns if c in [
            "fusion_sell", "fusion_buy",
            "div_sell", "div_buy", "ma_sell", "ma_buy",
        ]]
        if not cols_to_align:
            continue
        sub = feat_df[cols_to_align].copy()
        sub.columns = [f"{tf}_{c}" for c in cols_to_align]
        # reindex + forward fill
        sub = sub.reindex(target_idx, method="ffill")
        cross_feats = cross_feats.join(sub, how="left")

    return cross_feats


# ── 6. 标准化 ────────────────────────────────────────────

def normalize_features(train_df: pd.DataFrame, val_df: pd.DataFrame,
                       test_df: pd.DataFrame):
    """Z-score 标准化，用训练集统计量"""
    mean = train_df.mean()
    std = train_df.std().replace(0, 1)

    train_norm = (train_df - mean) / std
    val_norm = (val_df - mean) / std
    test_norm = (test_df - mean) / std

    stats = pd.DataFrame({"mean": mean, "std": std})
    return train_norm, val_norm, test_norm, stats


# ── 7. 主流程 ────────────────────────────────────────────

def prepare_dataset(tf: str = PRIMARY_TF, days: int = FETCH_DAYS):
    """
    完整数据准备管线:
    1. 获取数据 → 2. 提取特征 → 3. 构造标签 → 4. 跨周期对齐
    5. 拼接 → 6. 时间序列切分 → 7. 标准化 → 8. 保存
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. 获取所有周期数据
    data_all = fetch_all_tf_data(days)

    # 2. 每个周期提取特征
    features_all = {}
    for t in TIMEFRAMES:
        features_all[t] = extract_features(data_all, t)

    # 3. 标签 (主周期)
    labels = make_labels(data_all[tf], tf)

    # 4. Regime
    regime = compute_regime(data_all[tf])

    # 5. 跨周期对齐
    cross_feats = align_cross_tf_features(features_all, tf)

    # 6. 拼接
    main_feats = features_all[tf]
    full_df = main_feats.join(cross_feats, how="left")
    full_df["regime"] = regime
    full_df = full_df.join(labels, how="left")

    # 删除无标签行 (尾部 horizon 行)
    full_df = full_df.dropna(subset=["cls_label", "reg_label"])
    # 删除前部 warmup 行
    full_df = full_df.iloc[max(LOOKBACK, 240):]

    print(f"\n[prepare] 最终数据集: {full_df.shape}")
    print(f"  时间范围: {full_df.index[0]} ~ {full_df.index[-1]}")

    # 7. 时间序列切分 (无 shuffle, 防止数据泄露)
    n = len(full_df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_df = full_df.iloc[:n_train]
    val_df = full_df.iloc[n_train:n_train + n_val]
    test_df = full_df.iloc[n_train + n_val:]

    print(f"  训练: {len(train_df)} | 验证: {len(val_df)} | 测试: {len(test_df)}")

    # 8. 分离特征和标签
    label_cols = ["cls_label", "reg_label"]
    skip_norm_cols = ["regime"]  # regime 保持整数, 不标准化
    feat_cols = [c for c in full_df.columns if c not in label_cols]
    norm_cols = [c for c in feat_cols if c not in skip_norm_cols]

    # 9. 标准化 (仅对 norm_cols)
    train_norm, val_norm, test_norm, stats = normalize_features(
        train_df[norm_cols], val_df[norm_cols], test_df[norm_cols]
    )

    # 重新拼接标签 + 非标准化列
    train_out = train_norm.copy()
    for col in skip_norm_cols:
        train_out[col] = train_df[col].values
    train_out[label_cols] = train_df[label_cols].values
    val_out = val_norm.copy()
    for col in skip_norm_cols:
        val_out[col] = val_df[col].values
    val_out[label_cols] = val_df[label_cols].values
    test_out = test_norm.copy()
    for col in skip_norm_cols:
        test_out[col] = test_df[col].values
    test_out[label_cols] = test_df[label_cols].values

    # 10. 保存
    train_out.to_parquet(os.path.join(DATA_DIR, f"train_{tf}.parquet"))
    val_out.to_parquet(os.path.join(DATA_DIR, f"val_{tf}.parquet"))
    test_out.to_parquet(os.path.join(DATA_DIR, f"test_{tf}.parquet"))
    stats.to_parquet(os.path.join(DATA_DIR, f"stats_{tf}.parquet"))

    # 保存特征列名
    import json
    meta = {
        "feat_cols": feat_cols,
        "label_cols": label_cols,
        "tf": tf,
        "lookback": LOOKBACK,
        "n_features": len(feat_cols),
        "n_train": len(train_out),
        "n_val": len(val_out),
        "n_test": len(test_out),
    }
    with open(os.path.join(DATA_DIR, f"meta_{tf}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[prepare] 数据已保存到 {DATA_DIR}/")
    print(f"  特征: {len(feat_cols)} 列")
    return meta


if __name__ == "__main__":
    prepare_dataset()
