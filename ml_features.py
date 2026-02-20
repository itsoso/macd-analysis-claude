"""
ML 特征工程模块
从现有技术指标体系提取 70+ 维特征，供时间序列预测模型使用。

特征分类:
  1. 价格动量 (returns, momentum, acceleration)
  2. 趋势 (MA 交叉, 斜率, 偏离度)
  3. 振荡器 (RSI, KDJ, CCI, MFI 状态)
  4. 波动率 (ATR, Bollinger, 历史波动率)
  5. 量价 (OBV, 量比, 资金流)
  6. 微结构 (蜡烛形态特征)
  7. 时间特征 (小时、星期、月份周期编码)
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_ml_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    从带指标的 DataFrame 提取 ML 特征矩阵。

    参数:
        df: 已计算好全部指标的 DataFrame (来自 add_all_indicators + add_moving_averages)
        lookback: 滚动窗口长度

    返回:
        feat_df: 与 df 同索引的特征 DataFrame (NaN 行需自行处理)
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    open_ = df['open'].astype(float)

    # ================================================================
    # 1. 价格动量特征
    # ================================================================
    for p in [1, 2, 3, 5, 8, 13, 21]:
        feat[f'ret_{p}'] = close.pct_change(p)

    feat['log_ret_1'] = np.log(close / close.shift(1))
    feat['log_ret_5'] = np.log(close / close.shift(5))

    # 动量 (rate of change)
    for p in [5, 10, 20]:
        feat[f'roc_{p}'] = (close - close.shift(p)) / close.shift(p)

    # 动量加速度
    mom_5 = close - close.shift(5)
    feat['momentum_accel'] = mom_5 - mom_5.shift(5)

    # 价格相对位置 (0-1 归一化到 lookback 窗口)
    feat['price_percentile'] = (
        close.rolling(lookback).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)
    )

    # ================================================================
    # 2. 趋势特征
    # ================================================================
    # MA 偏离度
    for ma_col in ['MA5', 'MA10', 'MA20', 'MA60']:
        if ma_col in df.columns:
            ma_val = df[ma_col].astype(float)
            feat[f'dist_{ma_col}'] = (close - ma_val) / ma_val
            feat[f'slope_{ma_col}'] = ma_val.pct_change(3)

    # MA 交叉信号
    if 'MA5' in df.columns and 'MA20' in df.columns:
        ma5 = df['MA5'].astype(float)
        ma20 = df['MA20'].astype(float)
        feat['ma_cross_5_20'] = (ma5 - ma20) / ma20
        feat['ma_cross_5_20_prev'] = feat['ma_cross_5_20'].shift(1)

    if 'MA10' in df.columns and 'MA60' in df.columns:
        ma10 = df['MA10'].astype(float)
        ma60 = df['MA60'].astype(float)
        feat['ma_cross_10_60'] = (ma10 - ma60) / ma60

    # MACD 特征
    if 'DIF' in df.columns and 'DEA' in df.columns:
        dif = df['DIF'].astype(float)
        dea = df['DEA'].astype(float)
        macd_bar = df.get('MACD_BAR', dif - dea).astype(float)
        feat['macd_dif'] = dif / close
        feat['macd_dea'] = dea / close
        feat['macd_bar'] = macd_bar / close
        feat['macd_bar_change'] = macd_bar.diff()
        feat['macd_cross'] = (dif - dea).apply(np.sign)
        feat['macd_cross_prev'] = feat['macd_cross'].shift(1)

    # 趋势强度: ADX 近似 (DI+/DI- 差异)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean() / close
    feat['atr_5'] = tr.rolling(5).mean() / close

    # ================================================================
    # 3. 振荡器特征
    # ================================================================
    # RSI
    for col in ['RSI6', 'RSI12']:
        if col in df.columns:
            rsi = df[col].astype(float)
            feat[col.lower()] = rsi / 100.0
            feat[f'{col.lower()}_slope'] = rsi.diff(3) / 100.0

    # KDJ
    for col in ['kdj_k', 'kdj_d', 'kdj_j', 'K', 'D', 'J']:
        if col in df.columns:
            v = df[col].astype(float)
            feat_name = col.lower().replace('kdj_', 'kdj_')
            if feat_name not in feat.columns:
                feat[feat_name] = v / 100.0

    if 'kdj_k_slope' in df.columns:
        feat['kdj_k_slope'] = df['kdj_k_slope'].astype(float)
    if 'kdj_d_slope' in df.columns:
        feat['kdj_d_slope'] = df['kdj_d_slope'].astype(float)
    if 'kd_macd' in df.columns:
        feat['kd_macd'] = df['kd_macd'].astype(float) / 100.0

    # CCI
    if 'CCI' in df.columns:
        cci = df['CCI'].astype(float)
        feat['cci'] = cci / 200.0  # 归一化到 [-1, 1] 附近
        feat['cci_slope'] = cci.diff(3) / 200.0

    # MFI
    if 'mfi' in df.columns:
        feat['mfi'] = df['mfi'].astype(float) / 100.0

    # ================================================================
    # 4. 波动率特征
    # ================================================================
    # 历史波动率
    log_ret = np.log(close / close.shift(1))
    feat['hvol_5'] = log_ret.rolling(5).std() * np.sqrt(24)  # 年化近似
    feat['hvol_20'] = log_ret.rolling(20).std() * np.sqrt(24)
    feat['hvol_ratio'] = feat['hvol_5'] / feat['hvol_20'].replace(0, np.nan)

    # Bollinger 特征
    if 'bb_pct_b' in df.columns:
        feat['bb_pct_b'] = df['bb_pct_b'].astype(float)
    if 'bb_bandwidth' in df.columns:
        feat['bb_bw'] = df['bb_bandwidth'].astype(float)
        feat['bb_bw_rank'] = feat['bb_bw'].rolling(lookback).rank(pct=True)
    if 'bb_mid_slope' in df.columns:
        feat['bb_mid_slope'] = df['bb_mid_slope'].astype(float)

    # 价格区间位置
    feat['hl_position'] = (close - low) / (high - low).replace(0, np.nan)

    # 最近 N bar 的高低点距离
    feat['dist_high_20'] = (close - high.rolling(20).max()) / close
    feat['dist_low_20'] = (close - low.rolling(20).min()) / close

    # ================================================================
    # 5. 量价特征
    # ================================================================
    feat['vol_ratio_5_20'] = (
        volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
    )
    feat['vol_change'] = volume.pct_change()
    feat['vol_price_corr_10'] = close.rolling(10).corr(volume)

    if 'obv' in df.columns:
        obv = df['obv'].astype(float)
        feat['obv_slope'] = obv.pct_change(5)
        if 'obv_sma20' in df.columns:
            feat['obv_div'] = (obv - df['obv_sma20'].astype(float)) / obv.abs().replace(0, np.nan)

    if 'vwap' in df.columns:
        feat['vwap_dist'] = (close - df['vwap'].astype(float)) / close

    # 量价配合度
    price_dir = close.diff().apply(np.sign)
    vol_dir = volume.diff().apply(np.sign)
    feat['vol_price_agree'] = (price_dir * vol_dir).rolling(5).mean()

    # 主动买入占比
    if 'taker_buy_base' in df.columns:
        tbv = df['taker_buy_base'].astype(float)
        feat['taker_buy_ratio'] = tbv / volume.replace(0, np.nan)

        # Order Flow Imbalance (OFI) - 高频微结构特征
        taker_sell = volume - tbv
        feat['ofi'] = (tbv - taker_sell) / volume.replace(0, np.nan)
        feat['ofi_ma5'] = feat['ofi'].rolling(5).mean()
        feat['ofi_std5'] = feat['ofi'].rolling(5).std()

        # 累积 OFI (类似 OBV 但基于买卖单不平衡)
        feat['cum_ofi'] = feat['ofi'].cumsum()
        feat['cum_ofi_slope'] = feat['cum_ofi'].pct_change(5)

        # 大单占比 (假设 volume > 平均的 2 倍为大单)
        vol_ma = volume.rolling(20).mean()
        is_large_trade = (volume > vol_ma * 2).astype(float)
        feat['large_trade_ratio'] = is_large_trade.rolling(10).mean()

        # 买卖压力不平衡度
        buy_pressure = tbv.rolling(5).sum()
        sell_pressure = taker_sell.rolling(5).sum()
        feat['buy_sell_pressure'] = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure).replace(0, np.nan)

    # VWAP 增强特征
    if 'vwap' in df.columns:
        vwap = df['vwap'].astype(float)
        # VWAP 偏离度的变化率
        vwap_dist = (close - vwap) / close
        feat['vwap_dist_change'] = vwap_dist.diff()
        feat['vwap_dist_ma5'] = vwap_dist.rolling(5).mean()

        # 价格在 VWAP 上方/下方的持续时间
        above_vwap = (close > vwap).astype(int)
        feat['above_vwap_streak'] = above_vwap.groupby((above_vwap != above_vwap.shift()).cumsum()).cumcount() + 1
        feat['above_vwap_streak'] = feat['above_vwap_streak'] * above_vwap

    # ================================================================
    # 6. 微结构 (蜡烛) 特征
    # ================================================================
    body = (close - open_).abs()
    total_range = (high - low).replace(0, np.nan)
    feat['body_ratio'] = body / total_range
    feat['upper_shadow_ratio'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / total_range
    feat['lower_shadow_ratio'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / total_range
    feat['is_bull'] = (close > open_).astype(float)

    # 连续阴阳统计
    bull = (close > open_).astype(int)
    feat['bull_streak'] = bull.groupby((bull != bull.shift()).cumsum()).cumcount() + 1
    feat['bull_streak'] = feat['bull_streak'] * bull  # 阳线为正, 阴线为0
    bear_streak = (1 - bull).groupby(((1 - bull) != (1 - bull).shift()).cumsum()).cumcount() + 1
    feat['bear_streak'] = bear_streak * (1 - bull)

    # 最近 3 根 K 线的形态统计
    feat['avg_body_3'] = body.rolling(3).mean() / close
    feat['max_range_3'] = total_range.rolling(3).max() / close

    # ================================================================
    # 7. 时间特征 (周期编码)
    # ================================================================
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
        feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    if hasattr(df.index, 'dayofweek'):
        dow = df.index.dayofweek
        feat['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        feat['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # ================================================================
    # 8. 跨周期统计特征 (均值回归 / 趋势延续)
    # ================================================================
    # 累计收益在 lookback 中的排名
    cum_ret_5 = close.pct_change(5)
    feat['cum_ret_5_rank'] = cum_ret_5.rolling(lookback).rank(pct=True)

    cum_ret_20 = close.pct_change(20)
    feat['cum_ret_20_rank'] = cum_ret_20.rolling(lookback).rank(pct=True)

    # 波动率调整后的收益
    feat['sharpe_5'] = cum_ret_5 / log_ret.rolling(5).std().replace(0, np.nan)
    feat['sharpe_20'] = cum_ret_20 / log_ret.rolling(20).std().replace(0, np.nan)

    # ================================================================
    # 9. 期货特有特征 (如果可用)
    # ================================================================
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].astype(float)
        feat['funding_rate'] = fr
        feat['funding_rate_ma'] = fr.rolling(8).mean()

    if 'open_interest_value' in df.columns:
        oi = df['open_interest_value'].astype(float)
        feat['oi_change'] = oi.pct_change()
        feat['oi_change_5'] = oi.pct_change(5)

    return feat


def compute_labels(df: pd.DataFrame, horizons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    计算预测标签 (未来收益)。

    参数:
        df: 原始 DataFrame
        horizons: 预测时间窗口 (bar 数), 默认 [1, 3, 5, 12, 24]

    返回:
        labels: 包含 fwd_ret_{h} 和 fwd_dir_{h} 列的 DataFrame
    """
    if horizons is None:
        horizons = [1, 3, 5, 12, 24]

    labels = pd.DataFrame(index=df.index)
    close = df['close'].astype(float)

    for h in horizons:
        fwd_ret = close.shift(-h) / close - 1
        labels[f'fwd_ret_{h}'] = fwd_ret
        labels[f'fwd_dir_{h}'] = (fwd_ret > 0).astype(int)
        # 三分类: -1=下跌, 0=震荡, 1=上涨 (阈值 0.3%)
        labels[f'fwd_cls3_{h}'] = pd.cut(
            fwd_ret,
            bins=[-np.inf, -0.003, 0.003, np.inf],
            labels=[-1, 0, 1],
        ).astype(float)

    return labels


def compute_profit_labels(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    cost_pct: float = 0.0015,
) -> pd.DataFrame:
    """
    计算利润化标签 — 只有扣除手续费后仍然盈利才算正类。

    对于做多: fwd_ret - cost > 0  → 1
    对于做空: -fwd_ret - cost > 0 → 1 (空头标签)
    综合:     |fwd_ret| > cost    → sign(fwd_ret)

    参数:
        cost_pct: 单次开平手续费 (默认 0.15%, 含滑点约 0.3% 来回)
    """
    if horizons is None:
        horizons = [3, 5, 12, 24]

    labels = pd.DataFrame(index=df.index)
    close = df['close'].astype(float)
    roundtrip_cost = cost_pct * 2  # 开+平

    for h in horizons:
        fwd_ret = close.shift(-h) / close - 1
        labels[f'fwd_ret_{h}'] = fwd_ret

        # 做多利润标签: 扣除来回手续费后仍盈利
        labels[f'profitable_long_{h}'] = (fwd_ret > roundtrip_cost).astype(int)
        # 做空利润标签
        labels[f'profitable_short_{h}'] = (fwd_ret < -roundtrip_cost).astype(int)

        # 最大有利偏移 (MAE-aware label): 未来 h bar 内的最大收益
        high_fwd = df['high'].rolling(h).max().shift(-h)
        low_fwd = df['low'].rolling(h).min().shift(-h)
        labels[f'max_long_ret_{h}'] = (high_fwd / close - 1)
        labels[f'max_short_ret_{h}'] = (1 - low_fwd / close)

        # 趋势标签: 价格最终方向且中途不被止损 (max drawdown < 3%)
        long_dd = (low_fwd / close - 1)  # 最大回撤 (做多)
        labels[f'trend_long_{h}'] = (
            (fwd_ret > roundtrip_cost) & (long_dd > -0.03)
        ).astype(int)

    return labels


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.85,
    importance_top_n: int = 35,
) -> List[str]:
    """
    特征精选: 去冗余 + 重要性筛选。

    1. 移除 >85% 互相关的冗余特征 (保留与标签相关性更高的)
    2. 用快速 LightGBM 计算特征重要性, 保留 top N
    """
    import lightgbm as lgb

    X_clean = X.replace([np.inf, -np.inf], np.nan)

    # 第一步: 相关性过滤
    corr_matrix = X_clean.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 计算每个特征与标签的相关性
    valid = y.notna()
    label_corr = X_clean[valid].corrwith(y[valid]).abs().fillna(0)

    to_drop = set()
    for col in upper.columns:
        high_corr_cols = upper.index[upper[col] > corr_threshold].tolist()
        for hc in high_corr_cols:
            if hc in to_drop:
                continue
            # 保留与标签相关性更高的那个
            if label_corr.get(col, 0) >= label_corr.get(hc, 0):
                to_drop.add(hc)
            else:
                to_drop.add(col)

    features_after_corr = [c for c in X_clean.columns if c not in to_drop]

    # 第二步: 快速 LightGBM 重要性排序
    X_sel = X_clean[features_after_corr]
    valid_mask = y.notna() & X_sel.notna().all(axis=1)
    if valid_mask.sum() < 100:
        return features_after_corr[:importance_top_n]

    dtrain = lgb.Dataset(X_sel[valid_mask], label=y[valid_mask])
    quick_model = lgb.train(
        {
            'objective': 'binary', 'metric': 'auc',
            'num_leaves': 15, 'learning_rate': 0.1,
            'feature_fraction': 0.7, 'bagging_fraction': 0.7,
            'bagging_freq': 3, 'min_child_samples': 50,
            'verbose': -1, 'n_jobs': -1, 'seed': 42,
        },
        dtrain, num_boost_round=100,
    )
    imp = dict(zip(features_after_corr, quick_model.feature_importance(importance_type='gain').tolist()))
    sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)

    selected = [f for f, _ in sorted_feats[:importance_top_n]]
    return selected


def get_feature_names() -> List[str]:
    """返回所有可能的特征名列表 (用于模型解释)"""
    names = []
    for p in [1, 2, 3, 5, 8, 13, 21]:
        names.append(f'ret_{p}')
    names.extend(['log_ret_1', 'log_ret_5'])
    for p in [5, 10, 20]:
        names.append(f'roc_{p}')
    names.append('momentum_accel')
    names.append('price_percentile')
    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        names.extend([f'dist_{ma}', f'slope_{ma}'])
    names.extend([
        'ma_cross_5_20', 'ma_cross_5_20_prev',
        'ma_cross_10_60',
        'macd_dif', 'macd_dea', 'macd_bar', 'macd_bar_change',
        'macd_cross', 'macd_cross_prev',
        'atr_14', 'atr_5',
        'rsi6', 'rsi6_slope', 'rsi12', 'rsi12_slope',
        'k', 'd', 'j', 'kdj_k', 'kdj_d', 'kdj_j',
        'kdj_k_slope', 'kdj_d_slope', 'kd_macd',
        'cci', 'cci_slope', 'mfi',
        'hvol_5', 'hvol_20', 'hvol_ratio',
        'bb_pct_b', 'bb_bw', 'bb_bw_rank', 'bb_mid_slope',
        'hl_position', 'dist_high_20', 'dist_low_20',
        'vol_ratio_5_20', 'vol_change', 'vol_price_corr_10',
        'obv_slope', 'obv_div', 'vwap_dist',
        'vol_price_agree', 'taker_buy_ratio',
        'ofi', 'ofi_ma5', 'ofi_std5', 'cum_ofi', 'cum_ofi_slope',
        'large_trade_ratio', 'buy_sell_pressure',
        'vwap_dist_change', 'vwap_dist_ma5', 'above_vwap_streak',
        'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
        'is_bull', 'bull_streak', 'bear_streak',
        'avg_body_3', 'max_range_3',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'cum_ret_5_rank', 'cum_ret_20_rank',
        'sharpe_5', 'sharpe_20',
        'funding_rate', 'funding_rate_ma',
        'oi_change', 'oi_change_5',
    ])
    return names
