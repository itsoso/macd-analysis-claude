"""
六书信号核心模块

目的:
1. 统一研究/回测/实盘的信号计算实现
2. 避免实盘模块直接依赖超大优化脚本

性能优化:
- compute_signals_six: 子模块 profiling
- calc_fusion_score_six_batch: 批量向量化评分（P1优化核心）
- get_signal_at: 二分查找替代线性扫描
"""

import time

import numpy as np
import pandas as pd

from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from ma_indicators import compute_ma_signals
from candlestick_patterns import compute_candlestick_scores
from bollinger_strategy import compute_bollinger_scores
from volume_price_strategy import compute_volume_price_scores
from kdj_strategy import compute_kdj_scores


def compute_signals_six(df, tf, data_all, max_bars=0):
    """为指定时间框架计算六维信号(含KDJ)

    参数:
        df: 主周期 DataFrame (已含指标)
        tf: 时间框架字符串
        data_all: 多周期数据 dict
        max_bars: >0 时只保留尾部 max_bars 根K线用于计算，大幅降低耗时。
                  0 = 不限制 (全量计算，用于回测)。

    返回:
        signals: dict + 额外 '_perf' key 记录子模块耗时
    """
    # ---- 尾部截断优化 ----
    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()

    signals = {}
    perf = {}

    # 1. 背离信号(主周期)
    t0 = time.time()
    lookback = max(60, min(200, len(df) // 3))
    div_signals = analyze_signals_enhanced(df, lookback)
    signals["div"] = div_signals
    perf['div_main'] = time.time() - t0

    # 8h辅助背离(如果可用)
    t0 = time.time()
    signals["div_8h"] = {}
    if "8h" in data_all and tf not in ("8h", "12h", "16h", "24h"):
        signals["div_8h"] = analyze_signals_enhanced(data_all["8h"], 90)
    perf['div_8h'] = time.time() - t0

    # 2. 均线信号
    t0 = time.time()
    ma_signals = compute_ma_signals(df, timeframe=tf)
    signals["ma"] = ma_signals
    perf['ma'] = time.time() - t0

    # 3. 蜡烛图
    t0 = time.time()
    cs_sell, cs_buy, _cs_names = compute_candlestick_scores(df)
    signals["cs_sell"] = cs_sell
    signals["cs_buy"] = cs_buy
    perf['candlestick'] = time.time() - t0

    # 4. 布林带
    t0 = time.time()
    bb_sell, bb_buy, _bb_names = compute_bollinger_scores(df)
    signals["bb_sell"] = bb_sell
    signals["bb_buy"] = bb_buy
    perf['bollinger'] = time.time() - t0

    # 5. 量价
    t0 = time.time()
    vp_sell, vp_buy, _vp_names = compute_volume_price_scores(df)
    signals["vp_sell"] = vp_sell
    signals["vp_buy"] = vp_buy
    perf['volume_price'] = time.time() - t0

    # 6. KDJ
    t0 = time.time()
    kdj_sell, kdj_buy, _kdj_names = compute_kdj_scores(df)
    signals["kdj_sell"] = kdj_sell
    signals["kdj_buy"] = kdj_buy
    perf['kdj'] = time.time() - t0

    signals['_perf'] = perf
    return signals


# ──────────────────────────────────────────────────────────
# 单 bar 评分 (保持向后兼容, 实盘使用)
# ──────────────────────────────────────────────────────────

def calc_fusion_score_six(signals, df, idx, dt, config):
    """六维融合评分 — 支持c6_veto_4/kdj_weighted/kdj_timing三种模式"""
    mode = config.get("fusion_mode", "c6_veto_4")

    # 1. 背离
    sig_main = get_signal_at(signals["div"], dt) or dict(DEFAULT_SIG)
    sig_8h = get_signal_at(signals.get("div_8h", {}), dt) or dict(DEFAULT_SIG)

    merged = dict(DEFAULT_SIG)
    merged["top"] = 0
    merged["bottom"] = 0
    for sig_src, w in [(sig_main, 1.0), (sig_8h, 0.5)]:
        merged["top"] += sig_src.get("top", 0) * w
        merged["bottom"] += sig_src.get("bottom", 0) * w
        for k in DEFAULT_SIG:
            if isinstance(DEFAULT_SIG[k], bool) and sig_src.get(k):
                merged[k] = True
            elif isinstance(DEFAULT_SIG[k], (int, float)) and k not in ("top", "bottom"):
                merged[k] = max(merged.get(k, 0), sig_src.get(k, 0))

    trend = {
        "is_downtrend": False,
        "is_uptrend": False,
        "ma_bearish": False,
        "ma_bullish": False,
        "ma_slope_down": False,
        "ma_slope_up": False,
    }
    if idx >= 30:
        c5 = df["close"].iloc[max(0, idx - 5) : idx].mean()
        c20 = df["close"].iloc[max(0, idx - 20) : idx].mean()
        if c5 < c20 * 0.99:
            trend["is_downtrend"] = True
            trend["ma_bearish"] = True
        elif c5 > c20 * 1.01:
            trend["is_uptrend"] = True
            trend["ma_bullish"] = True

    div_sell, _ = _calc_top_score(merged, trend)
    div_buy = _calc_bottom_score(merged, trend)

    # 2. 均线
    ma_sell = float(signals["ma"]["sell_score"].iloc[idx]) if idx < len(signals["ma"]["sell_score"]) else 0
    ma_buy = float(signals["ma"]["buy_score"].iloc[idx]) if idx < len(signals["ma"]["buy_score"]) else 0

    # 3-6. K线、布林、量价、KDJ
    cs_sell = float(signals["cs_sell"].iloc[idx]) if idx < len(signals["cs_sell"]) else 0
    cs_buy = float(signals["cs_buy"].iloc[idx]) if idx < len(signals["cs_buy"]) else 0
    bb_sell = float(signals["bb_sell"].iloc[idx]) if idx < len(signals["bb_sell"]) else 0
    bb_buy = float(signals["bb_buy"].iloc[idx]) if idx < len(signals["bb_buy"]) else 0
    vp_sell = float(signals["vp_sell"].iloc[idx]) if idx < len(signals["vp_sell"]) else 0
    vp_buy = float(signals["vp_buy"].iloc[idx]) if idx < len(signals["vp_buy"]) else 0
    kdj_sell = float(signals["kdj_sell"].iloc[idx]) if idx < len(signals["kdj_sell"]) else 0
    kdj_buy = float(signals["kdj_buy"].iloc[idx]) if idx < len(signals["kdj_buy"]) else 0

    # MA排列加成
    ma_arr_bonus_sell = 1.0
    ma_arr_bonus_buy = 1.0
    if idx < len(signals["ma"]["sell_score"]):
        ma_data = signals["ma"]
        if "arrangement" in ma_data:
            arr_series = ma_data.get("arrangement", None)
            if arr_series is not None and hasattr(arr_series, "iloc") and idx < len(arr_series):
                try:
                    arr_val = float(arr_series.iloc[idx])
                    if arr_val < 0:
                        ma_arr_bonus_sell = 1.10
                    elif arr_val > 0:
                        ma_arr_bonus_buy = 1.10
                except (ValueError, TypeError):
                    pass

    return _fuse_scores(
        mode, config,
        div_sell, div_buy,
        ma_sell, ma_buy,
        cs_sell, cs_buy,
        bb_sell, bb_buy,
        vp_sell, vp_buy,
        kdj_sell, kdj_buy,
        ma_arr_bonus_sell, ma_arr_bonus_buy,
    )


def _fuse_scores(mode, config,
                 div_sell, div_buy,
                 ma_sell, ma_buy,
                 cs_sell, cs_buy,
                 bb_sell, bb_buy,
                 vp_sell, vp_buy,
                 kdj_sell, kdj_buy,
                 ma_arr_bonus_sell, ma_arr_bonus_buy):
    """纯计算函数: 根据模式融合六维分数为 (sell_score, buy_score)"""

    if mode == "c6_veto_4":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        veto_threshold = config.get("veto_threshold", 25)
        kdj_bonus = config.get("kdj_bonus", 0.09)
        bb_bonus = config.get("bb_bonus", 0.10)
        vp_bonus = config.get("vp_bonus", 0.08)
        cs_bonus = config.get("cs_bonus", 0.06)
        veto_dampen = config.get("veto_dampen", 0.30)

        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy, kdj_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell, kdj_sell] if s >= veto_threshold)

        if sell_vetoes >= 2:
            sell_score = base_sell * veto_dampen
        else:
            sb = 0
            if bb_sell >= 15: sb += bb_bonus
            if vp_sell >= 15: sb += vp_bonus
            if cs_sell >= 25: sb += cs_bonus
            if kdj_sell >= 15: sb += kdj_bonus
            sell_score = base_sell * (1 + sb)

        if buy_vetoes >= 2:
            buy_score = base_buy * veto_dampen
        else:
            bb_ = 0
            if bb_buy >= 15: bb_ += bb_bonus
            if vp_buy >= 15: bb_ += vp_bonus
            if cs_buy >= 25: bb_ += cs_bonus
            if kdj_buy >= 15: bb_ += kdj_bonus
            buy_score = base_buy * (1 + bb_)

    elif mode == "c6_veto":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        veto_threshold = config.get("veto_threshold", 25)

        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell] if s >= veto_threshold)

        if sell_vetoes >= 2:
            sell_score = base_sell * 0.3
        else:
            sb = 0
            if bb_sell >= 15: sb += 0.10
            if vp_sell >= 15: sb += 0.08
            if cs_sell >= 25: sb += 0.06
            sell_score = base_sell * (1 + sb)

        if buy_vetoes >= 2:
            buy_score = base_buy * 0.3
        else:
            bb_ = 0
            if bb_buy >= 15: bb_ += 0.10
            if vp_buy >= 15: bb_ += 0.08
            if cs_buy >= 25: bb_ += 0.06
            buy_score = base_buy * (1 + bb_)

    elif mode == "kdj_weighted":
        kdj_w = config.get("kdj_weight", 0.20)
        div_w = config.get("div_weight", 0.55)
        ma_w = 1.0 - div_w - kdj_w
        base_sell = (div_sell * div_w + ma_sell * ma_w + kdj_sell * kdj_w) * ma_arr_bonus_sell
        base_buy = (div_buy * div_w + ma_buy * ma_w + kdj_buy * kdj_w) * ma_arr_bonus_buy

        sb = 0
        if bb_sell >= 15: sb += 0.08
        if vp_sell >= 15: sb += 0.06
        if cs_sell >= 25: sb += 0.05
        sell_score = base_sell * (1 + sb)

        bb_ = 0
        if bb_buy >= 15: bb_ += 0.08
        if vp_buy >= 15: bb_ += 0.06
        if cs_buy >= 25: bb_ += 0.05
        buy_score = base_buy * (1 + bb_)

    elif mode == "kdj_timing":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        kdj_strong = config.get("kdj_strong_mult", 1.25)
        kdj_normal = config.get("kdj_normal_mult", 1.12)
        kdj_reverse = config.get("kdj_reverse_mult", 0.70)

        if kdj_sell >= 30: sell_score = base_sell * kdj_strong
        elif kdj_sell >= 15: sell_score = base_sell * kdj_normal
        elif kdj_buy >= 25: sell_score = base_sell * kdj_reverse
        else: sell_score = base_sell

        if kdj_buy >= 30: buy_score = base_buy * kdj_strong
        elif kdj_buy >= 15: buy_score = base_buy * kdj_normal
        elif kdj_sell >= 25: buy_score = base_buy * kdj_reverse
        else: buy_score = base_buy

        if bb_sell >= 15: sell_score *= 1.08
        if vp_sell >= 15: sell_score *= 1.06
        if bb_buy >= 15: buy_score *= 1.08
        if vp_buy >= 15: buy_score *= 1.06

    elif mode == "kdj_gate":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        kdj_gate_threshold = config.get("kdj_gate_threshold", 10)

        if kdj_sell < kdj_gate_threshold and base_sell > 0:
            sell_score = base_sell * 0.4
        else:
            sb = 0
            if bb_sell >= 15: sb += 0.10
            if vp_sell >= 15: sb += 0.08
            if kdj_sell >= 20: sb += 0.12
            sell_score = base_sell * (1 + sb)

        if kdj_buy < kdj_gate_threshold and base_buy > 0:
            buy_score = base_buy * 0.4
        else:
            bb_ = 0
            if bb_buy >= 15: bb_ += 0.10
            if vp_buy >= 15: bb_ += 0.08
            if kdj_buy >= 20: bb_ += 0.12
            buy_score = base_buy * (1 + bb_)

    else:
        sell_score = div_sell * 0.5 + ma_sell * 0.2 + kdj_sell * 0.15 + bb_sell * 0.08 + vp_sell * 0.05 + cs_sell * 0.02
        buy_score = div_buy * 0.5 + ma_buy * 0.2 + kdj_buy * 0.15 + bb_buy * 0.08 + vp_buy * 0.05 + cs_buy * 0.02

    return sell_score, buy_score


# ──────────────────────────────────────────────────────────
# P1 优化: 批量向量化评分 (替换逐 bar 循环)
# ──────────────────────────────────────────────────────────

def _align_div_signals_to_index(div_signals, index):
    """将稀疏的背离信号 dict {timestamp: sig_dict} 前向填充对齐到目标 index。

    返回: dict[key] -> numpy array, 长度与 index 相同
    """
    if not div_signals:
        result = {}
        for k, v in DEFAULT_SIG.items():
            if isinstance(v, bool):
                result[k] = np.zeros(len(index), dtype=np.bool_)
            elif isinstance(v, (int, float)):
                result[k] = np.zeros(len(index), dtype=np.float64)
            else:
                result[k] = np.array([v] * len(index), dtype=object)
        return result

    # 获取信号时间点和数据
    sig_times = list(div_signals.keys())
    sig_data = list(div_signals.values())

    # 数值型 key 列表
    num_keys = [k for k, v in DEFAULT_SIG.items() if isinstance(v, (int, float))]
    bool_keys = [k for k, v in DEFAULT_SIG.items() if isinstance(v, bool)]

    n = len(index)
    result = {}

    # 初始化结果数组
    for k in num_keys:
        result[k] = np.zeros(n, dtype=np.float64)
    for k in bool_keys:
        result[k] = np.zeros(n, dtype=np.bool_)

    if not sig_times:
        return result

    # 用 searchsorted 做前向填充 (每个 index 位置找最近的 <= 它的信号)
    sig_times_arr = np.array(sig_times)
    index_arr = np.array(index)

    # searchsorted: 找到每个 index 在 sig_times 中的插入位置
    positions = np.searchsorted(sig_times_arr, index_arr, side='right') - 1
    # positions < 0 表示该位置之前没有信号

    valid_mask = positions >= 0
    valid_positions = positions[valid_mask]

    for k in num_keys:
        vals = np.array([s.get(k, 0) for s in sig_data], dtype=np.float64)
        result[k][valid_mask] = vals[valid_positions]

    for k in bool_keys:
        vals = np.array([s.get(k, False) for s in sig_data], dtype=np.bool_)
        result[k][valid_mask] = vals[valid_positions]

    return result


def _vectorized_top_score(merged_arrays, trend_down):
    """向量化版 _calc_top_score: 输入为 numpy 数组, 输出 numpy 数组"""
    n = len(trend_down)
    score = np.zeros(n, dtype=np.float64)

    # MACD背离
    sep_top = merged_arrays.get('sep_divs_top', np.zeros(n))
    separated_top = merged_arrays.get('separated_top', np.zeros(n))
    area_top = merged_arrays.get('area_top_div', np.zeros(n))
    dif_top = merged_arrays.get('dif_top_div', np.zeros(n))
    exhaust_sell = merged_arrays.get('exhaust_sell', np.zeros(n, dtype=np.bool_))
    zero_ret_top = merged_arrays.get('zero_returns_top', np.zeros(n))
    top = merged_arrays.get('top', np.zeros(n))

    # 双隔堆 30, 单隔堆 18 (互斥)
    mask_double = sep_top >= 2
    mask_single = (~mask_double) & ((sep_top >= 1) | (separated_top >= 1))
    score += np.where(mask_double, 30, 0)
    score += np.where(mask_single, 18, 0)

    score += np.where(area_top >= 1, 12, 0)
    score += np.where(dif_top >= 1, 8, 0)
    score += np.where(exhaust_sell, 18, 0)
    score += np.where(zero_ret_top >= 1, 10, 0)
    score += np.where(top >= 20, top * 0.2, 0)

    # 趋势加成
    score = np.where(trend_down, np.floor(score * 1.3), score)

    return score


def _vectorized_bottom_score(merged_arrays, trend_up):
    """向量化版 _calc_bottom_score"""
    n = len(trend_up)
    score = np.zeros(n, dtype=np.float64)

    sep_bot = merged_arrays.get('sep_divs_bottom', np.zeros(n))
    separated_bot = merged_arrays.get('separated_bottom', np.zeros(n))
    exhaust_buy = merged_arrays.get('exhaust_buy', np.zeros(n, dtype=np.bool_))
    area_bot = merged_arrays.get('area_bottom_div', np.zeros(n))
    bottom = merged_arrays.get('bottom', np.zeros(n))

    mask_double = sep_bot >= 2
    mask_single = (~mask_double) & ((sep_bot >= 1) | (separated_bot >= 1))
    score += np.where(mask_double, 30, 0)
    score += np.where(mask_single, 15, 0)

    score += np.where(exhaust_buy, 18, 0)
    score += np.where(area_bot >= 1, 10, 0)
    score += np.where(bottom >= 20, bottom * 0.15, 0)

    score = np.where(trend_up, np.floor(score * 1.2), score)

    return score


def _vectorized_fuse_scores(mode, config, n,
                            div_sell, div_buy,
                            ma_sell, ma_buy,
                            cs_sell, cs_buy,
                            bb_sell, bb_buy,
                            vp_sell, vp_buy,
                            kdj_sell, kdj_buy,
                            ma_arr_bonus_sell, ma_arr_bonus_buy):
    """向量化版融合评分: 输入全部为长度 n 的 numpy 数组"""

    if mode == "c6_veto_4":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        veto_threshold = config.get("veto_threshold", 25)
        kdj_b = config.get("kdj_bonus", 0.09)
        bb_b = config.get("bb_bonus", 0.10)
        vp_b = config.get("vp_bonus", 0.08)
        cs_b = config.get("cs_bonus", 0.06)
        veto_dampen = config.get("veto_dampen", 0.30)

        # 否决计数 (向量化)
        sell_vetoes = ((bb_buy >= veto_threshold).astype(np.int32) +
                       (vp_buy >= veto_threshold).astype(np.int32) +
                       (cs_buy >= veto_threshold).astype(np.int32) +
                       (kdj_buy >= veto_threshold).astype(np.int32))
        buy_vetoes = ((bb_sell >= veto_threshold).astype(np.int32) +
                      (vp_sell >= veto_threshold).astype(np.int32) +
                      (cs_sell >= veto_threshold).astype(np.int32) +
                      (kdj_sell >= veto_threshold).astype(np.int32))

        # 加成 (向量化)
        sell_bonus = ((bb_sell >= 15).astype(np.float64) * bb_b +
                      (vp_sell >= 15).astype(np.float64) * vp_b +
                      (cs_sell >= 25).astype(np.float64) * cs_b +
                      (kdj_sell >= 15).astype(np.float64) * kdj_b)
        buy_bonus = ((bb_buy >= 15).astype(np.float64) * bb_b +
                     (vp_buy >= 15).astype(np.float64) * vp_b +
                     (cs_buy >= 25).astype(np.float64) * cs_b +
                     (kdj_buy >= 15).astype(np.float64) * kdj_b)

        sell_score = np.where(sell_vetoes >= 2,
                              base_sell * veto_dampen,
                              base_sell * (1 + sell_bonus))
        buy_score = np.where(buy_vetoes >= 2,
                             base_buy * veto_dampen,
                             base_buy * (1 + buy_bonus))

    elif mode == "c6_veto":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        veto_threshold = config.get("veto_threshold", 25)

        sell_vetoes = ((bb_buy >= veto_threshold).astype(np.int32) +
                       (vp_buy >= veto_threshold).astype(np.int32) +
                       (cs_buy >= veto_threshold).astype(np.int32))
        buy_vetoes = ((bb_sell >= veto_threshold).astype(np.int32) +
                      (vp_sell >= veto_threshold).astype(np.int32) +
                      (cs_sell >= veto_threshold).astype(np.int32))

        sell_bonus = ((bb_sell >= 15).astype(np.float64) * 0.10 +
                      (vp_sell >= 15).astype(np.float64) * 0.08 +
                      (cs_sell >= 25).astype(np.float64) * 0.06)
        buy_bonus = ((bb_buy >= 15).astype(np.float64) * 0.10 +
                     (vp_buy >= 15).astype(np.float64) * 0.08 +
                     (cs_buy >= 25).astype(np.float64) * 0.06)

        sell_score = np.where(sell_vetoes >= 2, base_sell * 0.3, base_sell * (1 + sell_bonus))
        buy_score = np.where(buy_vetoes >= 2, base_buy * 0.3, base_buy * (1 + buy_bonus))

    elif mode == "kdj_weighted":
        kdj_w = config.get("kdj_weight", 0.20)
        div_w = config.get("div_weight", 0.55)
        ma_w = 1.0 - div_w - kdj_w
        base_sell = (div_sell * div_w + ma_sell * ma_w + kdj_sell * kdj_w) * ma_arr_bonus_sell
        base_buy = (div_buy * div_w + ma_buy * ma_w + kdj_buy * kdj_w) * ma_arr_bonus_buy

        sell_bonus = ((bb_sell >= 15).astype(np.float64) * 0.08 +
                      (vp_sell >= 15).astype(np.float64) * 0.06 +
                      (cs_sell >= 25).astype(np.float64) * 0.05)
        buy_bonus = ((bb_buy >= 15).astype(np.float64) * 0.08 +
                     (vp_buy >= 15).astype(np.float64) * 0.06 +
                     (cs_buy >= 25).astype(np.float64) * 0.05)

        sell_score = base_sell * (1 + sell_bonus)
        buy_score = base_buy * (1 + buy_bonus)

    elif mode == "kdj_timing":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        kdj_strong = config.get("kdj_strong_mult", 1.25)
        kdj_normal = config.get("kdj_normal_mult", 1.12)
        kdj_reverse = config.get("kdj_reverse_mult", 0.70)

        sell_mult = np.where(kdj_sell >= 30, kdj_strong,
                    np.where(kdj_sell >= 15, kdj_normal,
                    np.where(kdj_buy >= 25, kdj_reverse, 1.0)))
        buy_mult = np.where(kdj_buy >= 30, kdj_strong,
                   np.where(kdj_buy >= 15, kdj_normal,
                   np.where(kdj_sell >= 25, kdj_reverse, 1.0)))

        sell_score = base_sell * sell_mult
        buy_score = base_buy * buy_mult

        sell_score = sell_score * np.where(bb_sell >= 15, 1.08, 1.0)
        sell_score = sell_score * np.where(vp_sell >= 15, 1.06, 1.0)
        buy_score = buy_score * np.where(bb_buy >= 15, 1.08, 1.0)
        buy_score = buy_score * np.where(vp_buy >= 15, 1.06, 1.0)

    elif mode == "kdj_gate":
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        kdj_gate_threshold = config.get("kdj_gate_threshold", 10)

        sell_bonus = ((bb_sell >= 15).astype(np.float64) * 0.10 +
                      (vp_sell >= 15).astype(np.float64) * 0.08 +
                      (kdj_sell >= 20).astype(np.float64) * 0.12)
        buy_bonus = ((bb_buy >= 15).astype(np.float64) * 0.10 +
                     (vp_buy >= 15).astype(np.float64) * 0.08 +
                     (kdj_buy >= 20).astype(np.float64) * 0.12)

        sell_score = np.where(
            (kdj_sell < kdj_gate_threshold) & (base_sell > 0),
            base_sell * 0.4,
            base_sell * (1 + sell_bonus),
        )
        buy_score = np.where(
            (kdj_buy < kdj_gate_threshold) & (base_buy > 0),
            base_buy * 0.4,
            base_buy * (1 + buy_bonus),
        )

    else:
        sell_score = div_sell * 0.5 + ma_sell * 0.2 + kdj_sell * 0.15 + bb_sell * 0.08 + vp_sell * 0.05 + cs_sell * 0.02
        buy_score = div_buy * 0.5 + ma_buy * 0.2 + kdj_buy * 0.15 + bb_buy * 0.08 + vp_buy * 0.05 + cs_buy * 0.02

    return sell_score, buy_score


def calc_fusion_score_six_batch(signals, df, config, warmup=60):
    """
    P1 核心优化: 批量向量化计算整个 DataFrame 的融合评分。

    替代逐 bar 调用 calc_fusion_score_six, 预计 41s → <3s。

    返回: (score_dict, ordered_ts)
        score_dict: {timestamp: (sell_score, buy_score)}
        ordered_ts: 有序时间戳列表
    """
    n = len(df)
    if n <= warmup:
        return {}, []

    mode = config.get("fusion_mode", "c6_veto_4")
    index = df.index

    # ─── 1. 对齐背离信号到 df.index (前向填充) ───
    div_main_aligned = _align_div_signals_to_index(signals["div"], index)
    div_8h_aligned = _align_div_signals_to_index(signals.get("div_8h", {}), index)

    # ─── 2. 合并主周期和8h背离信号 ───
    num_keys = [k for k, v in DEFAULT_SIG.items() if isinstance(v, (int, float))]
    bool_keys = [k for k, v in DEFAULT_SIG.items() if isinstance(v, bool)]

    merged = {}
    # top/bottom: 加权合并
    merged['top'] = div_main_aligned.get('top', np.zeros(n)) * 1.0 + div_8h_aligned.get('top', np.zeros(n)) * 0.5
    merged['bottom'] = div_main_aligned.get('bottom', np.zeros(n)) * 1.0 + div_8h_aligned.get('bottom', np.zeros(n)) * 0.5

    for k in num_keys:
        if k in ('top', 'bottom'):
            continue
        merged[k] = np.maximum(
            div_main_aligned.get(k, np.zeros(n)),
            div_8h_aligned.get(k, np.zeros(n)),
        )
    for k in bool_keys:
        merged[k] = div_main_aligned.get(k, np.zeros(n, dtype=np.bool_)) | div_8h_aligned.get(k, np.zeros(n, dtype=np.bool_))

    # ─── 3. 向量化趋势检测 ───
    # 原始代码: c5 = df["close"].iloc[idx-5:idx].mean()  (不含当前bar)
    # 用 shift(1) 排除当前 bar, 再 rolling
    close = df['close'].values.astype(np.float64)
    close_shifted = pd.Series(close).shift(1)
    c5 = close_shifted.rolling(5, min_periods=1).mean().values
    c20 = close_shifted.rolling(20, min_periods=1).mean().values

    trend_down = np.zeros(n, dtype=np.bool_)
    trend_up = np.zeros(n, dtype=np.bool_)
    mask_30 = np.arange(n) >= 30
    trend_down[mask_30] = c5[mask_30] < c20[mask_30] * 0.99
    trend_up[mask_30] = c5[mask_30] > c20[mask_30] * 1.01

    # ─── 4. 向量化 top/bottom score ───
    div_sell = _vectorized_top_score(merged, trend_down)
    div_buy = _vectorized_bottom_score(merged, trend_up)

    # ─── 5. 提取 5 个信号 Series 为 numpy 数组 ───
    ma_sell = signals["ma"]["sell_score"].values.astype(np.float64) if hasattr(signals["ma"]["sell_score"], 'values') else np.zeros(n)
    ma_buy = signals["ma"]["buy_score"].values.astype(np.float64) if hasattr(signals["ma"]["buy_score"], 'values') else np.zeros(n)
    cs_sell = signals["cs_sell"].values.astype(np.float64) if hasattr(signals["cs_sell"], 'values') else np.zeros(n)
    cs_buy = signals["cs_buy"].values.astype(np.float64) if hasattr(signals["cs_buy"], 'values') else np.zeros(n)
    bb_sell = signals["bb_sell"].values.astype(np.float64) if hasattr(signals["bb_sell"], 'values') else np.zeros(n)
    bb_buy = signals["bb_buy"].values.astype(np.float64) if hasattr(signals["bb_buy"], 'values') else np.zeros(n)
    vp_sell = signals["vp_sell"].values.astype(np.float64) if hasattr(signals["vp_sell"], 'values') else np.zeros(n)
    vp_buy = signals["vp_buy"].values.astype(np.float64) if hasattr(signals["vp_buy"], 'values') else np.zeros(n)
    kdj_sell = signals["kdj_sell"].values.astype(np.float64) if hasattr(signals["kdj_sell"], 'values') else np.zeros(n)
    kdj_buy = signals["kdj_buy"].values.astype(np.float64) if hasattr(signals["kdj_buy"], 'values') else np.zeros(n)

    # ─── 6. MA 排列加成 ───
    ma_arr_bonus_sell = np.ones(n, dtype=np.float64)
    ma_arr_bonus_buy = np.ones(n, dtype=np.float64)
    ma_data = signals.get("ma", {})
    arr_series = ma_data.get("arrangement", None)
    if arr_series is not None and hasattr(arr_series, 'values'):
        try:
            arr_vals = arr_series.values.astype(np.float64)
            # 确保长度对齐
            min_len = min(len(arr_vals), n)
            ma_arr_bonus_sell[:min_len] = np.where(arr_vals[:min_len] < 0, 1.10, 1.0)
            ma_arr_bonus_buy[:min_len] = np.where(arr_vals[:min_len] > 0, 1.10, 1.0)
        except (ValueError, TypeError):
            pass

    # ─── 7. 向量化融合 ───
    sell_scores, buy_scores = _vectorized_fuse_scores(
        mode, config, n,
        div_sell, div_buy,
        ma_sell, ma_buy,
        cs_sell, cs_buy,
        bb_sell, bb_buy,
        vp_sell, vp_buy,
        kdj_sell, kdj_buy,
        ma_arr_bonus_sell, ma_arr_bonus_buy,
    )

    # ─── 8. 构建输出字典 (只保留 warmup 之后的) ───
    score_dict = {}
    ordered_ts = []
    for i in range(warmup, n):
        dt = index[i]
        score_dict[dt] = (float(sell_scores[i]), float(buy_scores[i]))
        ordered_ts.append(dt)

    return score_dict, ordered_ts
