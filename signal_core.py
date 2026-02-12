"""
六书信号核心模块

目的:
1. 统一研究/回测/实盘的信号计算实现
2. 避免实盘模块直接依赖超大优化脚本
"""

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
    """
    # ---- 尾部截断优化 ----
    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()

    signals = {}

    # 1. 背离信号(主周期)
    lookback = max(60, min(200, len(df) // 3))
    div_signals = analyze_signals_enhanced(df, lookback)
    signals["div"] = div_signals

    # 8h辅助背离(如果可用)
    signals["div_8h"] = {}
    if "8h" in data_all and tf not in ("8h", "12h", "16h", "24h"):
        signals["div_8h"] = analyze_signals_enhanced(data_all["8h"], 90)

    # 2. 均线信号
    ma_signals = compute_ma_signals(df, timeframe=tf)
    signals["ma"] = ma_signals

    # 3. 蜡烛图
    cs_sell, cs_buy, _cs_names = compute_candlestick_scores(df)
    signals["cs_sell"] = cs_sell
    signals["cs_buy"] = cs_buy

    # 4. 布林带
    bb_sell, bb_buy, _bb_names = compute_bollinger_scores(df)
    signals["bb_sell"] = bb_sell
    signals["bb_buy"] = bb_buy

    # 5. 量价
    vp_sell, vp_buy, _vp_names = compute_volume_price_scores(df)
    signals["vp_sell"] = vp_sell
    signals["vp_buy"] = vp_buy

    # 6. KDJ
    kdj_sell, kdj_buy, _kdj_names = compute_kdj_scores(df)
    signals["kdj_sell"] = kdj_sell
    signals["kdj_buy"] = kdj_buy

    return signals


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

    if mode == "c6_veto_4":
        # === C6底座 + 四书否决权 ===
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        veto_threshold = config.get("veto_threshold", 25)
        kdj_bonus = config.get("kdj_bonus", 0.09)
        bb_bonus = config.get("bb_bonus", 0.10)
        vp_bonus = config.get("vp_bonus", 0.08)
        cs_bonus = config.get("cs_bonus", 0.06)

        # 否决逻辑: 4个辅助系统, 至少2个强烈反对 → 削弱
        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy, kdj_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell, kdj_sell] if s >= veto_threshold)

        veto_dampen = config.get("veto_dampen", 0.30)

        if sell_vetoes >= 2:
            sell_score = base_sell * veto_dampen
        else:
            sb = 0
            if bb_sell >= 15:
                sb += bb_bonus
            if vp_sell >= 15:
                sb += vp_bonus
            if cs_sell >= 25:
                sb += cs_bonus
            if kdj_sell >= 15:
                sb += kdj_bonus
            sell_score = base_sell * (1 + sb)

        if buy_vetoes >= 2:
            buy_score = base_buy * veto_dampen
        else:
            bb_ = 0
            if bb_buy >= 15:
                bb_ += bb_bonus
            if vp_buy >= 15:
                bb_ += vp_bonus
            if cs_buy >= 25:
                bb_ += cs_bonus
            if kdj_buy >= 15:
                bb_ += kdj_bonus
            buy_score = base_buy * (1 + bb_)

    elif mode == "c6_veto":
        # 五书兼容模式(无KDJ) — 用作对照
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        veto_threshold = config.get("veto_threshold", 25)

        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell] if s >= veto_threshold)

        if sell_vetoes >= 2:
            sell_score = base_sell * 0.3
        else:
            sb = 0
            if bb_sell >= 15:
                sb += 0.10
            if vp_sell >= 15:
                sb += 0.08
            if cs_sell >= 25:
                sb += 0.06
            sell_score = base_sell * (1 + sb)

        if buy_vetoes >= 2:
            buy_score = base_buy * 0.3
        else:
            bb_ = 0
            if bb_buy >= 15:
                bb_ += 0.10
            if vp_buy >= 15:
                bb_ += 0.08
            if cs_buy >= 25:
                bb_ += 0.06
            buy_score = base_buy * (1 + bb_)

    elif mode == "kdj_weighted":
        # KDJ加权模式: KDJ参与基础分计算
        kdj_w = config.get("kdj_weight", 0.20)
        div_w = config.get("div_weight", 0.55)
        ma_w = 1.0 - div_w - kdj_w  # 剩余给均线
        base_sell = (div_sell * div_w + ma_sell * ma_w + kdj_sell * kdj_w) * ma_arr_bonus_sell
        base_buy = (div_buy * div_w + ma_buy * ma_w + kdj_buy * kdj_w) * ma_arr_bonus_buy

        sb = 0
        if bb_sell >= 15:
            sb += 0.08
        if vp_sell >= 15:
            sb += 0.06
        if cs_sell >= 25:
            sb += 0.05
        sell_score = base_sell * (1 + sb)

        bb_ = 0
        if bb_buy >= 15:
            bb_ += 0.08
        if vp_buy >= 15:
            bb_ += 0.06
        if cs_buy >= 25:
            bb_ += 0.05
        buy_score = base_buy * (1 + bb_)

    elif mode == "kdj_timing":
        # KDJ择时模式: C6基础上KDJ做乘法确认
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        kdj_strong = config.get("kdj_strong_mult", 1.25)
        kdj_normal = config.get("kdj_normal_mult", 1.12)
        kdj_reverse = config.get("kdj_reverse_mult", 0.70)

        if kdj_sell >= 30:
            sell_score = base_sell * kdj_strong
        elif kdj_sell >= 15:
            sell_score = base_sell * kdj_normal
        elif kdj_buy >= 25:
            sell_score = base_sell * kdj_reverse
        else:
            sell_score = base_sell

        if kdj_buy >= 30:
            buy_score = base_buy * kdj_strong
        elif kdj_buy >= 15:
            buy_score = base_buy * kdj_normal
        elif kdj_sell >= 25:
            buy_score = base_buy * kdj_reverse
        else:
            buy_score = base_buy

        if bb_sell >= 15:
            sell_score *= 1.08
        if vp_sell >= 15:
            sell_score *= 1.06
        if bb_buy >= 15:
            buy_score *= 1.08
        if vp_buy >= 15:
            buy_score *= 1.06

    elif mode == "kdj_gate":
        # KDJ门控模式: KDJ不满足条件直接不开仓
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        kdj_gate_threshold = config.get("kdj_gate_threshold", 10)

        # KDJ必须同向才能开仓
        if kdj_sell < kdj_gate_threshold and base_sell > 0:
            sell_score = base_sell * 0.4  # KDJ不配合, 大幅削弱
        else:
            sb = 0
            if bb_sell >= 15:
                sb += 0.10
            if vp_sell >= 15:
                sb += 0.08
            if kdj_sell >= 20:
                sb += 0.12
            sell_score = base_sell * (1 + sb)

        if kdj_buy < kdj_gate_threshold and base_buy > 0:
            buy_score = base_buy * 0.4
        else:
            bb_ = 0
            if bb_buy >= 15:
                bb_ += 0.10
            if vp_buy >= 15:
                bb_ += 0.08
            if kdj_buy >= 20:
                bb_ += 0.12
            buy_score = base_buy * (1 + bb_)

    else:
        # fallback
        sell_score = div_sell * 0.5 + ma_sell * 0.2 + kdj_sell * 0.15 + bb_sell * 0.08 + vp_sell * 0.05 + cs_sell * 0.02
        buy_score = div_buy * 0.5 + ma_buy * 0.2 + kdj_buy * 0.15 + bb_buy * 0.08 + vp_buy * 0.05 + cs_buy * 0.02

    return sell_score, buy_score
