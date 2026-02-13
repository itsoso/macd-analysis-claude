"""
多周期联合决策共识算法 (共享模块)

三层判断:
  1. 加权得分: 大周期权重远高于小周期 (24h=28, 15m=3)
  2. 连续共振链: 检测相邻周期连续同向的链条 (如 15m→30m→1h)
  3. 大周期定调: ≥4h 的周期单独统计，作为趋势基调

决策矩阵:
  - 大小同向 + 共振链 → 强信号，可入场
  - 大周期有方向 + 小周期反向 → 等待，不逆势
  - 小周期有方向 + 大周期中性 → 弱信号，轻仓或观望
  - 多空分歧 → 观望

使用者:
  - live_runner.py (实时信号检测 CLI)
  - live_trading_engine.py (实盘交易引擎)
  - optimize_six_book.py (回测引擎)
"""

# ================================================================
# 时间框架常量
# ================================================================

# 时间框架排序（从小到大）
TF_ORDER = [
    '1m', '3m', '5m', '10m', '15m', '30m',
    '1h', '2h', '3h', '4h', '6h', '8h',
    '12h', '16h', '24h', '1d',
]

# 各时间框架权重 (大周期权重远高于小周期)
TF_WEIGHT = {
    '1m': 1, '3m': 1, '5m': 1,
    '10m': 2, '15m': 3, '30m': 5,
    '1h': 8, '2h': 10, '3h': 12,
    '4h': 15, '6h': 18, '8h': 20,
    '12h': 22, '16h': 25, '24h': 28, '1d': 28,
}

# 各时间框架对应分钟数
TF_MINUTES = {
    '1m': 1, '3m': 3, '5m': 5, '10m': 10, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '3h': 180, '4h': 240, '6h': 360,
    '8h': 480, '12h': 720, '16h': 960, '24h': 1440, '1d': 1440,
}

# 默认决策周期组合
DEFAULT_DECISION_TFS = ['15m', '30m', '1h', '4h', '8h', '24h']

# 大周期阈值 (≥4h 视为大周期)
LARGE_TF_THRESHOLD_MIN = 240


# ================================================================
# 核心共识算法
# ================================================================

def compute_weighted_consensus(results, timeframes=None):
    """
    智能多周期加权共识算法

    参数:
        results: list[dict] - 各TF的信号结果
            每个 dict 至少包含: {"tf": "1h", "ok": True, "action": "OPEN_LONG"}
        timeframes: list[str] - 参与决策的时间框架列表 (可选)

    返回:
        dict - 包含 decision, weighted_scores, resonance_chains, large_tf_signal 等
    """
    # 按时间框架从小到大排序
    ok_results = [r for r in results if r.get("ok")]
    ok_results.sort(key=lambda r: TF_ORDER.index(r["tf"]) if r["tf"] in TF_ORDER else 99)

    long_tfs = [r["tf"] for r in ok_results if "LONG" in r.get("action", "")]
    short_tfs = [r["tf"] for r in ok_results if "SHORT" in r.get("action", "")]
    hold_tfs = [r["tf"] for r in ok_results if r.get("action") == "HOLD"]
    n_ok = len(ok_results)

    # ── 1. 加权得分 ──
    long_score = sum(TF_WEIGHT.get(tf, 5) for tf in long_tfs)
    short_score = sum(TF_WEIGHT.get(tf, 5) for tf in short_tfs)
    total_weight = sum(TF_WEIGHT.get(r["tf"], 5) for r in ok_results)
    # 归一化到 0~100
    long_pct = round(long_score / total_weight * 100, 1) if total_weight > 0 else 0
    short_pct = round(short_score / total_weight * 100, 1) if total_weight > 0 else 0
    net_score = round(long_pct - short_pct, 1)

    weighted_scores = {
        "long": long_pct,
        "short": short_pct,
        "net": net_score,
        "long_raw": long_score,
        "short_raw": short_score,
        "total_weight": total_weight,
    }

    # ── 2. 连续共振链检测 ──
    resonance_chains = _detect_resonance_chains(ok_results)

    # ── 3. 大周期定调 (≥4h) ──
    large_tf_signal = _compute_large_tf_signal(ok_results)

    # 小周期方向 (<4h)
    small_long = [r["tf"] for r in ok_results
                  if "LONG" in r.get("action", "") and TF_MINUTES.get(r["tf"], 0) < LARGE_TF_THRESHOLD_MIN]
    small_short = [r["tf"] for r in ok_results
                   if "SHORT" in r.get("action", "") and TF_MINUTES.get(r["tf"], 0) < LARGE_TF_THRESHOLD_MIN]

    # ── 4. 综合决策 ──
    best_chain = resonance_chains[0] if resonance_chains else None
    decision = _make_decision(
        weighted_scores, best_chain, large_tf_signal,
        long_tfs, short_tfs, hold_tfs,
        small_long, small_short, n_ok
    )

    return {
        "long_tfs": long_tfs,
        "short_tfs": short_tfs,
        "hold_tfs": hold_tfs,
        "weighted_scores": weighted_scores,
        "resonance_chains": resonance_chains,
        "large_tf_signal": large_tf_signal,
        "decision": decision,
        # 兼容旧格式
        "long": long_tfs,
        "short": short_tfs,
        "hold": hold_tfs,
        "direction": decision["direction"],
    }


def _detect_resonance_chains(ok_results):
    """检测连续共振链"""
    resonance_chains = []
    n_ok = len(ok_results)
    if n_ok < 2:
        return resonance_chains

    # 为每个结果标记方向
    directions = []
    for r in ok_results:
        if "LONG" in r.get("action", ""):
            directions.append(("long", r["tf"]))
        elif "SHORT" in r.get("action", ""):
            directions.append(("short", r["tf"]))
        else:
            directions.append(("hold", r["tf"]))

    # 扫描连续同向链（只看 long/short，允许1个 hold 间隔）
    for target_dir in ["long", "short"]:
        chain = []
        gap_count = 0
        for d, tf in directions:
            if d == target_dir:
                chain.append(tf)
                gap_count = 0
            elif d == "hold" and chain and gap_count == 0:
                gap_count += 1
                continue
            else:
                if len(chain) >= 2:
                    has_4h = any(TF_MINUTES.get(t, 0) >= LARGE_TF_THRESHOLD_MIN for t in chain)
                    resonance_chains.append({
                        "direction": target_dir,
                        "chain": chain,
                        "length": len(chain),
                        "has_4h_plus": has_4h,
                        "weight": sum(TF_WEIGHT.get(t, 5) for t in chain),
                    })
                chain = []
                gap_count = 0
                if d == target_dir:
                    chain = [tf]

        # 末尾收尾
        if len(chain) >= 2:
            has_4h = any(TF_MINUTES.get(t, 0) >= LARGE_TF_THRESHOLD_MIN for t in chain)
            resonance_chains.append({
                "direction": target_dir,
                "chain": chain,
                "length": len(chain),
                "has_4h_plus": has_4h,
                "weight": sum(TF_WEIGHT.get(t, 5) for t in chain),
            })

    # 按权重排序
    resonance_chains.sort(key=lambda c: c["weight"], reverse=True)
    return resonance_chains


def _compute_large_tf_signal(ok_results):
    """计算大周期 (≥4h) 方向"""
    large_long = [r["tf"] for r in ok_results
                  if "LONG" in r.get("action", "") and TF_MINUTES.get(r["tf"], 0) >= LARGE_TF_THRESHOLD_MIN]
    large_short = [r["tf"] for r in ok_results
                   if "SHORT" in r.get("action", "") and TF_MINUTES.get(r["tf"], 0) >= LARGE_TF_THRESHOLD_MIN]

    if large_long and not large_short:
        return {"direction": "long", "tfs": large_long}
    elif large_short and not large_long:
        return {"direction": "short", "tfs": large_short}
    elif large_long and large_short:
        return {"direction": "conflict", "tfs": large_long + large_short}
    else:
        return {"direction": "neutral", "tfs": []}


def _make_decision(ws, best_chain, large_sig, long_tfs, short_tfs, hold_tfs,
                   small_long, small_short, n_ok):
    """
    决策矩阵 — 综合加权得分、共振链、大周期方向

    优先级:
      1. 大周期+小周期同向+共振链 → 强入场 (strength 70-100)
      2. 大周期明确+小周期同向(无共振链) → 中等入场 (strength 50-70)
      3. 只有小周期信号+大周期中性 → 弱/观望 (strength 20-40)
      4. 大周期与小周期反向 → 不做 (strength 0-15)
      5. 完全中性 → 观望 (strength 0)
    """
    net = ws["net"]
    large_dir = large_sig["direction"]

    # ---------- 情况 A: 大小同向 + 共振链 → 强信号 ----------
    if best_chain and best_chain["has_4h_plus"] and best_chain["length"] >= 3:
        direction = best_chain["direction"]
        if (direction == "long" and large_dir in ("long", "neutral") and net > 15) or \
           (direction == "short" and large_dir in ("short", "neutral") and net < -15):
            strength = min(100, 50 + best_chain["weight"] * 0.5 + abs(net) * 0.3)
            label_dir = "做多" if direction == "long" else "做空"
            return {
                "direction": direction,
                "label": f"🔥 强{label_dir}共振",
                "strength": round(strength),
                "reason": (f"连续{best_chain['length']}级共振"
                           f"({' → '.join(best_chain['chain'])}), "
                           f"大周期{large_dir}, 加权净分{net:+.1f}"),
                "actionable": True,
            }

    # ---------- 情况 B: 大周期明确 + 小周期同向 → 中等信号 ----------
    if large_dir == "long" and small_long and not small_short:
        strength = min(80, 40 + net * 0.5)
        return {
            "direction": "long",
            "label": "📈 大周期看多 + 小周期确认",
            "strength": round(max(strength, 40)),
            "reason": f"大周期({','.join(large_sig['tfs'])})看多, "
                      f"小周期({','.join(small_long)})确认, 净分{net:+.1f}",
            "actionable": True,
        }
    if large_dir == "short" and small_short and not small_long:
        strength = min(80, 40 + abs(net) * 0.5)
        return {
            "direction": "short",
            "label": "📉 大周期看空 + 小周期确认",
            "strength": round(max(strength, 40)),
            "reason": f"大周期({','.join(large_sig['tfs'])})看空, "
                      f"小周期({','.join(small_short)})确认, 净分{net:+.1f}",
            "actionable": True,
        }

    # ---------- 情况 C: 大小周期反向 → 不做 ----------
    if large_dir == "long" and small_short and not small_long:
        return {
            "direction": "hold",
            "label": "⛔ 大小周期反向 — 不做",
            "strength": round(max(0, 10 - abs(net) * 0.1)),
            "reason": f"大周期看多但小周期({','.join(small_short)})看空, "
                      f"可能是回调, 等小周期转向后再顺势做多",
            "actionable": False,
        }
    if large_dir == "short" and small_long and not small_short:
        return {
            "direction": "hold",
            "label": "⛔ 大小周期反向 — 不做",
            "strength": round(max(0, 10 - abs(net) * 0.1)),
            "reason": f"大周期看空但小周期({','.join(small_long)})看多, "
                      f"可能是反弹, 等小周期转向后再顺势做空",
            "actionable": False,
        }

    # ---------- 情况 D: 多空同时存在 → 观望 ----------
    if long_tfs and short_tfs:
        return {
            "direction": "hold",
            "label": "⚠️ 多空分歧 — 观望",
            "strength": round(min(15, abs(net))),
            "reason": f"做多({','.join(long_tfs)}) vs 做空({','.join(short_tfs)}), "
                      f"方向不明确, 净分{net:+.1f}, 等待分歧解除",
            "actionable": False,
        }

    # ---------- 情况 E: 只有小周期信号 + 大周期中性 → 弱信号 ----------
    if small_long and large_dir == "neutral":
        chain_bonus = best_chain["length"] * 5 if best_chain and best_chain["direction"] == "long" else 0
        strength = min(40, 15 + net * 0.3 + chain_bonus)
        return {
            "direction": "long",
            "label": "📊 小周期看多 — 弱信号",
            "strength": round(max(strength, 10)),
            "reason": f"仅小周期({','.join(small_long)})看多, "
                      f"大周期中性, 可轻仓试探或等待大周期确认",
            "actionable": False,
        }
    if small_short and large_dir == "neutral":
        chain_bonus = best_chain["length"] * 5 if best_chain and best_chain["direction"] == "short" else 0
        strength = min(40, 15 + abs(net) * 0.3 + chain_bonus)
        return {
            "direction": "short",
            "label": "📊 小周期看空 — 弱信号",
            "strength": round(max(strength, 10)),
            "reason": f"仅小周期({','.join(small_short)})看空, "
                      f"大周期中性, 可轻仓试探或等待大周期确认",
            "actionable": False,
        }

    # ---------- 情况 F: 完全中性 ----------
    return {
        "direction": "hold",
        "label": "⚪ 中性 — 无信号",
        "strength": 0,
        "reason": f"全部{len(hold_tfs)}个周期观望, 市场无方向, 耐心等待",
        "actionable": False,
    }


# ================================================================
# 统一连续分数融合 (回测 + 实盘共用)
# ================================================================

def fuse_tf_scores(tf_scores, decision_tfs, config=None):
    """
    统一的多周期连续分数融合算法 — 回测与实盘共用一套逻辑。

    替代旧实盘路径 (先离散化为 OPEN_LONG/SHORT 再加权) 和
    旧回测路径 (optimize_six_book.calc_multi_tf_consensus 内联逻辑)。

    参数:
        tf_scores : dict[str, tuple[float, float]]
            每个TF的 (sell_score, buy_score), 值域 0~100。
            如果某个TF计算失败, 可以不包含在 tf_scores 中。
        decision_tfs : list[str]
            期望参与决策的TF列表 (用于计算 coverage)。
        config : dict | None
            可选配置, 支持的 key:
              - short_threshold (default 25)
              - long_threshold  (default 40)
              - coverage_min    (default 0.5) — 低于此值时 fail-closed

    返回:
        dict — 结构:
        {
            "weighted_ss": float,
            "weighted_bs": float,
            "tf_scores":   {tf: {"ss": .., "bs": .., "dir": ..}},
            "coverage":    float (0~1),
            "decision":    {direction, strength, actionable, label, reason},
            "meta":        {chain_len, chain_dir, chain_has_4h, large_ss, large_bs,
                            small_ss, small_bs},
            # 兼容旧 compute_weighted_consensus 格式
            "long_tfs":  [...],
            "short_tfs": [...],
            "hold_tfs":  [...],
            "weighted_scores": {...},
            "resonance_chains": [...],
            "large_tf_signal":  {...},
            "direction": str,
        }
    """
    config = config or {}
    short_threshold = config.get('short_threshold', 25)
    long_threshold = config.get('long_threshold', 40)
    coverage_min = config.get('coverage_min', 0.5)
    # 可配置融合阈值 (原硬编码, 现参数化)
    dominance_ratio = float(config.get('dominance_ratio', 1.3))  # 方向判定主导比
    chain_boost_per_tf = float(config.get('chain_boost_per_tf', 0.08))  # 共振链每级增强
    chain_boost_weak_per_tf = float(config.get('chain_boost_weak_per_tf', 0.04))  # 弱共振链每级增强

    # ── 0. Coverage (有效TF占比) ──
    available_tfs = [tf for tf in decision_tfs if tf in tf_scores]
    target_weight = sum(TF_WEIGHT.get(tf, 5) for tf in decision_tfs)
    actual_weight = sum(TF_WEIGHT.get(tf, 5) for tf in available_tfs)
    coverage = actual_weight / target_weight if target_weight > 0 else 0.0

    if not available_tfs:
        return _empty_consensus(decision_tfs, coverage)

    # ── 1. 加权平均 ss/bs (支持 regime 驱动动态权重) ──
    # regime_label 由上游 _compute_regime_controls 计算后通过 config 传入
    regime_label = config.get('_regime_label', 'neutral')
    effective_weights = {}
    for tf in available_tfs:
        base_w = TF_WEIGHT.get(tf, 5)
        tf_min = TF_MINUTES.get(tf, 60)
        if regime_label in ('high_vol_choppy', 'high_vol'):
            # 高波动/震荡: 降低小周期权重(噪声大), 维持大周期
            if tf_min < 60:
                base_w *= 0.6
            elif tf_min >= 240:
                base_w *= 1.15
        elif regime_label in ('trend', 'low_vol_trend'):
            # 趋势市场: 抬高大周期权重, 略降小周期
            if tf_min >= 240:
                base_w *= 1.25
            elif tf_min < 60:
                base_w *= 0.8
        effective_weights[tf] = base_w

    total_w = sum(effective_weights[tf] for tf in available_tfs)
    weighted_ss = sum(tf_scores[tf][0] * effective_weights[tf]
                      for tf in available_tfs) / total_w
    weighted_bs = sum(tf_scores[tf][1] * effective_weights[tf]
                      for tf in available_tfs) / total_w

    # ── 2. 每个TF的方向判定 (用于共振/决策矩阵) ──
    sorted_tfs = sorted(available_tfs,
                        key=lambda t: TF_MINUTES.get(t, 0))
    directions = []          # [(tf, 'long'|'short'|'hold')]
    long_tfs, short_tfs, hold_tfs = [], [], []
    tf_detail = {}           # 用于输出

    for tf in sorted_tfs:
        ss, bs = tf_scores[tf]
        if ss >= short_threshold and ss > bs * dominance_ratio:
            d = 'short'
            short_tfs.append(tf)
        elif bs >= long_threshold and bs > ss * dominance_ratio:
            d = 'long'
            long_tfs.append(tf)
        else:
            d = 'hold'
            hold_tfs.append(tf)
        directions.append((tf, d))
        tf_detail[tf] = {"ss": round(ss, 1), "bs": round(bs, 1), "dir": d}

    # ── 3. 大/小周期分析 ──
    large_tfs = [tf for tf in available_tfs
                 if TF_MINUTES.get(tf, 0) >= LARGE_TF_THRESHOLD_MIN]
    small_tfs = [tf for tf in available_tfs
                 if TF_MINUTES.get(tf, 0) < LARGE_TF_THRESHOLD_MIN]

    large_ss_avg = large_bs_avg = 0.0
    if large_tfs:
        lw = sum(TF_WEIGHT.get(tf, 5) for tf in large_tfs)
        large_ss_avg = sum(tf_scores[tf][0] * TF_WEIGHT.get(tf, 5)
                           for tf in large_tfs) / lw
        large_bs_avg = sum(tf_scores[tf][1] * TF_WEIGHT.get(tf, 5)
                           for tf in large_tfs) / lw

    small_ss_avg = small_bs_avg = 0.0
    if small_tfs:
        sw = sum(TF_WEIGHT.get(tf, 5) for tf in small_tfs)
        small_ss_avg = sum(tf_scores[tf][0] * TF_WEIGHT.get(tf, 5)
                           for tf in small_tfs) / sw
        small_bs_avg = sum(tf_scores[tf][1] * TF_WEIGHT.get(tf, 5)
                           for tf in small_tfs) / sw

    # ── 4. 大小周期反向衰减 (与回测一致) ──
    if large_tfs and small_tfs:
        # 大周期偏空 + 小周期偏多 → 衰减买入
        if large_ss_avg > large_bs_avg * dominance_ratio and small_bs_avg > small_ss_avg * dominance_ratio:
            weighted_bs *= 0.5
            weighted_ss *= 1.15
        # 大周期偏多 + 小周期偏空 → 衰减卖出
        elif large_bs_avg > large_ss_avg * dominance_ratio and small_ss_avg > small_bs_avg * dominance_ratio:
            weighted_ss *= 0.5
            weighted_bs *= 1.15

    # ── 5. 共振链检测 + 增强 (与回测一致) ──
    best_chain_len = 0
    best_chain_dir = 'hold'
    best_chain_has_4h = False
    best_chain_tfs = []

    for target in ['long', 'short']:
        chain = []
        gap = 0
        for tf, d in directions:
            if d == target:
                chain.append(tf)
                gap = 0
            elif d == 'hold' and chain and gap == 0:
                gap += 1
                continue
            else:
                if len(chain) > best_chain_len:
                    best_chain_len = len(chain)
                    best_chain_dir = target
                    best_chain_has_4h = any(
                        TF_MINUTES.get(t, 0) >= LARGE_TF_THRESHOLD_MIN for t in chain)
                    best_chain_tfs = list(chain)
                chain = []
                gap = 0
                if d == target:
                    chain = [tf]
        if len(chain) > best_chain_len:
            best_chain_len = len(chain)
            best_chain_dir = target
            best_chain_has_4h = any(
                TF_MINUTES.get(t, 0) >= LARGE_TF_THRESHOLD_MIN for t in chain)
            best_chain_tfs = list(chain)

    if best_chain_len >= 3 and best_chain_has_4h:
        chain_boost = 1.0 + best_chain_len * chain_boost_per_tf
        if best_chain_dir == 'short':
            weighted_ss *= chain_boost
        else:
            weighted_bs *= chain_boost
    elif best_chain_len >= 2:
        chain_boost = 1.0 + best_chain_len * chain_boost_weak_per_tf
        if best_chain_dir == 'short':
            weighted_ss *= chain_boost
        else:
            weighted_bs *= chain_boost

    # ── 6. Coverage 惩罚 ──
    if coverage < 1.0:
        # 线性衰减: coverage=0.5 → 信号打5折
        weighted_ss *= coverage
        weighted_bs *= coverage

    weighted_ss = float(weighted_ss)
    weighted_bs = float(weighted_bs)

    # ── 7. 生成决策 (复用决策矩阵) ──
    # 构造兼容旧格式的 weighted_scores 和 large_tf_signal
    long_score = sum(TF_WEIGHT.get(tf, 5) for tf in long_tfs)
    short_score = sum(TF_WEIGHT.get(tf, 5) for tf in short_tfs)
    long_pct = round(long_score / total_w * 100, 1) if total_w > 0 else 0
    short_pct = round(short_score / total_w * 100, 1) if total_w > 0 else 0
    net_score = round(long_pct - short_pct, 1)
    weighted_scores = {
        "long": long_pct, "short": short_pct, "net": net_score,
        "long_raw": long_score, "short_raw": short_score,
        "total_weight": total_w,
    }

    large_tf_signal = _compute_large_tf_signal(
        [{"tf": tf, "action": ("OPEN_LONG" if d == "long"
                                else "OPEN_SHORT" if d == "short"
                                else "HOLD")}
         for tf, d in directions
         if TF_MINUTES.get(tf, 0) >= LARGE_TF_THRESHOLD_MIN]
    )

    small_long = [tf for tf, d in directions
                  if d == 'long' and TF_MINUTES.get(tf, 0) < LARGE_TF_THRESHOLD_MIN]
    small_short = [tf for tf, d in directions
                   if d == 'short' and TF_MINUTES.get(tf, 0) < LARGE_TF_THRESHOLD_MIN]

    best_chain_obj = None
    if best_chain_len >= 2:
        best_chain_obj = {
            "direction": best_chain_dir,
            "chain": best_chain_tfs,
            "length": best_chain_len,
            "has_4h_plus": best_chain_has_4h,
            "weight": sum(TF_WEIGHT.get(t, 5) for t in best_chain_tfs),
        }

    resonance_chains = [best_chain_obj] if best_chain_obj else []

    decision = _make_decision(
        weighted_scores, best_chain_obj, large_tf_signal,
        long_tfs, short_tfs, hold_tfs,
        small_long, small_short, len(available_tfs),
    )

    # Coverage 不足时强制 fail-closed
    if coverage < coverage_min:
        decision = {
            "direction": "hold",
            "label": f"⛔ 覆盖不足 ({coverage:.0%} < {coverage_min:.0%})",
            "strength": 0,
            "reason": (f"仅 {len(available_tfs)}/{len(decision_tfs)} 个TF有数据, "
                       f"覆盖率 {coverage:.0%}, 低于最低要求 {coverage_min:.0%}"),
            "actionable": False,
        }

    meta = {
        'chain_len': best_chain_len,
        'chain_dir': best_chain_dir,
        'chain_has_4h': best_chain_has_4h,
        'large_ss': round(large_ss_avg, 1),
        'large_bs': round(large_bs_avg, 1),
        'small_ss': round(small_ss_avg, 1),
        'small_bs': round(small_bs_avg, 1),
    }

    return {
        # 连续分数 (回测直接用)
        "weighted_ss": weighted_ss,
        "weighted_bs": weighted_bs,
        # 详细TF级分数
        "tf_scores": tf_detail,
        # Coverage
        "coverage": round(coverage, 3),
        # 决策 (实盘门控用)
        "decision": decision,
        "meta": meta,
        # 兼容旧 compute_weighted_consensus 格式
        "long_tfs": long_tfs,
        "short_tfs": short_tfs,
        "hold_tfs": hold_tfs,
        "weighted_scores": weighted_scores,
        "resonance_chains": resonance_chains,
        "large_tf_signal": large_tf_signal,
        "direction": decision["direction"],
        # 兼容旧 key
        "long": long_tfs,
        "short": short_tfs,
        "hold": hold_tfs,
    }


def _empty_consensus(decision_tfs, coverage):
    """无可用TF时的空共识"""
    return {
        "weighted_ss": 0.0,
        "weighted_bs": 0.0,
        "tf_scores": {},
        "coverage": round(coverage, 3),
        "decision": {
            "direction": "hold",
            "label": "⛔ 无可用数据",
            "strength": 0,
            "reason": f"0/{len(decision_tfs)} 个TF有数据, 无法决策",
            "actionable": False,
        },
        "meta": {},
        "long_tfs": [], "short_tfs": [], "hold_tfs": [],
        "weighted_scores": {"long": 0, "short": 0, "net": 0,
                            "long_raw": 0, "short_raw": 0, "total_weight": 0},
        "resonance_chains": [],
        "large_tf_signal": {"direction": "neutral", "tfs": []},
        "direction": "hold",
        "long": [], "short": [], "hold": [],
    }
